# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.ko.graph_utils import NEMO_DIGIT, GraphFst, delete_space, insert_space
from nemo_text_processing.text_normalization.ko.utils import get_abs_path, load_labels


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money.
    Produces tokens like:
      money { integer_part: "삼백오십" currency_maj: "원" }
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="classify", deterministic=deterministic)

        graph_cardinal = cardinal.graph
        SP = pynini.closure(delete_space)  # absorb any amount of spaces in input

        # --- Numbers (integer / optional minor) ---
        # Integer part: "0" or a non-zero leading digit; allow commas (e.g., 18,925,000)
        integer_part_fst = ((NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT | pynutil.delete(","))) | NEMO_DIGIT

        # Plain integer → integer_part: "<Korean number>"
        graph_integer_plain = (
            pynutil.insert('integer_part: "') + (integer_part_fst @ graph_cardinal) + pynutil.insert('" ')
        )

        # Optional 2-digit decimal (kept as minor_part if ever used downstream)
        decimal_part_fst = NEMO_DIGIT + NEMO_DIGIT
        graph_minor = pynutil.insert('minor_part: "') + (decimal_part_fst @ graph_cardinal) + pynutil.insert('" ')

        # Integer with scale suffix (만/억/조) → wrap the whole thing in one integer_part
        scale_unit = pynini.union("만", "억", "조")
        value_with_scale = (integer_part_fst @ graph_cardinal) + scale_unit
        graph_integer_with_suffix = (
            pynutil.insert('integer_part: "') + value_with_scale + pynutil.insert('" ')
        ).optimize()

        # Integer (+ optional ".<2-digit>" minor)
        number_component_plain = graph_integer_plain + pynini.closure(pynutil.delete(".") + graph_minor, 0, 1)
        number_component = (graph_integer_with_suffix | number_component_plain).optimize()

        # --- Currency (prefix or suffix) ---
        # currency_major.tsv example:
        #   ₩    원
        #   KRW  원
        #   원   원
        maj_labels = load_labels(get_abs_path("data/money/currency_major.tsv"))

        # Prefix currency (e.g., ₩, KRW): emit currency_maj then number
        currency_major_prepended = pynini.union(
            *[pynutil.delete(surface) + pynutil.insert(f'currency_maj: "{unit}" ') for surface, unit in maj_labels]
        ).optimize()

        # Suffix currency (e.g., ...원, ...달러): convert unit literal to currency_maj
        currency_major_appended = pynini.union(
            *[pynutil.delete(unit) + pynutil.insert(f'currency_maj: "{unit}" ') for _, unit in maj_labels]
        ).optimize()

        # --- Compose (NO period handling) ---
        # NOTE: We deliberately do NOT consume '/월', '/년', '/주', '/일', '/시간' here.
        # If present in the raw text, they remain outside the money token and can be handled upstream/elsewhere.

        # [currency] [number]
        graph_prepend = (currency_major_prepended + SP + number_component).optimize()

        # [number] [currency]
        graph_append = (number_component + currency_major_appended).optimize()

        graph = (graph_prepend | graph_append).optimize()

        self.fst = self.add_tokens(graph).optimize()
