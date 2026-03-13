# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.pt.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space, insert_space
from nemo_text_processing.text_normalization.pt.utils import get_abs_path, load_labels


class DecimalFst(GraphFst):
    """
    Finite state transducer for verbalizing Portuguese decimal numbers, e.g.
        decimal { integer_part: "um" fractional_part: "vinte e seis" } -> um vírgula vinte e seis
        decimal { negative: "true" integer_part: "um" ... } -> menos um vírgula ...
        decimal { integer_part: "um" quantity: "milhão" } -> um milhão

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="decimal", kind="verbalize", deterministic=deterministic)
        labels = load_labels(get_abs_path("data/numbers/decimal_specials.tsv"))
        spec = {r[0]: r[1] for r in labels if len(r) >= 2}
        sep = spec.get("separator", "vírgula")
        minus = spec.get("minus", "menos")

        optional_sign = pynini.closure(pynini.cross('negative: "true" ', minus + " "), 0, 1)

        integer = pynutil.delete('integer_part: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')
        fractional = pynutil.delete('fractional_part: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')
        quantity = (
            delete_space
            + insert_space
            + pynutil.delete('quantity: "')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        integer_quantity = integer + quantity
        decimal_part = (
            integer
            + delete_space
            + insert_space
            + pynutil.insert(sep + " ")
            + fractional
            + pynini.closure(quantity, 0, 1)
        )

        graph = optional_sign + pynini.union(integer_quantity, decimal_part)

        self.fst = self.delete_tokens(graph).optimize()
