# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
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

from nemo_text_processing.text_normalization.hi.graph_utils import GraphFst, insert_space
from nemo_text_processing.text_normalization.hi.utils import get_abs_path

quantities = pynini.string_file(get_abs_path("data/numbers/thousands.tsv"))


def get_quantity(decimal: 'pynini.FstLike', cardinal_up_to_hundred: 'pynini.FstLike') -> 'pynini.FstLike':

    """
    Returns FST that transforms either a cardinal or decimal followed by a quantity into a numeral,
    e.g. १ लाख -> integer_part: "एक" quantity: "लाख"
    e.g. १.५ लाख -> integer_part: "एक" fractional_part: "पाँच" quantity: "लाख"

    Args: 
        decimal: decimal FST
        cardinal_up_to_hundred: cardinal FST
    """
    numbers = cardinal_up_to_hundred

    res = (
        pynutil.insert("integer_part: \"")
        + numbers
        + pynutil.insert("\"")
        + insert_space
        + pynutil.insert("quantity: \"")
        + quantities
        + pynutil.insert("\"")
    )
    res |= decimal + insert_space + pynutil.insert("quantity: \"") + quantities + pynutil.insert("\"")
    return res


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal, e.g. 
        -१२.५००६ अरब -> decimal { negative: "true" integer_part: "बारह"  fractional_part: "पाँच शून्य शून्य छह" quantity: "अरब" }
        १ अरब -> decimal { integer_part: "एक" quantity: "अरब" }

    cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_digit |= pynini.string_file(get_abs_path("data/numbers/zero.tsv"))

        cardinal_graph = cardinal.final_graph

        self.graph = graph_digit + pynini.closure(insert_space + graph_digit).optimize()

        point = pynutil.delete(".")

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\"") + insert_space, 0, 1,
        )

        self.graph_fractional = pynutil.insert("fractional_part: \"") + self.graph + pynutil.insert("\"")
        self.graph_integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")

        final_graph_wo_sign = self.graph_integer + point + insert_space + self.graph_fractional

        self.final_graph_wo_negative = final_graph_wo_sign | get_quantity(final_graph_wo_sign, cardinal_graph)

        final_graph = optional_graph_negative + self.final_graph_wo_negative

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
