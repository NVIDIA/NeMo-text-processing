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

        graph_digit = cardinal.digit | cardinal.zero
        graph_tens = cardinal.teens_and_ties
        
        delete_decimal = pynutil.delete(".")

        optional_graph_negative = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\"") + insert_space, 0, 1,)

        dedh_dhai = pynini.string_map([("१.५", "डेढ़"), ("२.५", "ढाई")])
        dedh_dhai_graph = pynutil.insert("integer_part: \"") + dedh_dhai + pynutil.insert("\"")

        delete_zeros = pynini.closure(pynutil.delete("०"), 0)
        sadhe_numbers = (graph_digit | graph_tens) + pynini.cross(".५", "") + delete_zeros
        sadhe_graph = pynutil.insert("integer_part: \"साढ़े ") + sadhe_numbers + pynutil.insert("\"")

        savva_numbers = (graph_digit | graph_tens) + pynini.cross(".२५", "") + delete_zeros
        savva_graph = pynutil.insert("integer_part: \"सवा ") + savva_numbers + pynutil.insert("\"")

        cardinal_graph = cardinal.final_graph
        integer_graph = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")

        fraction_digits = graph_digit + pynini.closure(insert_space + graph_digit, 0)
        fractional_graph = pynutil.insert("fractional_part: \"") + fraction_digits + pynutil.insert("\"")

        integer_fraction_graph = integer_graph + delete_decimal + insert_space + fractional_graph

        weighted_graph = (
            pynutil.add_weight(dedh_dhai_graph, 0.05) 
            | pynutil.add_weight(sadhe_graph, 0.1) 
            | pynutil.add_weight(savva_graph, 0.1) 
            | pynutil.add_weight(integer_fraction_graph, 0.2)
        )

        self.final_graph = optional_graph_negative + weighted_graph
        
        self.fst = self.add_tokens(self.final_graph).optimize()

if __name__ == '__main__':
    from decimal import DecimalFst
    from cardinal import CardinalFst
    from nemo_text_processing.text_normalization.hi.utils import apply_fst

    cardinal = CardinalFst()
    decimal = DecimalFst(cardinal=cardinal)
    input_text = "१००१११.५"
    # input_text = "९०.५००"
    # input_text = "९०.२५००"
    # input_text = "५"
    # input_text = "१००७.५"
    apply_fst(input_text, decimal.fst)