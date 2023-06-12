# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2023, Jim O'Regan for Språkbanken Tal
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
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_SIGMA, GraphFst
from nemo_text_processing.text_normalization.se.utils import get_abs_path
from pynini.lib import pynutil

quantities = pynini.string_file(get_abs_path("data/numbers/millions.tsv"))


def get_quantity(decimal: 'pynini.FstLike', cardinal_up_to_thousand: 'pynini.FstLike',) -> 'pynini.FstLike':
    """
    Returns FST that transforms either a cardinal or decimal followed by a quantity into a numeral,
    e.g. 1 miljárda -> integer_part: "okta" quantity: "miljárda"
    e.g. 1,5 miljárdda -> integer_part: "okta" fractional_part: "vihtta" quantity: "miljárdda"

    Args:
        decimal: decimal FST
        cardinal_up_to_hundred: cardinal FST
    """
    nom_to_gen_endings = pynini.string_map(("on", "ovnna"), ("árda", "árdda",))
    quantities_gen = quantities @ pynini.cdrewrite(nom_to_gen_endings, "", "[EOS]", NEMO_SIGMA)

    res = (
        pynutil.insert("integer_part: \"")
        + cardinal_up_to_thousand
        + pynutil.insert("\"")
        + pynini.closure(pynutil.delete(" "), 0, 1)
        + pynutil.insert(" quantity: \"")
        + quantities_gen
        + pynutil.insert("\"")
    )
    res |= (
        pynutil.insert("integer_part: \"")
        + pynini.cross("1", "okta")
        + pynutil.insert("\"")
        + pynini.closure(pynutil.delete(" "), 0, 1)
        + pynutil.insert(" quantity: \"")
        + quantities
        + pynutil.insert("\"")
    )
    res |= (
        decimal
        + pynini.closure(pynutil.delete(" "), 0, 1)
        + pynutil.insert(" quantity: \"")
        + quantities_gen
        + pynutil.insert("\"")
    )
    return res


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal, e.g.
        -12,5006 biljovnna -> decimal { negative: "true" integer_part: "guoktenuppelohkái" fractional_part: "vihtta nolla nolla guhtta" quantity: "biljovnna" }
        1 biljon -> decimal { integer_part: "okta" quantity: "biljon" }

    cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.graph
        cardinal_graph_hundreds_one_non_zero = cardinal.graph_hundreds_component_at_least_one_non_zero_digit_no_one

        self.graph = cardinal.two_or_three_digits_read_frac

        if not deterministic:
            self.graph |= cardinal.single_digits_graph.optimize()
            self.graph |= cardinal_graph

        point = pynutil.delete(",")
        optional_graph_negative = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        self.graph_fractional = pynutil.insert("fractional_part: \"") + self.graph + pynutil.insert("\"")
        self.graph_integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        final_graph_wo_sign = (
            pynini.closure(self.graph_integer + pynutil.insert(" "), 0, 1)
            + point
            + pynutil.insert(" ")
            + self.graph_fractional
        )
        self.final_graph_wo_sign = final_graph_wo_sign

        quantity_w_abbr = get_quantity(final_graph_wo_sign, cardinal_graph_hundreds_one_non_zero)
        self.final_graph_wo_negative = final_graph_wo_sign | quantity_w_abbr

        final_graph = optional_graph_negative + self.final_graph_wo_negative

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
