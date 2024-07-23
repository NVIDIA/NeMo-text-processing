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

from nemo_text_processing.text_normalization.ja.graph_utils import GraphFst
from nemo_text_processing.text_normalization.ja.utils import get_abs_path


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fractions, e.g.
    1/2 -> tokens { fraction { denominator: "二" numerator: "一"} }
    1と3/4 -> fraction { integer: "一" denominator: "四" numerator: "三" }
    一荷四分の三 -> fraction { integer: "1" denominator: "4" numerator: "3" }
    ルート三分の一 -> fraction { denominator: "√3" numerator: "1" }
    一点六五分の五十 -> fraction { denominator: "1.65" numerator: "50" }
    マイナス1/2 -> tokens { fraction { denominator: "二" numerator: "一"} }

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        cardinal = cardinal.just_cardinals
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        
        quantity = pynini.union(pynini.accep("万"), pynini.accep("百万"), pynini.accep("千万"), pynini.accep("億"), pynini.accep("百奥"), pynini.accep("千億"))
        slash = pynutil.delete('/')
        morphemes = pynutil.delete('分の')
        root = pynini.accep('√')

        # integer_component = pynutil.insert('integer_part: \"') + (cardinal | root + cardinal) + pynutil.insert("\"")
        # denominator_component = pynutil.insert("denominator: \"") + (cardinal | root + cardinal) + pynutil.insert("\"")
        # numerator_component = pynutil.insert("numerator: \"") + (cardinal | root + cardinal) + pynutil.insert("\"")

        integer_component = pynutil.insert('integer_part: \"') + cardinal + pynutil.insert("\"")
        denominator_component = pynutil.insert("denominator: \"") + cardinal + pynutil.insert("\"")
        numerator_component = pynutil.insert("numerator: \"") + cardinal + pynutil.insert("\"")

        graph_fraction_slash = numerator_component + slash + pynutil.insert(" ") + denominator_component
        # graph_fraction_slash_integer = integer_component + pynini.closure((pynini.accep("と") | pynini.accep("荷")), 0, 1) + pynutil.insert(" ") + graph_fraction_slash

        # graph_fraction_morphemes = denominator_component + morphemes + pynutil.insert(" ") + numerator_component
        # graph_fraction_morphemes_integer = integer_component + pynini.closure((pynini.accep("と") | pynini.accep("荷")), 0, 1) + pynutil.insert(" ") + graph_fraction_morphemes

        # graph_fraction = graph_fraction_slash | graph_fraction_slash_integer | graph_fraction_morphemes | graph_fraction_morphemes_integer

        # optional_sign = (
        #     pynutil.insert("negative: \"") + (pynini.accep("-") | pynini.cross("マイナス", "-")) + pynutil.insert("\"")
        # )

        # graph = graph_fraction | optional_sign + graph_fraction

        final_graph = self.add_tokens(graph_fraction_slash) ##
        self.fst = final_graph.optimize()
        