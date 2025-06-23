# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.ja.graph_utils import NEMO_DIGIT, GraphFst
from nemo_text_processing.inverse_text_normalization.ja.utils import get_abs_path


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals
        e.g. 二十三 -> cardinal { integer: "23" }
        e.g. にじゅうさん -> cardinal { integer: "23" }
    """

    def __init__(self):
        super().__init__(name="cardinal", kind="classify")

        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv"))
        graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))

        graph_all = graph_ties + (graph_digit | pynutil.insert("0")) | graph_teen

        hundred = pynutil.delete("百") | pynutil.delete("ひゃく") | pynutil.delete("びゃく") | pynutil.delete("ぴゃく")
        hundred_alt = (
            pynini.cross("百", "1") | pynini.cross("ひゃく", "1") | pynini.cross("びゃく", "1") | pynini.cross("ぴゃく", "1")
        )
        graph_hundred_component = pynini.union(((graph_digit + hundred) | hundred_alt), pynutil.insert("0"))
        graph_hundred_component += pynini.union(
            graph_teen | pynutil.insert("00"), (graph_ties | pynutil.insert("0")) + (graph_digit | pynutil.insert("0"))
        )

        thousand = pynutil.delete("千") | pynutil.delete("せん") | pynutil.delete("ぜん")
        thousand_alt = pynini.cross("千", "1") | pynini.cross("せん", "1") | pynini.cross("ぜん", "1")
        graph_thousand_component = pynini.union(((graph_digit + thousand) | thousand_alt), pynutil.insert("0"))
        graph_thousand_component += graph_hundred_component
        graph_thousand_component = graph_thousand_component | pynutil.insert("0000")

        tenthousand = pynutil.delete("万") | pynutil.delete("まん")
        graph_tenthousand_component = pynini.union(graph_digit + tenthousand, pynutil.insert("0"))
        graph_tenthousand_component += graph_thousand_component

        graph_hundredthousand_component = pynini.union(
            (graph_teen | ((graph_ties | pynutil.insert("0")) + (graph_digit | pynutil.insert("0")))) + tenthousand,
            pynutil.insert("00"),
        )
        graph_hundredthousand_component += graph_thousand_component

        graph_million_component = pynini.union(graph_hundred_component + tenthousand, pynutil.insert("000"))
        graph_million_component += graph_thousand_component

        graph_tenmillion_component = pynini.union(graph_thousand_component + tenthousand, pynutil.insert("0000"))
        graph_tenmillion_component += graph_thousand_component

        hundredmillion = pynutil.delete("億") | pynutil.delete("おく")
        graph_hundredmillion_component = pynini.union(graph_digit + hundredmillion, pynutil.insert("0"))
        graph_hundredmillion_component += graph_tenmillion_component

        graph_billion_component = pynini.union(
            (graph_teen | ((graph_ties | pynutil.insert("0")) + (graph_digit | pynutil.insert("0")))) + hundredmillion,
            pynutil.insert("00"),
        )
        graph_billion_component += graph_tenmillion_component

        graph_tenbillion_component = pynini.union(graph_hundred_component + hundredmillion, pynutil.insert("000"))
        graph_tenbillion_component += graph_tenmillion_component

        graph_hundredbillion_component = pynini.union(
            graph_thousand_component + hundredmillion, pynutil.insert("0000")
        )
        graph_hundredbillion_component += graph_tenmillion_component

        graph_thousandbillion_component = pynini.union(
            graph_tenthousand_component + hundredmillion, pynutil.insert("00000")
        )
        graph_thousandbillion_component += graph_tenmillion_component  # e.g.,五万億 = 五兆

        graph_zyumannoku = pynini.union(graph_hundredthousand_component + hundredmillion, pynutil.insert("000000"))
        graph_zyumannoku += graph_tenmillion_component  # 五十万億 = 五十兆

        graph_hyakumanoku = pynini.union(graph_million_component + hundredmillion, pynutil.insert("0000000"))
        graph_hyakumanoku += graph_tenmillion_component  # 五百万億 = 五百兆

        graphsenmanoku = pynini.union(graph_tenmillion_component + hundredmillion, pynutil.insert("00000000"))
        graphsenmanoku += graph_tenmillion_component  # 五千万億 = 五千兆

        trillion = pynutil.delete("兆")
        graph_trillion_component = pynini.union(graph_digit + trillion, pynutil.insert("0"))
        graph_trillion_component += graph_hundredbillion_component

        graph_tentrillion_component = pynini.union(
            (graph_teen | ((graph_ties | pynutil.insert("0")) + (graph_digit | pynutil.insert("0")))) + trillion,
            pynutil.insert("00"),
        )
        graph_tentrillion_component += graph_hundredbillion_component

        graph_hundredtrillion_component = pynini.union(graph_hundred_component + trillion, pynutil.insert("000"))
        graph_hundredtrillion_component += graph_hundredbillion_component

        graph_thousandtrillion_component = pynini.union(graph_thousand_component + trillion, pynutil.insert("0000"))
        graph_thousandtrillion_component += graph_hundredbillion_component

        graph = pynini.union(
            (graph_thousandtrillion_component | graphsenmanoku),
            (graph_hundredtrillion_component | graph_hyakumanoku),
            (graph_tentrillion_component | graph_zyumannoku),
            (graph_trillion_component | graph_thousandbillion_component),
            graph_hundredbillion_component,
            graph_tenbillion_component,
            graph_billion_component,
            graph_hundredmillion_component,
            graph_tenmillion_component,
            graph_million_component,
            graph_hundredthousand_component,
            graph_tenthousand_component,
            graph_thousand_component,
            graph_hundred_component,
            graph_all,
            graph_digit,
            graph_zero,
        )

        leading_zero = (
            pynutil.delete(pynini.closure("0")) + pynini.difference(NEMO_DIGIT, "0") + pynini.closure(NEMO_DIGIT)
        )
        graph = graph @ leading_zero | graph_zero

        self.just_cardinals = graph

        optional_sign = (
            pynutil.insert("negative: \"") + (pynini.accep("-") | pynini.cross("マイナス", "-")) + pynutil.insert("\"")
        )

        final_graph = (
            optional_sign + pynutil.insert(" ") + pynutil.insert("integer: \"") + graph + pynutil.insert("\"")
        ) | (pynutil.insert("integer: \"") + graph + pynutil.insert("\""))

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
