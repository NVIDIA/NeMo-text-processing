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

from nemo_text_processing.text_normalization.ja.graph_utils import NEMO_NOT_QUOTE, NEMO_SPACE, GraphFst, delete_space


class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing fractionss, e.g.
    tokens { fraction { denominator: "二" numerator: "一"} } ->     1/2
    tokens { fraction { integer: "一" denominator: "四" numerator: "三" } } -> 1と3/4
    tokens { fraction { integer: "1" denominator: "4" numerator: "3" } } -> 一荷四分の三
    tokens { fraction { denominator: "√3" numerator: "1" } } -> ルート三分の一
    tokens { fraction { denominator: "1.65" numerator: "50" } } -> 一点六五分の五十
    tokens { fraction { denominator: "二" numerator: "一"} } -> マイナス1/2
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="fraction", kind="verbalize", deterministic=deterministic)

        denominator_component = (
            pynutil.delete('denominator: \"') + pynini.closure(NEMO_NOT_QUOTE - "√") + pynutil.delete("\"")
        )
        numerator_component = (
            pynutil.delete('numerator: \"') + pynini.closure(NEMO_NOT_QUOTE - "√") + pynutil.delete("\"")
        )

        # 1/3
        graph_regular_fraction = (
            denominator_component + pynutil.delete(NEMO_SPACE) + pynutil.insert("分の") + numerator_component
        )

        denominator_component_root = (
            pynutil.delete('denominator: \"')
            + pynini.cross("√", "ルート")
            + pynini.closure(NEMO_NOT_QUOTE - "√")
            + pynutil.delete("\"")
        )
        numerator_component_root = (
            pynutil.delete('numerator: \"')
            + pynini.cross("√", "ルート")
            + pynini.closure(NEMO_NOT_QUOTE - "√")
            + pynutil.delete("\"")
        )
        # √3/1
        graph_regular_fraction_root = (
            (denominator_component_root | denominator_component)
            + pynutil.delete(NEMO_SPACE)
            + pynutil.insert("分の")
            + (numerator_component_root | numerator_component)
        )

        # 3分の1
        graph_regular_fraction_char = (
            (denominator_component | denominator_component_root)
            + pynutil.delete(NEMO_SPACE)
            + pynutil.delete("morphosyntactic_features: \"")
            + pynini.closure(NEMO_NOT_QUOTE)
            + pynutil.delete("\"")
            + pynutil.delete(NEMO_SPACE)
            + (numerator_component | numerator_component_root)
        )

        graph_integer = (
            pynutil.delete("integer_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(pynini.cross("√", "ルート"), 0, 1)
            + pynini.closure(
                NEMO_NOT_QUOTE - pynini.union("荷", "と", "√")
            )  # had to remove these 3 items fron nemo_not _quote so the root is properly converted in a deterministic way.
            + pynutil.insert("荷")
            + pynutil.delete("\"")
        )

        graph_integer_with_char = (
            pynutil.delete("integer_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(pynini.cross("√", "ルート"), 0, 1)
            + pynini.closure(NEMO_NOT_QUOTE - pynini.union("荷", "と", "√"))
            + (pynini.accep("と") | pynini.accep("荷"))
            + pynutil.delete("\"")
        )

        graph_regular_integer = (
            (graph_integer | graph_integer_with_char)
            + delete_space
            + (graph_regular_fraction | graph_regular_fraction_root | graph_regular_fraction_char)
        )

        optional_sign = (
            pynutil.delete("negative:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE)
            + pynutil.delete("\"")
            + delete_space
        )

        graph = pynini.closure(optional_sign, 0, 1) + (
            graph_regular_integer | graph_regular_fraction | graph_regular_fraction_root | graph_regular_fraction_char
        )

        # graph = pynini.closure(graph_optional_sign, 0, 1) + graph_fractions

        final_graph = self.delete_tokens(graph)
        self.fst = final_graph.optimize()
