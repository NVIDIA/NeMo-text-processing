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
from nemo_text_processing.text_normalization.ja.graph_utils import (
    NEMO_DIGIT,
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_space,
)
from pynini.lib import pynutil


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
        super().__init__(name="cardinal", kind="verbalize", deterministic=deterministic)

        graph_optional_sign = (
            pynutil.delete("negative:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE)
            + pynutil.delete("\"")
            + delete_space
        )

        graph_denominator = pynutil.delete('denominator: \"') +  pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        graph_numerator = pynutil.delete('numerator: \"') +  pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        #graph_regular = graph_denominator + pynutil.delete(" ") + pynutil.insert("分の") + graph_numerator
        graph_regular = graph_numerator + pynutil.delete(" ") + pynutil.insert("分の") + graph_denominator
        
        graph_integer = (
            pynutil.delete("integer_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE)
            + pynutil.delete("\"")
        )

        graph_with_integer = graph_integer + delete_space  + graph_regular

        optional_sign = (
            pynutil.delete("negative:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE)
            + pynutil.delete("\"")
            + delete_space
        )

        graph_fractions = graph_with_integer | graph_regular

        final_graph = pynini.closure(graph_optional_sign, 0, 1) + graph_fractions

        final_graph = self.delete_tokens(graph_regular)
        self.fst = final_graph.optimize()