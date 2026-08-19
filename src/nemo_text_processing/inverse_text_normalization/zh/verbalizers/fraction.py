# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.zh.graph_utils import NEMO_DIGIT, GraphFst, delete_space


class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing fraction
        e.g. tokens { fraction { denominator: "2" numerator: "1"} } -> 1/2
        e.g. tokens { fraction { integer_part: "1" denominator: "2" numerator: "1" } } -> 1又1/2
    """

    def __init__(self):
        super().__init__(name="fraction", kind="verbalize")

        integer_part = (
            pynutil.delete("integer_part:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_DIGIT)
            + pynutil.insert("又")
            + pynutil.delete('"')
        )
        denominator_part = (
            pynutil.delete("denominator:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_DIGIT)
            + pynutil.delete('"')
        )
        numerator_part = (
            pynutil.delete("numerator:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_DIGIT)
            + pynutil.insert("/")
            + pynutil.delete('"')
        )

        graph_with_integer = integer_part + delete_space + numerator_part + delete_space + denominator_part
        graph_no_integer = numerator_part + delete_space + denominator_part

        final_graph = graph_with_integer | graph_no_integer

        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()
