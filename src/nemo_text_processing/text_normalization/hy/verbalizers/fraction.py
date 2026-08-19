# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, NEMO_SIGMA, GraphFst, delete_space


class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing fraction
        e.g. fraction { numerator: "երկու" denominator: "երրորդ" } } -> 2/3

    """

    def __init__(self):
        super().__init__(name="fraction", kind="verbalize")
        numerator = pynutil.delete("numerator: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")

        denominator = (
            pynutil.insert(' ')
            + pynutil.delete("denominator: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        suffix = pynini.cdrewrite(pynini.cross("ըերորդ", "ներորդ"), "", "[EOS]", NEMO_SIGMA).optimize()

        graph = (numerator + delete_space + pynini.compose(denominator, suffix)).optimize()
        self.numbers = graph.optimize()
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
