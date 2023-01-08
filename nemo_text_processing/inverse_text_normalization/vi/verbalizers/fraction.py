# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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
from nemo_text_processing.inverse_text_normalization.vi.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space
from pynini.lib import pynutil


class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing fraction,
        e.g. fraction { numerator: "2" denominator: "3" } } -> 2/3
        e.g. fraction { numerator: "20" denominator: "3" negative: "true"} } -> 2/3
    """

    def __init__(self):
        super().__init__(name="fraction", kind="verbalize")
        optional_sign = pynini.closure(pynini.cross('negative: "true"', "-") + delete_space, 0, 1)
        numerator = pynutil.delete('numerator: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')

        denominator = (
            pynutil.insert("/")
            + pynutil.delete('denominator: "')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        graph = (numerator + delete_space + denominator).optimize()
        self.numbers = graph
        delete_tokens = self.delete_tokens(optional_sign + graph)
        self.fst = delete_tokens.optimize()
