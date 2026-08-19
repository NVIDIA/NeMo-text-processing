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

from nemo_text_processing.inverse_text_normalization.ja.graph_utils import NEMO_NOT_QUOTE, GraphFst


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals
        e.g. ordinal { morphsyntactic_feature: "第" integer: "23" }  -> 第23
        e.g. ordinal { integer: "100" morphsyntactic_feature: "番目" } -> 100番目
    """

    def __init__(self):
        super().__init__(name="ordinal", kind="verbalize")

        integer_component = pynutil.delete("integer: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")

        final_graph = self.delete_tokens(integer_component)
        self.fst = final_graph.optimize()
