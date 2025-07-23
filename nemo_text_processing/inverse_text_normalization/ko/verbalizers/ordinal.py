# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.ko.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals
        e.g. 스물세번째 -> ordinal {integer: "23", 23번째}
        e.g. 사십오번째 -> ordinal but the integer part is written in cardinal(due to korean grammar)
        { integer: "45", 45번쨰}
    """

    def __init__(self):
        super().__init__(name="ordinal", kind="verbalize")

        integer_component = pynutil.delete("integer: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        counter_component = pynutil.delete("counter: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")

        graph_with_counter = integer_component + delete_space + counter_component

        ordinal_verbalizer = pynini.union(graph_with_counter, integer_component)

        final_graph = self.delete_tokens(ordinal_verbalizer)
        self.fst = final_graph.optimize()
