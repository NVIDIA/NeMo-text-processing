# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.ko.graph_utils import GraphFst, NEMO_CHAR
from nemo_text_processing.inverse_text_normalization.ko.utils import get_abs_path


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinal
        Expressing integers in ordinal way for 1-39 and cardinal for 40+ due to Korean grammar.
        e.g. 스물세번째 -> ordinal {integer: "23", 23번째}
        e.g. 사십오번째 -> ordinal but the integer part is written in cardinal(due to korean grammar)
        { integer: "45", 45번쨰}
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="ordinal", kind="classify")

        cardinals = cardinal.just_cardinals
        graph_digit = pynini.string_file(get_abs_path("data/ordinals/digit.tsv"))
        ordinals = pynini.accep("째") | pynini.accep("번째")

        ordinal_graph = (
            pynutil.insert("integer: \"") + ((graph_digit + ordinals) | (cardinals + ordinals)) + pynutil.insert("\"")
        )

        final_graph = self.add_tokens(ordinal_graph)
        self.fst = final_graph.optimize()