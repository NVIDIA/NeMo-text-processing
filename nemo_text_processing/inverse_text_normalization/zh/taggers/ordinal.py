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

from nemo_text_processing.inverse_text_normalization.zh.graph_utils import GraphFst


class OrdinalFst(GraphFst):
    def __init__(self, cardinal: GraphFst):
        super().__init__(name="ordinal", kind="classify")

        graph_cardinals = cardinal.for_ordinals
        mandarin_morpheme = pynini.accep("第")
        graph_ordinal = mandarin_morpheme + graph_cardinals
        graph_ordinal_final = pynutil.insert('integer: "') + graph_ordinal + pynutil.insert('"')
        graph_ordinal_final = self.add_tokens(graph_ordinal_final)
        self.fst = graph_ordinal_final.optimize()
