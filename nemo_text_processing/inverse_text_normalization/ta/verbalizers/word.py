# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
# limitations under the License.import pynini
import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.ta.graph_utils import NEMO_NOT_QUOTE, GraphFst


class WordFst(GraphFst):
    def __init__(self):
        super().__init__(name="word", kind="verbalize")

        graph = pynutil.delete('name: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')

        self.fst = graph.optimize()
