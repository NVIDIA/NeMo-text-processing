# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from nemo_text_processing.text_normalization.en.graph_utils import GraphFst
from nemo_text_processing.text_normalization.ru.alphabet import RU_ALPHA
from pynini.lib import pynutil


class DateFst(GraphFst):
    """
    Finite state transducer for verbalizing date, e.g.
        tokens { date { day: "первое мая" } } -> "первое мая"

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="date", kind="verbalize", deterministic=deterministic)

        graph = pynutil.delete("day: \"") + pynini.closure(RU_ALPHA | " ", 1) + pynutil.delete("\"")
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
