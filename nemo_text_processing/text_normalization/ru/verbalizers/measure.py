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
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_NON_BREAKING_SPACE,
    NEMO_SPACE,
    GraphFst,
    delete_space,
)
from nemo_text_processing.text_normalization.ru.alphabet import RU_ALPHA
from pynini.lib import pynutil


class MeasureFst(GraphFst):
    """
    Finite state transducer for verbalizing measure, e.g.
        measure { cardinal { integer: "два килограма" } } -> "два килограма"
    
    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="measure", kind="verbalize", deterministic=deterministic)

        graph = (
            pynutil.delete(" cardinal { integer: \"")
            + pynini.closure(RU_ALPHA | NEMO_SPACE | NEMO_NON_BREAKING_SPACE)
            + pynutil.delete("\"")
            + delete_space
            + pynutil.delete("}")
        )

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
