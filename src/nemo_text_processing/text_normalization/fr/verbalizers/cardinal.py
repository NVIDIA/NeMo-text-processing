# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, GraphFst, insert_space


class CardinalFst(GraphFst):
    """
    Finite state transducer for verbalizing cardinals
            e.g. cardinal { negative: "true" integer: "un milliard et un" } -> "moins un milliard et un"
    Args:
            deterministic: if True will provide a single transduction option,
                    for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="verbalize", deterministic=deterministic)

        optional_sign = pynini.closure(pynini.cross("negative: \"true\" ", "moins") + insert_space, 0, 1)
        number = pynutil.delete("integer: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        final_graph = optional_sign + number

        self.fst = self.delete_tokens(final_graph).optimize()
