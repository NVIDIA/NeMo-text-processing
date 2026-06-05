# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.vi.graph_utils import NEMO_DIGIT, GraphFst
from nemo_text_processing.text_normalization.vi.utils import get_abs_path, load_labels


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying Vietnamese ordinals, e.g.
        thứ 1 -> ordinal { integer: "nhất" }
        thứ 4 -> ordinal { integer: "tư" }
        thứ 15 -> ordinal { integer: "mười lăm" }
    Args:
        cardinal: CardinalFst for number conversion
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, cardinal, deterministic: bool = True):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        prefix = "thứ "
        number_pattern = pynini.closure(NEMO_DIGIT, 1)

        ordinal_exceptions = {
            row[0]: row[1] for row in load_labels(get_abs_path("data/ordinal/ordinal_exceptions.tsv"))
        }

        exception_patterns = []
        for digit, word in ordinal_exceptions.items():
            exception_patterns.append(pynini.cross(digit, word))

        exception_graph = pynini.union(*exception_patterns) if exception_patterns else None

        combined_graph = cardinal.graph
        if exception_graph:
            combined_graph = pynini.union(exception_graph, cardinal.graph)

        self.graph = (
            pynutil.delete(prefix)
            + pynutil.insert("integer: \"")
            + pynini.compose(number_pattern, combined_graph)
            + pynutil.insert("\"")
        )

        self.fst = self.add_tokens(self.graph).optimize()
