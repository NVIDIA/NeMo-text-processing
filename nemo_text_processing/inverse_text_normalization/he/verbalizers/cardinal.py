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

from nemo_text_processing.inverse_text_normalization.he.graph_utils import NEMO_ALPHA_HE, GraphFst
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, delete_space


class CardinalFst(GraphFst):
    """
    Finite state transducer for verbalizing cardinal in Hebrew
        e.g. cardinal { prefix: "וב" integer: "3405"} -> וב-3,405
        e.g. cardinal { negative: "-" integer: "904" } -> -904
        e.g. cardinal { prefix: "כ" integer: "123" } -> כ-123

    """

    def __init__(self):
        super().__init__(name="cardinal", kind="verbalize")

        # Need parser to group digits by threes
        exactly_three_digits = NEMO_DIGIT**3
        at_most_three_digits = pynini.closure(NEMO_DIGIT, 1, 3)

        # Thousands separator
        group_by_threes = at_most_three_digits + (pynutil.insert(",") + exactly_three_digits).closure()

        # Keep the prefix if exists and add a dash
        optional_prefix = pynini.closure(
            pynutil.delete("morphosyntactic_features:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_ALPHA_HE, 1)
            + pynutil.insert("-")
            + pynutil.delete('"')
            + delete_space,
            0,
            1,
        )

        # Removes the negative attribute and leaves the sign if occurs
        optional_sign = pynini.closure(
            pynutil.delete("negative:")
            + delete_space
            + pynutil.delete('"')
            + pynini.accep("-")
            + pynutil.delete('"')
            + delete_space,
            0,
            1,
        )

        # removes integer aspect
        graph = (
            pynutil.delete("integer:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_DIGIT, 1)  # Accepts at least one digit
            + pynutil.delete('"')
        )

        # Add thousands separator
        graph = graph @ group_by_threes

        self.numbers = graph

        # add prefix and sign
        graph = optional_prefix + optional_sign + graph

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
