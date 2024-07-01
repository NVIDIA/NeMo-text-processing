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
from nemo_text_processing.inverse_text_normalization.he.graph_utils import NEMO_DIGIT, NEMO_ALPHA, GraphFst, delete_space
from pynini.lib import pynutil


class CardinalFst(GraphFst):
    """
    Finite state transducer for verbalizing cardinal
        e.g. cardinal { integer: "23" negative: "-" } -> -23
    """

    def __init__(self):
        super().__init__(name="cardinal", kind="verbalize")
        # Need parser to group digits by threes
        exactly_three_digits = NEMO_DIGIT ** 3
        at_most_three_digits = pynini.closure(NEMO_DIGIT, 1, 3)

        group_by_threes = (
            at_most_three_digits +
            (pynutil.insert(",") + exactly_three_digits).closure()
        )

        # Keep the prefix if exists and add a dash
        optional_prefix = pynini.closure(
            pynutil.delete("prefix:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_ALPHA, 1)
            + pynutil.insert("-")
            + pynutil.delete("\"")
            + delete_space,
            0,
            1,
        )

        # Removes the negative attribute and leaves the sign if occurs
        optional_sign = pynini.closure(
            pynutil.delete("negative:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.accep("-")
            + pynutil.delete("\"")
            + delete_space,
            0,
            1,
        )

        # removes integer aspect
        graph = (
                pynutil.delete("integer:")
                + delete_space
                + pynutil.delete("\"")
                + pynini.closure(NEMO_DIGIT, 1)  # Accepts at least one digit
                + pynutil.delete("\"")
        )
        graph = graph @ group_by_threes

        self.numbers = graph
        graph = optional_prefix + optional_sign + graph
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()

if __name__ == "__main__":
    from nemo_text_processing.inverse_text_normalization.he.graph_utils import apply_fst

    cardinal = CardinalFst().fst
    apply_fst('cardinal { negative: "-" integer: "204" }', cardinal)
    apply_fst('cardinal {integer: "3204" }', cardinal)
    apply_fst('cardinal {prefix: "ב" integer: "50"}', cardinal)
    apply_fst('cardinal {prefix: "כ" integer: "123"}', cardinal)
    apply_fst('cardinal {prefix: "ו" integer: "305"}', cardinal)
    apply_fst('cardinal {prefix: "וב" integer: "3405"}', cardinal)
