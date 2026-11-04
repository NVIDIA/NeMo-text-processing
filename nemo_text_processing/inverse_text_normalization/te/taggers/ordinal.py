# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.te.graph_utils import NEMO_CHAR, GraphFst
from nemo_text_processing.inverse_text_normalization.te.utils import get_abs_path


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying Telugu ordinals.

    Examples:
        మొదటి -> ordinal { integer: "౧" }
        పదకొండవ -> ordinal { integer: "౧౧" }
        ఇరవై ఒకటవ -> ordinal { integer: "౨౧" }
        రెండువందలవ -> ordinal { integer: "౨౦౦" }

    Args:
        cardinal: CardinalFst instance used to obtain the cardinal graph.
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="ordinal", kind="classify")

        graph_digit = pynini.string_file(get_abs_path("data/ordinals/digit.tsv"))
        graph_teens = pynini.string_file(get_abs_path("data/ordinals/teens_and_ties.tsv"))
        graph_hundred = pynini.string_file(get_abs_path("data/ordinals/hundred_digit.tsv"))
        graph_cardinal_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_cardinal_teens = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv"))
        graph_cardinal_hundred = pynini.string_file(get_abs_path("data/numbers/hundred.tsv"))

        cardinal_graph = cardinal.graph_no_exception

        cardinal_teens = pynini.project(graph_cardinal_teens, "input").optimize()

        teen_suffix = pynini.union(
            pynini.cross("డు", "డవ"),
            pynini.cross("డు", "డో"),
            pynini.cross("గు", "గవ"),
            pynini.cross("గు", "గో"),
            pynini.cross("ను", "నవ"),
            pynini.cross("ను", "నో"),
            pynini.cross("రు", "రవ"),
            pynini.cross("రు", "రో"),
        ).optimize()

        generated_teens = (cardinal_teens @ (pynini.closure(NEMO_CHAR) + teen_suffix)).invert().optimize()

        cardinal_digits = pynini.project(graph_cardinal_digit, "input").optimize()

        first_digit = pynini.project(
            graph_cardinal_digit @ pynini.accep("౧"),
            "input",
        ).optimize()

        hundred_multipliers = pynini.difference(
            cardinal_digits,
            first_digit,
        ).optimize()

        hundred_marker = (pynini.project(graph_cardinal_hundred, "input") @ pynini.accep("వందలు")).optimize()

        hundred_words = (hundred_multipliers + hundred_marker).optimize()

        hundred_suffix = pynini.union(
            pynini.cross("లు", "లవ"),
            pynini.cross("లు", "లో"),
        ).optimize()

        generated_hundreds = (hundred_words @ (pynini.closure(NEMO_CHAR) + hundred_suffix)).invert().optimize()

        ordinal_tail = pynini.union(
            graph_digit,
            graph_teens,
            graph_hundred,
            generated_teens,
            generated_hundreds,
        ).optimize()

        graph = pynini.compose(
            pynini.closure(NEMO_CHAR) + ordinal_tail,
            cardinal_graph,
        ).optimize()

        final_graph = pynutil.insert('integer: "') + graph + pynutil.insert('"')

        self.fst = self.add_tokens(final_graph).optimize()
