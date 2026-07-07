# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, GraphFst, convert_space


class RangeFst(GraphFst):
    """
    Composite class for range/quantity expressions in Greek text normalization.

    Handles:
        - Time ranges: "5:00-6:00" → "5:00 προς 6:00"
        - Year ranges: "1980-1985" → "1980 προς 1985"
        - Mid-year: "mid-1980" → "μέσα 1980"
        - Number ranges: "100-200" → "100 προς 200"
        - Plus: "100+" → "100 συν"
        - Approx: "~100" → "περίπου 100"
        - Ellipsis: "1...5" → "1 ... 5"
        - Multiplication/division (non-det): "10 x 20" → "10 επί 20", "10/20" → "10 δια 20"
        - Number/NO.: "No. 12" → "Αριθμός 12"

    Args:
        time: composed tagger and verbalizer (outputs spoken form directly)
        date: composed tagger and verbalizer (outputs spoken form directly)
        cardinal: tagger instance (uses graph_no_tokens)
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
        lm: whether to use for hybrid LM
    """

    def __init__(
        self,
        time: GraphFst,
        date: GraphFst,
        cardinal: GraphFst,
        deterministic: bool = True,
        lm: bool = False,
    ):
        super().__init__(name="range", kind="classify", deterministic=deterministic)

        delete_space = pynini.closure(pynutil.delete(" "), 0, 1)

        approx = pynini.cross("~", "περίπου")

        # TIME
        time_graph = time + delete_space + pynini.cross("-", " προς ") + delete_space + time
        self.graph = time_graph | (approx + time)

        cardinal_graph = cardinal.graph_no_tokens

        # YEAR
        date_year_four_digit = (NEMO_DIGIT**4 + pynini.closure(pynini.accep("s"), 0, 1)) @ date
        date_year_two_digit = (NEMO_DIGIT**2 + pynini.closure(pynini.accep("s"), 0, 1)) @ date
        year_to_year_graph = (
            date_year_four_digit
            + delete_space
            + pynini.cross("-", " προς ")
            + delete_space
            + (date_year_four_digit | date_year_two_digit | (NEMO_DIGIT**2 @ cardinal_graph))
        )
        mid_year_graph = (
            pynini.cross("mid", "μέσα") + pynini.cross("-", " ") + (date_year_four_digit | date_year_two_digit)
        )

        self.graph |= year_to_year_graph
        self.graph |= mid_year_graph

        # ADDITION / APPROX / ELLIPSIS
        range_graph = cardinal_graph + pynini.closure(pynini.cross("+", " συν ") + cardinal_graph, 1)
        range_graph |= cardinal_graph + pynini.closure(pynini.cross(" + ", " συν ") + cardinal_graph, 1)
        range_graph |= approx + cardinal_graph
        range_graph |= cardinal_graph + (pynini.cross("...", " ... ") | pynini.accep(" ... ")) + cardinal_graph

        if not deterministic or lm:
            # cardinal ----
            cardinal_to_cardinal_graph = (
                cardinal_graph
                + delete_space
                + pynini.cross("-", pynini.union(" προς ", " μείον "))
                + delete_space
                + cardinal_graph
            )

            range_graph |= cardinal_to_cardinal_graph | (
                cardinal_graph + delete_space + pynini.cross(":", " προς ") + delete_space + cardinal_graph
            )

            # MULTIPLY
            for x in [" x ", "x"]:
                range_graph |= cardinal_graph + pynini.cross(x, pynini.union(" επί ", " φορές ")) + cardinal_graph

            for x in [" x", "x"]:
                range_graph |= cardinal_graph + pynini.cross(x, " φορές")

                # 5x to 7x → πέντε προς επτά x/φορές
                range_graph |= (
                    cardinal_graph
                    + pynutil.delete(x)
                    + pynini.union(" προς ", "-", " - ")
                    + cardinal_graph
                    + pynini.cross(x, pynini.union(" x", " φορές"))
                )

            for x in ["*", " * "]:
                range_graph |= cardinal_graph + pynini.closure(pynini.cross(x, " φορές ") + cardinal_graph, 1)

            # No. 12 → Αριθμός δώδεκα
            range_graph |= (
                (pynini.cross(pynini.union("NO", "No"), "Αριθμός") | pynini.cross("no", "αριθμός"))
                + pynini.closure(pynini.union(". ", " "), 0, 1)
                + cardinal_graph
            )

            for x in ["/", " / "]:
                range_graph |= cardinal_graph + pynini.closure(pynini.cross(x, " δια ") + cardinal_graph, 1)

            # 10% to 20% → δέκα τοις εκατό προς είκοσι τοις εκατό
            range_graph |= (
                cardinal_graph
                + pynini.closure(pynini.cross("%", " τοις εκατό") | pynutil.delete("%"), 0, 1)
                + pynini.union(" προς ", "-", " - ")
                + cardinal_graph
                + pynini.cross("%", " τοις εκατό")
            )

        self.graph |= range_graph

        self.graph = self.graph.optimize()
        graph = pynutil.insert("name: \"") + convert_space(self.graph).optimize() + pynutil.insert("\"")
        self.fst = graph.optimize()
