# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright (c) 2022, Jim O'Regan for Språkbanken Tal
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
from nemo_text_processing.text_normalization.ga.graph_utils import ensure_space
from nemo_text_processing.text_normalization.ga.utils import get_abs_path
from pynini.lib import pynutil


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction
    "23 4/5" ->
    tokens { fraction { integer: "fiche a trí" numerator: "a ceathair" denominator: "a cúig" } }

    Args:
        cardinal: CardinalFst
        ordinal: OrdinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, ordinal: GraphFst, deterministic: bool = True):
        super().__init__(name="fraction", kind="classify", deterministic=deterministic)
        cardinal_graph = cardinal.graph
        digit_sg = pynini.string_file(get_abs_path("data/fractions/digit_sg.tsv"))
        digit_pl = pynini.string_file(get_abs_path("data/fractions/digit_pl.tsv"))
        tens_sg = pynini.string_file(get_abs_path("data/fractions/tens_sg.tsv"))
        tens_pl = pynini.string_file(get_abs_path("data/fractions/tens_pl.tsv"))
        teen_sg = pynini.string_file(get_abs_path("data/fractions/teen_sg.tsv"))
        teen_pl = pynini.string_file(get_abs_path("data/fractions/teen_pl.tsv"))

        integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        numerator = (
            pynutil.insert("numerator: \"") + cardinal_graph + (pynini.cross("/", "\" ") | pynini.cross(" / ", "\" "))
        )

        denominator = pynutil.insert("denominator: \"") + cardinal_graph + pynutil.insert("\"")

        graph = pynini.closure(integer + pynini.accep(" "), 0, 1) + (numerator + denominator)
        graph |= pynini.closure(integer + ensure_space, 0, 1) + pynini.compose(
            pynini.string_file(get_abs_path("data/numbers/fraction.tsv")), (numerator + denominator)
        )

        self.graph = graph
        final_graph = self.add_tokens(self.graph)
        self.fst = final_graph.optimize()
