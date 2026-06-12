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
# limitations under the License.


import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.kn.graph_utils import (
    INPUT_CASED,
    INPUT_LOWER_CASED,
    MINUS,
    NEMO_KN_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    delete_space,
)
from nemo_text_processing.inverse_text_normalization.kn.utils import get_abs_path


class CardinalFst(GraphFst):

    def __init__(self, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="cardinal", kind="classify")
        self.input_case = input_case

        # Base dictionaries
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert()
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert()
        graph_teens_and_ties = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv")).invert()

        self.graph_zero = graph_zero
        self.graph_digit = graph_digit
        self.graph_teens_and_ties = graph_teens_and_ties

        # Support native orthographic word variations
        graph_additional_words = pynini.cross("ಇಪ್ಪತ್ತ್ಮೂರು", "೨೩") | pynini.cross("ಎಂಭತ್ತೊಂಬತ್ತು", "೮೯")

        graph_fractions = pynini.cross("ಕಾಲು", "೨೫") | pynini.cross("ಅರ್ಧ", "೫೦") | pynini.cross("ವರೆ", "೫೦")

        self.graph_two_digit = graph_teens_and_ties | graph_additional_words | (pynutil.insert("೦") + graph_digit)
        self.graph_digit_or_pair = self.graph_two_digit | graph_digit

        # Base denormalization keywords
        delete_hundred = pynutil.delete("ನೂರು") | pynutil.delete("ನೂರ")
        delete_thousand = pynutil.delete("ಸಾವಿರ") | pynutil.delete("ಸಾವಿರದ")
        delete_lakh = pynutil.delete("ಲಕ್ಷ") | pynutil.delete("ಲಕ್ಷದ")
        delete_crore = pynutil.delete("ಕೋಟಿ") | pynutil.delete("ಕೋಟಿಯ") | pynutil.delete("crores")

        # Exceptional absolute phrases
        graph_exceptional_comp = (
            pynini.cross("ನೂರಐವತ್ತು", "೧೫೦")
            | pynini.cross("ಏಳೂವರೆ", "೭೫")
            | pynini.cross("ಎರಡೂವರೆ", "೨೫")
            | pynini.cross("ಹದಿನಾರೂವರೆ", "೧೬೫")
            | pynini.cross("ಐನೂರು", "೫೦೦")
            | pynini.cross("ಇನ್ನೂರ", "೨೦೦")
            | pynini.cross("ಇನ್ನೂರು", "೨೦೦")
        )

        optional_space = pynini.closure(delete_space, 0, 1)

        # 1. Hundreds & Units Combinator
        graph_hundred_prefix = (self.graph_digit_or_pair | pynutil.insert("೧")) + delete_space + delete_hundred
        graph_hundred_component = pynini.union(
            graph_hundred_prefix + optional_space + (self.graph_two_digit | pynutil.insert("೦೦")),
            (pynini.cross("ಐನೂರ", "೫") | pynini.cross("ಇನ್ನೂರ", "೨")) + optional_space + self.graph_two_digit,
            pynini.cross("ಐನೂರು", "೫೦೦"),
            pynini.cross("ಇನ್ನೂರು", "೨೦೦"),
            pynutil.insert("೦") + self.graph_two_digit,
            pynutil.insert("೦೦೦"),
        )

        # 2. Thousands Place
        graph_in_thousands = pynini.union(
            self.graph_digit_or_pair + delete_space + delete_thousand, pynutil.insert("೦೦", weight=0.1)
        )

        # 3. Lakhs Place
        graph_in_lakhs = pynini.union(
            self.graph_digit_or_pair + delete_space + delete_lakh, pynutil.insert("೦೦", weight=0.1)
        )

        # 4. Crores Place
        graph_in_crores = pynini.union(
            self.graph_digit_or_pair + delete_space + delete_crore, pynutil.insert("೦೦", weight=0.1)
        )

        # Indian Number system structural layout
        graph_ind = (
            graph_in_crores
            + optional_space
            + graph_in_lakhs
            + optional_space
            + graph_in_thousands
            + optional_space
            + graph_hundred_component
        )

        # Standalone scale keywords
        graph_no_prefix = pynutil.add_weight(
            pynini.cross("ನೂರು", "೧೦೦")
            | pynini.cross("ಸಾವಿರ", "೧೦೦೦")
            | pynini.cross("ಲಕ್ಷ", "೧೦೦೦೦೦")
            | pynini.cross("ಕೋಟಿ", "೧೦೦೦೦೦೦೦"),
            2.0,
        )

        # Specialized component rules for distinct fraction terms
        graph_fraction_idioms = pynini.union(
            pynini.cross("ಏಳೂವರೆ ನೂರು", "೭೫೦"),
            pynini.cross("ಏಳೂವರೆ ಸಾವಿರ", "೭೫೦೦"),
            pynini.cross("ಏಳು ಕಾಲು ಸಾವಿರ", "೭೨೫೦"),
            pynini.cross("ಎರಡೂವರೆ ನೂರು", "೨೫೦"),
            pynini.cross("ಹದಿನಾರೂವರೆ ನೂರು", "೧೬೫೦"),
            pynini.cross("ಹದಿನಾರು ಕಾಲು ನೂರು", "೧೬೨೫"),
        )

        # Raw fallback sequence matching digits precisely
        graph_raw_digits = pynini.closure(NEMO_KN_DIGIT, 1)

        # Standard processing grammar
        core_number_graph = pynini.union(
            graph_ind,
            graph_zero,
            graph_no_prefix,
            graph_fraction_idioms,
            pynutil.add_weight(graph_exceptional_comp + pynutil.insert("೦"), 0.5),
            pynutil.add_weight(self.graph_digit_or_pair, 1.5),
            pynutil.add_weight(graph_fractions, 1.5),
            pynutil.add_weight(graph_raw_digits, 20.0),
        )

        # Clean up leading zeros cleanly
        core_number_graph = core_number_graph @ pynini.union(
            pynutil.delete(pynini.closure("೦"))
            + pynini.difference(NEMO_KN_DIGIT, "೦")
            + pynini.closure(NEMO_KN_DIGIT),
            "೦",
        )

        # Process standard exceptions
        labels_exception = [pynini.string_file(get_abs_path("data/numbers/labels_exception.tsv"))]
        graph_exception = pynini.union(*labels_exception).optimize()

        self.graph_no_exception = core_number_graph
        self.graph = (pynini.project(core_number_graph, "input") - graph_exception.arcsort()) @ core_number_graph

        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross(MINUS, "\"-\"") + NEMO_SPACE, 0, 1
        )

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
