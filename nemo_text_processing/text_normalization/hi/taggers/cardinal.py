# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.hi.graph_utils import GraphFst
from nemo_text_processing.text_normalization.hi.utils import get_abs_path


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g. 
        -२३ -> cardinal { negative: "true"  integer: "तेइस" } }
  s
    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        teens_ties = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv"))
        teens_and_ties = pynutil.add_weight(teens_ties, -0.1)

        def create_graph_suffix(digit_graph, suffix, zeros_counts):
            zero = pynutil.add_weight(pynutil.delete("०"), -0.1)
            if zeros_counts == 0:
                return digit_graph + suffix

            return digit_graph + (zero ** zeros_counts) + suffix

        def create_larger_number_graph(digit_graph, suffix, zeros_counts, sub_graph):
            insert_space = pynutil.insert(" ")
            zero = pynutil.add_weight(pynutil.delete("०"), -0.1)
            if zeros_counts == 0:
                return digit_graph + suffix + insert_space + sub_graph

            return digit_graph + suffix + (zero ** zeros_counts) + insert_space + sub_graph

        # Hundred graph
        suffix_hundreds = pynutil.insert(" सौ")
        graph_hundreds = create_graph_suffix(digit, suffix_hundreds, 2)
        graph_hundreds |= create_larger_number_graph(digit, suffix_hundreds, 1, digit)
        graph_hundreds |= create_larger_number_graph(digit, suffix_hundreds, 0, teens_ties)
        graph_hundreds.optimize()
        self.graph_hundreds = graph_hundreds

        # Transducer for eleven hundred -> 1100 or twenty one hundred eleven -> 2111
        graph_hundreds_as_thousand = create_graph_suffix(teens_and_ties, suffix_hundreds, 2)
        graph_hundreds_as_thousand |= create_larger_number_graph(teens_and_ties, suffix_hundreds, 1, digit)
        graph_hundreds_as_thousand |= create_larger_number_graph(teens_and_ties, suffix_hundreds, 0, teens_ties)
        self.graph_hundreds_as_thousand = graph_hundreds_as_thousand

        # Thousands and Ten thousands graph
        suffix_thousands = pynutil.insert(" हज़ार")
        graph_thousands = create_graph_suffix(digit, suffix_thousands, 3)
        graph_thousands |= create_larger_number_graph(digit, suffix_thousands, 2, digit)
        graph_thousands |= create_larger_number_graph(digit, suffix_thousands, 1, teens_ties)
        graph_thousands |= create_larger_number_graph(digit, suffix_thousands, 0, graph_hundreds)
        graph_thousands.optimize()
        self.graph_thousands = graph_thousands

        graph_ten_thousands = create_graph_suffix(teens_and_ties, suffix_thousands, 3)
        graph_ten_thousands |= create_larger_number_graph(teens_and_ties, suffix_thousands, 2, digit)
        graph_ten_thousands |= create_larger_number_graph(teens_and_ties, suffix_thousands, 1, teens_ties)
        graph_ten_thousands |= create_larger_number_graph(teens_and_ties, suffix_thousands, 0, graph_hundreds)
        graph_ten_thousands.optimize()

        # Lakhs graph and ten lakhs graph
        suffix_lakhs = pynutil.insert(" लाख")
        graph_lakhs = create_graph_suffix(digit, suffix_lakhs, 5)
        graph_lakhs |= create_larger_number_graph(digit, suffix_lakhs, 4, digit)
        graph_lakhs |= create_larger_number_graph(digit, suffix_lakhs, 3, teens_ties)
        graph_lakhs |= create_larger_number_graph(digit, suffix_lakhs, 2, graph_hundreds)
        graph_lakhs |= create_larger_number_graph(digit, suffix_lakhs, 1, graph_thousands)
        graph_lakhs |= create_larger_number_graph(digit, suffix_lakhs, 0, graph_ten_thousands)
        graph_lakhs.optimize()

        graph_ten_lakhs = create_graph_suffix(teens_and_ties, suffix_lakhs, 5)
        graph_ten_lakhs |= create_larger_number_graph(teens_and_ties, suffix_lakhs, 4, digit)
        graph_ten_lakhs |= create_larger_number_graph(teens_and_ties, suffix_lakhs, 3, teens_ties)
        graph_ten_lakhs |= create_larger_number_graph(teens_and_ties, suffix_lakhs, 2, graph_hundreds)
        graph_ten_lakhs |= create_larger_number_graph(teens_and_ties, suffix_lakhs, 1, graph_thousands)
        graph_ten_lakhs |= create_larger_number_graph(teens_and_ties, suffix_lakhs, 0, graph_ten_thousands)
        graph_ten_lakhs.optimize()

        # Crores graph ten crores graph
        suffix_crores = pynutil.insert(" करोड़")
        graph_crores = create_graph_suffix(digit, suffix_crores, 7)
        graph_crores |= create_larger_number_graph(digit, suffix_crores, 6, digit)
        graph_crores |= create_larger_number_graph(digit, suffix_crores, 5, teens_ties)
        graph_crores |= create_larger_number_graph(digit, suffix_crores, 4, graph_hundreds)
        graph_crores |= create_larger_number_graph(digit, suffix_crores, 3, graph_thousands)
        graph_crores |= create_larger_number_graph(digit, suffix_crores, 2, graph_ten_thousands)
        graph_crores |= create_larger_number_graph(digit, suffix_crores, 1, graph_lakhs)
        graph_crores |= create_larger_number_graph(digit, suffix_crores, 0, graph_ten_lakhs)
        graph_crores.optimize()

        graph_ten_crores = create_graph_suffix(teens_and_ties, suffix_crores, 7)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_crores, 6, digit)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_crores, 5, teens_ties)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_crores, 4, graph_hundreds)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_crores, 3, graph_thousands)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_crores, 2, graph_ten_thousands)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_crores, 1, graph_lakhs)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_crores, 0, graph_ten_lakhs)
        graph_ten_crores.optimize()

        # Arabs graph and ten arabs graph
        suffix_arabs = pynutil.insert(" अरब")
        graph_arabs = create_graph_suffix(digit, suffix_arabs, 9)
        graph_arabs |= create_larger_number_graph(digit, suffix_arabs, 8, digit)
        graph_arabs |= create_larger_number_graph(digit, suffix_arabs, 7, teens_ties)
        graph_arabs |= create_larger_number_graph(digit, suffix_arabs, 6, graph_hundreds)
        graph_arabs |= create_larger_number_graph(digit, suffix_arabs, 5, graph_thousands)
        graph_arabs |= create_larger_number_graph(digit, suffix_arabs, 4, graph_ten_thousands)
        graph_arabs |= create_larger_number_graph(digit, suffix_arabs, 3, graph_lakhs)
        graph_arabs |= create_larger_number_graph(digit, suffix_arabs, 2, graph_ten_lakhs)
        graph_arabs |= create_larger_number_graph(digit, suffix_arabs, 1, graph_crores)
        graph_arabs |= create_larger_number_graph(digit, suffix_arabs, 0, graph_ten_crores)
        graph_arabs.optimize()

        graph_ten_arabs = create_graph_suffix(teens_and_ties, suffix_arabs, 9)
        graph_ten_arabs |= create_larger_number_graph(teens_and_ties, suffix_arabs, 8, digit)
        graph_ten_arabs |= create_larger_number_graph(teens_and_ties, suffix_arabs, 7, teens_ties)
        graph_ten_arabs |= create_larger_number_graph(teens_and_ties, suffix_arabs, 6, graph_hundreds)
        graph_ten_arabs |= create_larger_number_graph(teens_and_ties, suffix_arabs, 5, graph_thousands)
        graph_ten_arabs |= create_larger_number_graph(teens_and_ties, suffix_arabs, 4, graph_ten_thousands)
        graph_ten_arabs |= create_larger_number_graph(teens_and_ties, suffix_arabs, 3, graph_lakhs)
        graph_ten_arabs |= create_larger_number_graph(teens_and_ties, suffix_arabs, 2, graph_ten_lakhs)
        graph_ten_arabs |= create_larger_number_graph(teens_and_ties, suffix_arabs, 1, graph_crores)
        graph_ten_arabs |= create_larger_number_graph(teens_and_ties, suffix_arabs, 0, graph_ten_crores)
        graph_ten_arabs.optimize()

        # Kharabs graph and ten kharabs graph
        suffix_kharabs = pynutil.insert(" खरब")
        graph_kharabs = create_graph_suffix(digit, suffix_kharabs, 11)
        graph_kharabs |= create_larger_number_graph(digit, suffix_kharabs, 10, digit)
        graph_kharabs |= create_larger_number_graph(digit, suffix_kharabs, 9, teens_ties)
        graph_kharabs |= create_larger_number_graph(digit, suffix_kharabs, 8, graph_hundreds)
        graph_kharabs |= create_larger_number_graph(digit, suffix_kharabs, 7, graph_thousands)
        graph_kharabs |= create_larger_number_graph(digit, suffix_kharabs, 6, graph_ten_thousands)
        graph_kharabs |= create_larger_number_graph(digit, suffix_kharabs, 5, graph_lakhs)
        graph_kharabs |= create_larger_number_graph(digit, suffix_kharabs, 4, graph_ten_lakhs)
        graph_kharabs |= create_larger_number_graph(digit, suffix_kharabs, 3, graph_crores)
        graph_kharabs |= create_larger_number_graph(digit, suffix_kharabs, 2, graph_ten_crores)
        graph_kharabs |= create_larger_number_graph(digit, suffix_kharabs, 1, graph_arabs)
        graph_kharabs |= create_larger_number_graph(digit, suffix_kharabs, 0, graph_ten_arabs)
        graph_kharabs.optimize()

        graph_ten_kharabs = create_graph_suffix(teens_and_ties, suffix_kharabs, 11)
        graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 10, digit)
        graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 9, teens_ties)
        graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 8, graph_hundreds)
        graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 7, graph_thousands)
        graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 6, graph_ten_thousands)
        graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 5, graph_lakhs)
        graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 4, graph_ten_lakhs)
        graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 3, graph_crores)
        graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 2, graph_ten_crores)
        graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 1, graph_arabs)
        graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 0, graph_ten_arabs)
        graph_ten_kharabs.optimize()

        # Nils graph and ten nils graph
        suffix_nils = pynutil.insert(" नील")
        graph_nils = create_graph_suffix(digit, suffix_nils, 13)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 12, digit)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 11, teens_ties)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 10, graph_hundreds)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 9, graph_thousands)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 8, graph_ten_thousands)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 7, graph_lakhs)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 6, graph_ten_lakhs)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 5, graph_crores)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 4, graph_ten_crores)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 3, graph_arabs)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 2, graph_ten_arabs)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 1, graph_kharabs)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 0, graph_ten_kharabs)
        graph_nils.optimize()

        graph_ten_nils = create_graph_suffix(teens_and_ties, suffix_nils, 13)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 12, digit)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 11, teens_ties)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 10, graph_hundreds)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 9, graph_thousands)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 8, graph_ten_thousands)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 7, graph_lakhs)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 6, graph_ten_lakhs)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 5, graph_crores)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 4, graph_ten_crores)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 3, graph_arabs)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 2, graph_ten_arabs)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 1, graph_kharabs)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 0, graph_ten_kharabs)
        graph_ten_nils.optimize()

        # Padmas graph and ten padmas graph
        suffix_padmas = pynutil.insert(" पद्म")
        graph_padmas = create_graph_suffix(digit, suffix_padmas, 15)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 14, digit)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 13, teens_ties)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 12, graph_hundreds)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 11, graph_thousands)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 10, graph_ten_thousands)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 9, graph_lakhs)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 8, graph_ten_lakhs)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 7, graph_crores)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 6, graph_ten_crores)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 5, graph_arabs)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 4, graph_ten_arabs)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 3, graph_kharabs)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 2, graph_ten_kharabs)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 1, graph_nils)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 0, graph_ten_nils)
        graph_padmas.optimize()

        graph_ten_padmas = create_graph_suffix(teens_and_ties, suffix_padmas, 15)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 14, digit)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 13, teens_ties)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 12, graph_hundreds)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 11, graph_thousands)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 10, graph_ten_thousands)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 9, graph_lakhs)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 8, graph_ten_lakhs)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 7, graph_crores)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 6, graph_ten_crores)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 5, graph_arabs)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 4, graph_ten_arabs)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 3, graph_kharabs)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 2, graph_ten_kharabs)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 1, graph_nils)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 0, graph_ten_nils)
        graph_ten_padmas.optimize()

        # Shankhs graph and ten shankhs graph
        suffix_shankhs = pynutil.insert(" शंख")
        graph_shankhs = create_graph_suffix(digit, suffix_shankhs, 17)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 16, digit)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 15, teens_ties)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 14, graph_hundreds)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 13, graph_thousands)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 12, graph_ten_thousands)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 11, graph_lakhs)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 10, graph_ten_lakhs)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 9, graph_crores)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 8, graph_ten_crores)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 7, graph_arabs)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 6, graph_ten_arabs)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 5, graph_kharabs)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 4, graph_ten_kharabs)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 3, graph_nils)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 2, graph_ten_nils)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 1, graph_padmas)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 0, graph_ten_padmas)
        graph_shankhs.optimize()

        graph_ten_shankhs = create_graph_suffix(teens_and_ties, suffix_shankhs, 17)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 16, digit)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 15, teens_ties)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 14, graph_hundreds)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 13, graph_thousands)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 12, graph_ten_thousands)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 11, graph_lakhs)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 10, graph_ten_lakhs)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 9, graph_crores)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 8, graph_ten_crores)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 7, graph_arabs)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 6, graph_ten_arabs)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 5, graph_kharabs)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 4, graph_ten_kharabs)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 3, graph_nils)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 2, graph_ten_nils)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 1, graph_padmas)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 0, graph_ten_padmas)
        graph_ten_shankhs.optimize()

        final_graph = (
            digit
            | zero
            | teens_and_ties
            | graph_hundreds
            | graph_thousands
            | graph_ten_thousands
            | graph_lakhs
            | graph_ten_lakhs
            | graph_crores
            | graph_ten_crores
            | graph_arabs
            | graph_ten_arabs
            | graph_kharabs
            | graph_ten_kharabs
            | graph_nils
            | graph_ten_nils
            | graph_padmas
            | graph_ten_padmas
            | graph_shankhs
            | graph_ten_shankhs
        )

        optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        self.final_graph = final_graph.optimize()
        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.final_graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph
