# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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

from nemo_text_processing.inverse_text_normalization.ko.graph_utils import NEMO_CHAR, GraphFst
from nemo_text_processing.inverse_text_normalization.ko.utils import get_abs_path


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinal
        Expressing integers in ordinal way for 1-39 and cardinal for 40+ due to Korean grammar.
        e.g. 스물세번째 -> ordinal {integer: "23", 23번째}
        e.g. 사십오번째 -> ordinal but the integer part is written in cardinal(due to korean grammar)
        { integer: "45", 45번쨰}
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="ordinal", kind="classify")

        cardinals = cardinal.just_cardinals
        ordinals_suffix = pynini.accep("번째")  # Korean ordinal's morphosyntactic feature

        graph_digit = pynini.string_file(get_abs_path("data/ordinals/digit.tsv"))  # 1-9
        graph_digit_no_one = pynini.string_file(get_abs_path("data/ordinals/digit_no_one.tsv"))  # 2-9
        cardinal_1to39 = pynini.string_file(get_abs_path("data/ordinals/cardinal_digit.tsv"))  # 1-39 in cardinals

        graph_tens_prefix = pynini.cross("열", "1")  # First digit for tens
        graph_twenties_prefix = pynini.cross("스물", "2")  # First digit for twenties
        graph_thirties_prefix = pynini.cross("서른", "3")  # First digit for thirties

        graph_one = pynini.cross("첫", "1")
        graph_single = graph_one | graph_digit_no_one
        # 1 has a unique ordinal case in Korean and does not repeat for 11, 21, 31

        graph_ten = pynini.cross("열", "10")
        graph_tens = graph_ten | graph_tens_prefix + graph_digit

        graph_twenty = pynini.cross("스무", "20")
        graph_twenties = graph_twenty | graph_twenties_prefix + graph_digit

        graph_thirty = pynini.cross("서른", "30")
        graph_thirties = graph_thirty | graph_thirties_prefix + graph_digit

        ordinals = pynini.union(
            graph_single, graph_tens, graph_twenties, graph_thirties  # 1-9  # 10-19  # 20-29  # 30-39
        ).optimize()

        cardinals_acceptor = pynini.project(cardinals, "input").optimize()  # Input includes all cardinal expressions
        cardinals_exception = pynini.project(
            cardinal_1to39, "input"
        ).optimize()  # Input includes cardinal expression from 1 to 39

        cardinal_plus_40 = pynini.difference(
            cardinals_acceptor, cardinals_exception
        ).optimize()  # All cardinal values - 1 to 39 cardinal values
        cardinal_ordinal = cardinal_plus_40 @ cardinals

        ordinal_final = pynini.union(ordinals, cardinal_ordinal)  # 1 to 39 in ordinal, everything else cardinal

        ordinal_graph = pynutil.insert("integer: \"") + ((ordinal_final + ordinals_suffix)) + pynutil.insert("\"")

        final_graph = self.add_tokens(ordinal_graph)
        self.fst = final_graph.optimize()
