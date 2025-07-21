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

from nemo_text_processing.inverse_text_normalization.ko.graph_utils import NEMO_CHAR, GraphFst
from nemo_text_processing.inverse_text_normalization.ko.utils import get_abs_path

def get_counter(ordinal):
    suffix = pynini.string_file(get_abs_path("data/ordinals/counter_suffix.tsv"))
    numbers = ordinal
    res = numbers + pynutil.insert('" counter: "') + suffix 

    return res

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

        graph_digit = pynini.string_file(get_abs_path("data/ordinals/digit.tsv"))  # 1-9 in ordinals
        cardinal_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))  # 1-9 in cardinals

        graph_tens_prefix = pynini.cross("열", "1")  # First digit for tens
        graph_twenties_prefix = pynini.cross("스물", "2")  # First digit for twenties
        graph_thirties_prefix = pynini.cross("서른", "3")  # First digit for thirties

        # Below exclude regular 1 in ordinal and replace with a special 1. Like "first" in English
        # The special 1 is a unique ordinal case for Korean and does not repeat for 11, 21, 31
        graph_one = pynini.cross("한", "1")
        single_digits = pynini.project(graph_digit, "input").optimize()
        graph_one_acceptor = pynini.project(graph_one, "input").optimize()
        two_to_nine = pynini.difference(single_digits, graph_one_acceptor).optimize()
        graph_two_to_nine = two_to_nine @ graph_digit
        graph_first = pynini.cross("첫", "1")
        graph_single = graph_two_to_nine | graph_first


        graph_ten = pynini.cross("열", "10")
        graph_tens = graph_ten | graph_tens_prefix + graph_digit

        graph_twenty = pynini.cross("스무", "20")
        graph_twenties = graph_twenty | graph_twenties_prefix + graph_digit

        graph_thirty = pynini.cross("서른", "30")
        graph_thirties = graph_thirty | graph_thirties_prefix + graph_digit

        ordinals = pynini.union(
            graph_single, graph_tens, graph_twenties, graph_thirties  # 1-9  # 10-19  # 20-29  # 30-39
        ).optimize()

        cardinal_10_to_19 = pynini.cross("십", "10") | (pynini.accep("십") + cardinal_digit)

        cardinal_20_to_29 = pynini.cross("이십", "20") | (pynini.accep("이십") + cardinal_digit)

        cardinal_30_to_39 = pynini.cross("삼십", "30") | (pynini.accep("삼십") + cardinal_digit)

        # FST that include 1-39 in cardinal expression
        cardinal_below_40 = pynini.union(
            cardinal_digit, cardinal_10_to_19, cardinal_20_to_29, cardinal_30_to_39
        ).optimize()

        # Input includes all cardinal expressions
        cardinals_acceptor = pynini.project(cardinals, "input").optimize()
        # Input includes cardinal expression from 1 to 39
        cardinals_exception = pynini.project(
            cardinal_below_40, "input"
        ).optimize()  

        # All cardinal values except 1 to 39 cardinal values
        cardinal_over_40 = pynini.difference(
            cardinals_acceptor, cardinals_exception
        ).optimize()  
        cardinal_ordinal_suffix = cardinal_over_40 @ cardinals

        # 1 to 39 in ordinal, everything else cardinal
        ordinal_final = pynini.union(ordinals, cardinal_ordinal_suffix)  

        ordinal_graph = pynutil.insert("integer: \"") + ((ordinal_final + ordinals_suffix)) + pynutil.insert("\"")

        #Adding various counter suffix for ordinal
        # For counting, Korean does not use the speical "첫" for 1. Instead the regular "한"
        counters = pynini.union(
            graph_digit, graph_tens, graph_twenties, graph_thirties
        ).optimize()
        
        counter_final = (get_counter(counters) | get_counter(cardinal_ordinal_suffix))

        counter_graph = pynutil.insert("integer: \"") + counter_final + pynutil.insert("\"")

        final_graph = (ordinal_graph | counter_graph)

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
