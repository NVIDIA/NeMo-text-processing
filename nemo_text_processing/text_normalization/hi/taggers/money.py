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

from nemo_text_processing.text_normalization.hi.graph_utils import GraphFst, insert_space
from nemo_text_processing.text_normalization.hi.utils import get_abs_path

currency_graph = pynini.string_file(get_abs_path("data/money/currency.tsv"))


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money, suppletive aware, e.g.
        ₹1 -> money { currency: "रुपए" integer_part: "एक" }
        ₹1.2 -> money { currency: "रुपए" integer_part: "एक" fractional_part: "दो" }
        
    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst):
        super().__init__(name="money", kind="classify")

        cardinal_graph = cardinal.final_graph

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\"") + insert_space, 0, 1,
        )
        self.currency = pynutil.insert("currency: \"") + currency_graph + pynutil.insert("\" ")
        self.interger = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\" ")
        self.fraction = pynutil.insert("fractional_part: \"") + cardinal_graph + pynutil.insert("\" ")

        graph_currencies = optional_graph_negative + self.currency + insert_space + self.interger
        graph_currencies |= (
            optional_graph_negative
            + self.currency
            + insert_space
            + self.interger
            + pynutil.delete(".")
            + insert_space
            + self.fraction
        )
        graph = graph_currencies
        self.graph = graph.optimize()
        final_graph = self.add_tokens(graph)
        self.fst = final_graph
