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

from nemo_text_processing.text_normalization.hi.graph_utils import NEMO_SPACE, GraphFst


class RangeFst(GraphFst):
    """
    Finite state transducer for classifying Hindi ranges with dash "-"
    Examples:
        10-20 -> tokens { name: "दस से बीस" }
        1.5-2.5 -> tokens { name: "एक दशमलव पाँच से दो दशमलव पाँच" }
        $10-$20 -> tokens { money { ... } } tokens { name: "से" } tokens { money { ... } }

    Args:
        time: composed time tagger and verbalizer
        date: composed date tagger and verbalizer
        cardinal: composed cardinal tagger and verbalizer
        decimal: composed decimal tagger and verbalizer
        money: money tagger (not composed with its verbalizer)
        measure: composed measure tagger and verbalizer
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(
        self,
        time: GraphFst,
        date: GraphFst,
        cardinal: GraphFst,
        decimal: GraphFst,
        money: GraphFst,
        measure: GraphFst,
        deterministic: bool = True,
    ):
        super().__init__(name="range", kind="classify", deterministic=deterministic)

        # optimize the composed inputs once; the composed measure graph is large
        time, date, cardinal, decimal, measure = (g.optimize() for g in (time, date, cardinal, decimal, measure))

        delete_space = pynini.closure(pynutil.delete(NEMO_SPACE), 0, 1)

        dash_to_se = pynini.cross("-", " से ")

        range_time = time + delete_space + dash_to_se + delete_space + time
        range_date = date + delete_space + dash_to_se + delete_space + date
        range_cardinal = cardinal + delete_space + dash_to_se + delete_space + cardinal
        range_decimal = decimal + delete_space + dash_to_se + delete_space + decimal

        range_measure = measure + delete_space + dash_to_se + delete_space + measure
        range_measure = pynutil.add_weight(range_measure, -1.5)

        range_pattern = range_time | range_date | range_cardinal | range_decimal | range_measure

        range_pattern = pynutil.add_weight(range_pattern, 0.1)

        self.graph = range_pattern.optimize()

        graph = pynutil.insert('name: "') + self.graph + pynutil.insert('"')

        # Composing money with its verbalizer keeps one copy of the number graph per currency (the verbalizer
        # says the currency after the number). Emit two money tokens with "से" between them instead.
        money_separator = pynutil.insert(' } tokens { name: "से" } tokens { ')
        range_money = money + delete_space + pynutil.delete("-") + delete_space + money_separator + money
        range_money = pynutil.add_weight(range_money, 0.1)

        self.fst = (graph | range_money).optimize()
