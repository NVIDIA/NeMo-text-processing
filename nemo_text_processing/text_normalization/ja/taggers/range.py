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

from nemo_text_processing.text_normalization.ja.graph_utils import NEMO_DIGIT, GraphFst, delete_space
from nemo_text_processing.text_normalization.ja.utils import get_abs_path


class RangeFst(GraphFst):
    """
    Finite state transducer for classifying Japanese ranges.

    Examples:
        2-5 -> tokens { name: "二から五" }
        10:00-11:00 -> tokens { name: "十時から十一時" }
        3kg-6kg -> tokens { name: "三キロから六キロ" }

    Args:
        cardinal: composed cardinal tagger and verbalizer
        date: composed date tagger and verbalizer
        time: composed time tagger and verbalizer
        money: composed money tagger and verbalizer
        measure: composed measure tagger and verbalizer
        deterministic: if True will provide a single transduction option,
            for False multiple transductions are generated
    """

    def __init__(
        self,
        cardinal: GraphFst,
        date: GraphFst,
        time: GraphFst,
        money: GraphFst,
        measure: GraphFst,
        deterministic: bool = True,
    ):
        super().__init__(name="range", kind="classify", deterministic=deterministic)

        separator = pynini.string_file(get_abs_path("data/range/separator.tsv"))
        sep_to_kara = delete_space + separator + delete_space

        endpoint = cardinal | date | time | money | measure
        graph = endpoint + sep_to_kara + endpoint

        # Some range suffixes, such as 人 and 歳, do not have a dedicated
        # semiotic class. Keep them as range-specific cardinal patterns.
        suffix = pynini.string_file(get_abs_path("data/range/suffix.tsv"))
        cardinal_pair = cardinal + sep_to_kara + cardinal
        cardinal_range = cardinal_pair + pynini.closure(suffix, 0, 1)
        graph |= cardinal_range
        graph |= cardinal + pynini.string_file(get_abs_path("data/range/operator.tsv")) + cardinal

        # Normalize English-style decade suffixes before reusing the date graph.
        year_alias = (NEMO_DIGIT**4 + pynini.string_file(get_abs_path("data/range/year_suffix.tsv"))) @ date
        graph |= pynutil.add_weight(year_alias + sep_to_kara + year_alias, -0.5)

        self.graph = graph.optimize()
        self.fst = (pynutil.insert('name: "') + self.graph + pynutil.insert('"')).optimize()
