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

from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_LOWER_CASED,
    NEMO_SIGMA,
    GraphFst,
    convert_space,
)
from nemo_text_processing.text_normalization.en.utils import load_labels
from nemo_text_processing.text_normalization.ta.utils import get_abs_path, table_fst


class WhiteListFst(GraphFst):
    """
    Finite state transducer for classifying whitelist entries, e.g.
        டாக். -> tokens { name: "டாக்டர்" }
        % -> tokens { name: "சதவீதம்" }

    Reads ``data/whitelist/abbreviations.tsv`` and ``data/whitelist/symbol.tsv``; a symbol
    is spoken wherever it stands, except the slash, which the fraction and measure classes own.

    Args:
        input_case: accepting either "lower_cased" or "cased" input.
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
        input_file: path to a file with whitelist replacements, added to the default tables
    """

    def __init__(self, input_case: str, deterministic: bool = True, input_file: str = None):
        super().__init__(name="whitelist", kind="classify", deterministic=deterministic)

        def _get_whitelist_graph(file: str) -> 'pynini.FstLike':
            whitelist = [row for row in load_labels(file) if len(row) >= 2]
            if input_case == INPUT_LOWER_CASED:
                whitelist = [[x.lower(), y] for x, y, *_ in whitelist]
            else:
                whitelist = [[x, y] for x, y, *_ in whitelist]
            return pynini.string_map(whitelist)

        graph = _get_whitelist_graph(get_abs_path("data/whitelist/abbreviations.tsv"))
        graph |= pynini.compose(
            pynini.difference(NEMO_SIGMA, pynini.accep("/")).optimize(),
            table_fst(get_abs_path("data/whitelist/symbol.tsv")),
        ).optimize()

        if input_file:
            graph |= _get_whitelist_graph(input_file)

        self.graph = convert_space(graph).optimize()
        self.fst = (pynutil.insert("name: \"") + self.graph + pynutil.insert("\"")).optimize()
