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

from nemo_text_processing.inverse_text_normalization.ta.graph_utils import GraphFst
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_SPACE
from nemo_text_processing.text_normalization.ta.graph_utils import punctuation_code_points

# ASCII marks that Unicode does not categorise as punctuation.
_ASCII_MARKS = "!#%&'()*+,-./:;<=>?@^_`{|}~\""


class PunctuationFst(GraphFst):
    """
    Finite state transducer for classifying punctuation, e.g.
        a, -> tokens { name: "a" } tokens { name: "," }

    Markup such as <b> or </b> stays one token instead of splitting into marks.
    """

    def __init__(self):
        super().__init__(name="punctuation", kind="classify")

        self.punct_marks = punctuation_code_points() + list(_ASCII_MARKS)
        marks = pynini.union(*[pynini.escape(p) for p in self.punct_marks])
        punct = pynini.closure(marks, 1).optimize()

        tag_body = pynini.closure(NEMO_NOT_SPACE - pynini.union("<", ">"), 1)
        emphasis = (
            pynini.accep("<")
            + pynini.union(tag_body + pynini.closure(pynini.accep("/"), 0, 1), pynini.accep("/") + tag_body)
            + pynini.accep(">")
        ).optimize()
        rest = pynini.difference(pynini.project(punct, "input"), pynini.project(emphasis, "input"))
        punct = pynini.union(emphasis, pynini.compose(rest, punct)).optimize()

        self.graph = punct
        self.graph_input = punct.copy().project("input").optimize()
        self.fst = (pynutil.insert("name: \"") + self.graph + pynutil.insert("\"")).optimize()
