# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright (c) 2023, Jim O'Regan for Språkbanken Tal
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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_SPACE, GraphFst, convert_space


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction
        e.g. halv -> tokens { name: "1/2" }
        e.g. ett och en halv -> tokens { name: "1 1/2" }
        e.g. tre och fyra femtedelar -> tokens { name: "3 4/5" }

    Args:
        itn_cardinal_tagger: ITN cardinal tagger
        tn_fraction_verbalizer: TN fraction verbalizer
    """

    def __init__(self, itn_cardinal_tagger: GraphFst, tn_fraction_tagger: GraphFst):
        super().__init__(name="fraction", kind="classify")
        cardinal = itn_cardinal_tagger.graph_no_exception
        fractions = tn_fraction_tagger.fractions_any.invert().optimize()

        minus = pynini.cross("minus ", "-")
        optional_minus = pynini.closure(minus, 0, 1)
        no_numerator = pynini.cross("och ", "1/")
        integer = optional_minus + cardinal

        self.graph = pynini.union(
            integer + NEMO_SPACE + no_numerator + fractions,
            integer + NEMO_SPACE + cardinal + pynini.cross(" ", "/") + fractions,
            integer + pynini.cross(" och ", " ") + cardinal + pynini.cross(" ", "/") + fractions,
            integer + pynini.cross(" och ", " ") + pynini.cross("en halv", "1/2"),
            cardinal + pynini.cross(" ", "/") + fractions,
        )

        graph = pynutil.insert("name: \"") + convert_space(self.graph) + pynutil.insert("\"")
        self.fst = graph.optimize()
