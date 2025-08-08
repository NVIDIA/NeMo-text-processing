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
        e.g. halv -> tokens { fraction { numerator: "1" denominator: "2" } }
        e.g. ett och en halv -> tokens { fraction { integer_part: "1" numerator: "1" denominator: "2" } }
        e.g. tre och fyra femtedelar -> tokens { fraction { integer_part: "3" numerator: "4" denominator: "5" } }

    Args:
        itn_cardinal_tagger: ITN cardinal tagger
        tn_fraction_tagger: TN fraction tagger
    """

    def __init__(
        self,
        itn_cardinal_tagger: GraphFst,
        tn_fraction_tagger: GraphFst,
        project_input: bool = False
    ):
        super().__init__(name="fraction", kind="classify", project_input=project_input)
        cardinal = itn_cardinal_tagger.graph_no_exception
        fractions = tn_fraction_tagger.fractions_any.invert().optimize()

        minus = pynini.cross("minus ", "-")
        optional_minus = pynini.closure(minus, 0, 1)
        
        # Need delete_space for proper space handling
        from nemo_text_processing.text_normalization.en.graph_utils import delete_space
        
        # Pattern 1: "fyra femtedelar" -> numerator: "4" denominator: "5"
        simple_fraction = (
            pynutil.insert("numerator: \"") + cardinal + pynutil.insert("\" ") +
            delete_space + 
            pynutil.insert("denominator: \"") + fractions + pynutil.insert("\"")
        )
        
        # Pattern 2: "tjugotre och fyra femtedelar" -> integer_part: "23" numerator: "4" denominator: "5" 
        mixed_fraction = (
            pynutil.insert("integer_part: \"") + optional_minus + cardinal + pynutil.insert("\" ") +
            pynutil.delete(" och ") + 
            pynutil.insert("numerator: \"") + cardinal + pynutil.insert("\" ") +
            delete_space +
            pynutil.insert("denominator: \"") + fractions + pynutil.insert("\"")
        )
        
        # Pattern 3: "två och halv" -> integer_part: "2" numerator: "1" denominator: "2"
        mixed_half = (
            pynutil.insert("integer_part: \"") + optional_minus + cardinal + pynutil.insert("\" ") +
            pynutil.delete(" och ") +
            pynutil.insert("numerator: \"") + pynini.cross("halv", "1") + pynutil.insert("\" ") +
            pynutil.insert("denominator: \"2\"")
        )
        
        # Pattern 4: "en halv" -> numerator: "1" denominator: "2"
        simple_half = (
            pynutil.insert("numerator: \"") + pynini.cross("en halv", "1") + pynutil.insert("\" ") +
            pynutil.insert("denominator: \"2\"")
        )
        
        # Pattern 5: Just "halv" -> numerator: "1" denominator: "2" 
        bare_half = (
            pynutil.insert("numerator: \"") + pynini.cross("halv", "1") + pynutil.insert("\" ") +
            pynutil.insert("denominator: \"2\"")
        )
        
        # Combine all patterns
        graph = pynini.union(
            mixed_fraction,
            simple_fraction, 
            mixed_half,
            simple_half,
            bare_half
        )
        
        # Use add_tokens() to create proper fraction tokens 
        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
