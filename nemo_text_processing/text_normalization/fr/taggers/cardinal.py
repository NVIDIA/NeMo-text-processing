# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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


# MAKE SURE ALL IMPORTS FROM A LANGUAGE OTHER THAN ENGLISH ARE IN THE CORRECT LANGUAGE
import pynini
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_SPACE, GraphFst
from nemo_text_processing.text_normalization.fr.utils import get_abs_path
from pynini.lib import pynutil


class CardinalFst(GraphFst):
    # MAKE SURE ANY COMMENTS APPLY TO YOUR LANGUAGE
    """
    Finite state transducer for classifying cardinals, e.g.
        "1000" ->  cardinal { integer: "mil" }
        "2.000.000" -> cardinal { integer: "dos millones" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        # DELETE THIS LINE WHEN YOU ADD YOUR GRAMMAR, MAKING SURE THAT YOUR GRAMMAR CONTAINS
        # A VARIABLE CALLED final_graph WITH AN FST COMPRISED OF ALL THE RULES
        digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        final_graph = pynutil.insert("integer: \"") + digit + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
