# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, GraphFst
from pynini.lib import pynutil


class FractionFst(GraphFst):
    # MAKE SURE ANY COMMENTS APPLY TO YOUR LANGUAGE
    """
	Finite state transducer for verbalizing fraction
		e.g. tokens { fraction { integer: "treinta y tres" numerator: "cuatro" denominator: "quinto" } } ->
            treinta y tres y cuatro quintos


	Args:
		deterministic: if True will provide a single transduction option,
			for False multiple transduction are generated (used for audio-based normalization)
	"""

    def __init__(self, deterministic: bool = True):
        super().__init__(name="fraction", kind="verbalize", deterministic=deterministic)

        # DELETE THIS LINE WHEN YOU ADD YOUR GRAMMAR, MAKING SURE THAT YOUR GRAMMAR CONTAINS
        # A VARIABLE CALLED final_graph WITH AN FST COMPRISED OF ALL THE RULES
        final_graph = pynutil.delete("integer_part: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")

        self.fst = self.delete_tokens(final_graph).optimize()