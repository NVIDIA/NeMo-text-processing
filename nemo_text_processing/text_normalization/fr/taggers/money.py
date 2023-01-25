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


class MoneyFst(GraphFst):
    # MAKE SURE ANY COMMENTS APPLY TO YOUR LANGUAGE
    """
    Finite state transducer for classifying money, e.g.
        "€1" -> money { currency_maj: "euro" integer_part: "un"}
        "€1,000" -> money { currency_maj: "euro" integer_part: "un" }
        "€1,001" -> money { currency_maj: "euro" integer_part: "un" fractional_part: "cero cero un" }
        "£1,4" -> money { integer_part: "una" currency_maj: "libra" fractional_part: "cuarenta" preserve_order: true }
               -> money { integer_part: "una" currency_maj: "libra" fractional_part: "cuarenta" currency_min: "penique" preserve_order: true }
        "0,01 £" -> money { fractional_part: "un" currency_min: "penique" preserve_order: true }
        "0,02 £" -> money { fractional_part: "dos" currency_min: "peniques" preserve_order: true }
        "£0,01 million" -> money { currency_maj: "libra" integer_part: "cero" fractional_part: "cero un" quantity: "million" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="classify", deterministic=deterministic)

        # DELETE THIS LINE WHEN YOU ADD YOUR GRAMMAR, MAKING SURE THAT YOUR GRAMMAR CONTAINS
        # A VARIABLE CALLED final_graph WITH AN FST COMPRISED OF ALL THE RULES
        final_graph = pynutil.insert("integer_part: \"") + pynini.closure(NEMO_NOT_SPACE, 1) + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()