# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from nemo_text_processing.text_normalization.en.graph_utils import (
GraphFst,
get_abs_path,
NEMO_SIGMA,
NEMO_LOWER,
delete_space,
delete_extra_space
)
from nemo_text_processing.text_normalization.en.taggers.decimal import DecimalFst
from nemo_text_processing.text_normalization.en.taggers.fraction import FractionFst
from nemo_text_processing.text_normalization.en.taggers.cardinal import CardinalFst
from pynini.lib import pynutil


class MathFst(GraphFst):
    """
    Finite state transducer for classifying math


    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: CardinalFst, decimal: DecimalFst, fraction:FractionFst, deterministic: bool = True):
        super().__init__(name="math", kind="classify", deterministic=deterministic)
        delete_spaces=pynini.closure(pynutil.delete(" "),1)
        graph_var = pynutil.insert("name: \"") + NEMO_LOWER + pynutil.insert("\"")
        graph_terms = pynutil.add_weight(fraction.fst,-0.01)|decimal.fst|cardinal.fst|graph_var
        symbol_graph = pynutil.add_weight(pynini.string_file(get_abs_path("data/whitelist/math.tsv")),0.001)
        insert_token = pynutil.insert(" } ") + pynutil.insert("tokens { ")
        graph_symbol =  pynutil.insert("name: \"") + delete_spaces + symbol_graph + delete_spaces + pynutil.insert("\"")
        final_graph = graph_terms + pynini.closure(insert_token + delete_space + graph_symbol + insert_token + delete_space + graph_terms, 2)


        self.fst = final_graph.optimize()



