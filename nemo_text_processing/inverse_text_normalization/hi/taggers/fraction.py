# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2024 and onwards Google, Inc.
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

from nemo_text_processing.inverse_text_normalization.hi.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_CASED,
    INPUT_LOWER_CASED,
    MIN_NEG_WEIGHT,
    MINUS,
    NEMO_DIGIT,
    NEMO_SIGMA,
    TO_LOWER,
    GraphFst,
    capitalized_input_graph,
    delete_extra_space,
    delete_space,
)
from nemo_text_processing.text_normalization.en.utils import load_labels


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction
        Fraction "/" is determined by "बटा"
            e.g. ऋण एक बटा छब्बीस -> fraction { negative: "true" numerator: "१" denominator: "२६" }
            e.g. छह सौ साठ बटा पाँच सौ तैंतालीस -> fraction { negative: "false" numerator: "६६०" denominator: "५४३" }

 
        The fractional rule assumes that fractions can be pronounced as:
        (a cardinal) + ('बटा') plus (a cardinal, excluding 'शून्य')
    Args:
        cardinal: CardinalFst
        fraction: FractionFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="fraction", kind="classify")
        # integer_part # numerator # denominator
        graph_cardinal = cardinal.graph_no_exception

        integer = pynutil.insert("integer_part: \"") + graph_cardinal + pynutil.insert("\" ")
        integer += delete_space
        delete_bata = pynini.union(pynutil.delete(" बटा ") | pynutil.delete(" बटे "))

        numerator = pynutil.insert("numerator: \"") + graph_cardinal + pynutil.insert("\"")
        denominator = pynutil.insert(" denominator: \"") + graph_cardinal + pynutil.insert("\"")
        graph_fraction = numerator + delete_bata + denominator

        graph = graph_fraction
        self.graph = graph.optimize()
        self.final_graph_wo_negative = graph
        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("ऋण", "\"true\"") + delete_extra_space, 0, 1,
        )
        graph = optional_graph_negative + graph
        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
