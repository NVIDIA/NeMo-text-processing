# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from nemo_text_processing.inverse_text_normalization.ger.graph_utils import (
    NEMO_DIGIT,
    NEMO_SPACE,
    NEMO_CHAR,
    delete_space,
    GraphFst,
)


class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing fractions
        e.g. fraction { integer_part: "1" numerator: "3" denominator: "4" } -> 1 3/4
    """

    def __init__(self):
        super().__init__(name="fraction", kind="verbalize")

        # Handles individual components of a fraction
        negative_sign = pynini.cross('negative: "-"', "-")
        fullstop_accep = pynini.accep(".")
        integer_chars = NEMO_DIGIT | fullstop_accep
        integer = (
            pynutil.delete("integer_part:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(integer_chars, 1)
            + pynutil.delete('"')
        )

        vinculum = pynutil.insert("/")

        numerator = (
            pynutil.delete("numerator:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(integer_chars, 1)
            + pynutil.delete('"')
        )

        denominator = (
            pynutil.delete("denominator:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(integer_chars, 1)
            + pynutil.delete('"')
        )

        morphosyntax = (
            pynutil.delete("morphosyntactic_features:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_CHAR, 1)
            + pynutil.delete('"')
        )

        # Builds the graph
        graph_fraction = numerator + delete_space + vinculum + denominator
        graph_integer_fraction = (
            integer + pynini.accep(NEMO_SPACE)
        ).ques + graph_fraction
        graph_morphosyntax = (
            graph_integer_fraction + (pynini.accep(NEMO_SPACE) + morphosyntax).ques
        )

        self.numbers = (graph_morphosyntax + delete_space).optimize()

        graph_negative_sign = (negative_sign + delete_space).ques + graph_morphosyntax
        delete_tokens = self.delete_tokens(graph_negative_sign)
        self.fst = delete_tokens.optimize()
