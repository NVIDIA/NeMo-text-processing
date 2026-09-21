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
from nemo_text_processing.inverse_text_normalization.de.graph_utils import (
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_space,   
)


class DecimalFst(GraphFst):
    """
    Finite state transducer for verbalizing decimal, e.g.
        decimal { negative: "-" integer_part: "12"  fractional_part: "5006" quantity: "millionen" } -> -12,5006 Mio.
    """

    def __init__(self):
        super().__init__(name="decimal", kind="verbalize")

        # the tagger writes the sign itself, so the verbalizer just reads it out of the field
        optional_sign = pynini.closure(
            pynutil.delete('negative: "') + NEMO_NOT_QUOTE + pynutil.delete('"') + delete_space, 0, 1
        )
        
        fullstop_accep = pynini.accep(".")
        integer_chars = NEMO_DIGIT | fullstop_accep
        integer = (
            pynutil.delete("integer_part:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(integer_chars, 1)
            + pynutil.delete('"')
        )
        optional_integer = pynini.closure(integer + delete_space, 0, 1)
        
        fractional = (
            pynutil.insert(",")
            + pynutil.delete("fractional_part:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_DIGIT, 1)
            + pynutil.delete('"')
        )
        optional_fractional = pynini.closure(fractional + delete_space, 0, 1)
        
        quantity_chars = NEMO_ALPHA | fullstop_accep
        quantity = (
            pynutil.delete("quantity:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(quantity_chars, 1)
            + pynutil.delete('"')
        )
        optional_quantity = pynini.closure(pynutil.insert(" ") + quantity + delete_space, 0, 1)
        
        graph = (optional_integer + optional_fractional + optional_quantity).optimize()
        
        self.numbers = graph  # This part of the graph to be passed to other classes
        delete_tokens = self.delete_tokens(optional_sign + graph)
        self.fst = delete_tokens.optimize()