# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
    NEMO_DIGIT,
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_space,
)


class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing fraction, e.g.
        fraction { numerator: "1" denominator: "2" } -> 1/2
        fraction { negative: "-" integer_part: "1" numerator: "3" denominator: "4" } -> -1 3/4
    """

    def __init__(self):
        super().__init__(name="fraction", kind="verbalize")

        optional_sign = pynini.closure(
            pynutil.delete('negative: "') + NEMO_NOT_QUOTE + pynutil.delete('"') + delete_space, 0, 1
        )

        fullstop_accep = pynini.accep(".")
        integer = (
            pynutil.delete("integer_part:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_DIGIT | fullstop_accep, 1)
            + pynutil.delete('"')
        )
        optional_integer = pynini.closure(integer + delete_space + pynutil.insert(" "), 0, 1)

        numerator = (
            pynutil.delete("numerator:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_DIGIT, 1)
            + pynutil.delete('"')
        )

        denominator = (
            pynutil.insert("/")
            + pynutil.delete("denominator:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_DIGIT, 1)
            + pynutil.delete('"')
        )

        graph = optional_integer + numerator + delete_space + denominator

        self.fst = self.delete_tokens(optional_sign + graph).optimize()
