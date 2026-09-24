# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.te.graph_utils import (
    NEMO_ALL_DIGIT,
    NEMO_DIGIT,
    GraphFst,
    insert_space,
)

class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal, e.g.
        -౧౨.౫౦౦౬ -> decimal { negative: "true" integer_part: "పన్నెండు" fractional_part: "ఐదు సున్నా సున్నా ఆరు" }

    The integer part uses CardinalFst and is covered up to 19 digits
    (through hundred crore crores (వంద కోట్ల కోట్లు) / 10^17).
    The fractional part is read digit by digit with no magnitude limit.
    Integer and fraction must be the same script: all ASCII or all Telugu.

    cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        te_digit = pynini.difference(NEMO_ALL_DIGIT, NEMO_DIGIT).optimize()
        comma = pynini.accep(",")
        point = pynutil.delete(".")
        optional_sign = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", '"true" '),
            0,
            1,
        )

        def same_script(zero, digits):
            nonzero = pynini.difference(digits, zero)
            integer = pynini.compose(
                pynini.accep(zero) | (nonzero + pynini.closure(digits | comma)),
                cardinal.final_graph,
            )
            fraction = pynini.compose(pynini.closure(digits, 1), cardinal.single_digits_graph)
            return (
                pynutil.insert('integer_part: "')
                + integer
                + pynutil.insert('"')
                + point
                + insert_space
                + pynutil.insert('fractional_part: "')
                + fraction
                + pynutil.insert('"')
            )

        def keep(integer_digits, fraction_digits):
            return (
                pynini.closure(integer_digits | comma, 1)
                + pynini.accep(".")
                + pynini.closure(fraction_digits, 1)
            )

        final_graph = optional_sign + (same_script("0", NEMO_DIGIT) | same_script("౦", te_digit))
        mixed = pynini.closure(pynini.accep("-"), 0, 1) + (
            keep(NEMO_DIGIT, te_digit) | keep(te_digit, NEMO_DIGIT)
        )
        mixed = pynutil.insert('name: "') + mixed + pynutil.insert('"')
        self.fst = (self.add_tokens(final_graph) | mixed).optimize()
