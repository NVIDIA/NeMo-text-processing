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

from nemo_text_processing.text_normalization.kn.graph_utils import (
    MINUS,
    NEMO_ALL_DIGIT,
    NEMO_DIGIT,
    PERIOD,
    COMMA,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.kn.utils import get_abs_path
_ZEROS = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))

class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal, e.g.
       -೧೨.೫೬ -> decimal { negative: "true" integer_part: "ಹನ್ನೆರಡು" fractional_part: "ಐದು ಆರು" }

    cardinal: GraphFst
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        graph_digit = cardinal.single_digits_graph
        cardinal_graph = cardinal.final_graph

        dot_delete = pynutil.delete(PERIOD)
        literal_dot = insert_space + pynini.accep(PERIOD) + insert_space
        opt_neg = pynini.closure(pynutil.insert("negative: ") + pynini.cross(MINUS, '"true"') + insert_space, 0, 1)

        def spell_digits(digits_fst):
            one = pynini.compose(digits_fst, graph_digit)
            return (one + pynini.closure(insert_space + one)).optimize()

        def same_script(digits_fst):
            frac = spell_digits(digits_fst)
            zero_char = pynini.project(pynini.compose(digits_fst, _ZEROS), "input")
            leading_zero_shape = (zero_char + pynini.closure(digits_fst, 1)).optimize()

            integer_domain = pynini.difference(pynini.closure(digits_fst | COMMA, 1), leading_zero_shape)
            integer = pynini.compose(integer_domain, cardinal_graph).optimize()
            leading_zero_reading = pynini.compose(leading_zero_shape, frac).optimize()

            with_leading_zero = (
                pynutil.insert('integer_part: "') + leading_zero_reading + literal_dot + frac + pynutil.insert('"')
            )
            with_integer = (
                pynutil.insert('integer_part: "')
                + integer
                + pynutil.insert('"')
                + dot_delete
                + insert_space
                + pynutil.insert('fractional_part: "')
                + frac
                + pynutil.insert('"')
            )
            fraction_only = dot_delete + pynutil.insert('fractional_part: "') + frac + pynutil.insert('"')

            return with_leading_zero | with_integer | fraction_only

        graph_same_script = (
            same_script(NEMO_DIGIT) | same_script(pynini.difference(NEMO_ALL_DIGIT, NEMO_DIGIT))
        ).optimize()

        shape = pynini.closure(NEMO_ALL_DIGIT | COMMA, 1) + pynini.accep(PERIOD) + pynini.closure(NEMO_ALL_DIGIT, 1)
        passthrough = (
            pynutil.insert('integer_part: "')
            + pynini.difference(shape, pynini.project(graph_same_script, "input"))
            + pynutil.insert('"')
        )

        final_graph = (opt_neg + graph_same_script) | (pynini.closure(pynini.accep(MINUS), 0, 1) + passthrough)

        self.fst = self.add_tokens(final_graph.optimize()).optimize()