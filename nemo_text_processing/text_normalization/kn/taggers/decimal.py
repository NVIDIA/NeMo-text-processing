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
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.kn.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.kn.utils import get_abs_path


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal, e.g.
       -೧೨.೫೬ -> decimal { negative: "true" integer_part: "ಹನ್ನೆರಡು" fractional_part: "ಐದು ಆರು" }

    cardinal: CardinalFst
    """

    def __init__(self, cardinal: CardinalFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        graph_digit = cardinal.single_digits_graph
        cardinal_graph = cardinal.final_graph
        _ZEROS = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))

        def _digit_graph(digits_fst):
            return pynini.compose(digits_fst, graph_digit)

        def _group(d):
            return d + pynini.closure(insert_space + d)

        ed = _digit_graph(NEMO_DIGIT)
        kd = _digit_graph(pynini.difference(NEMO_ALL_DIGIT, NEMO_DIGIT))

        frac = (_group(ed) | _group(kd)).optimize()

        point = pynutil.delete(PERIOD)
        opt_neg = pynini.closure(pynutil.insert("negative: ") + pynini.cross(MINUS, '"true"') + insert_space, 0, 1)

        fractional = pynutil.insert('fractional_part: "') + frac + pynutil.insert('"')
        integer = pynutil.insert('integer_part: "') + cardinal_graph + pynutil.insert('"')

        integer_leadingzero = _ZEROS + insert_space + frac
        leadingzero_graph = (
            pynutil.insert('integer_part: "')
            + integer_leadingzero
            + pynini.cross(PERIOD, " . ")
            + frac
            + pynutil.insert('"')
        )

        graph_with_integer = leadingzero_graph | (integer + point + insert_space + fractional)
        graph_without_integer = point + fractional

        final = opt_neg + (graph_with_integer | graph_without_integer)

        self.fst = self.add_tokens(final).optimize()
