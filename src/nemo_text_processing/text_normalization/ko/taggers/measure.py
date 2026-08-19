# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.ko.graph_utils import GraphFst, delete_space, insert_space
from nemo_text_processing.text_normalization.ko.utils import get_abs_path


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying Korean measure expressions.
        - 1kg       → measure { cardinal { integer: "일" } units: "킬로그램" }
        - 12.5km    → measure { decimal { integer_part: "십이" fractional_part: "오" } units: "킬로미터" }
        - 2/3m      → measure { fraction { numerator: "이" denominator: "삼" } units: "미터" }
        - 60km/h    → measure { cardinal { integer: "육십" } units: "킬로미터 퍼 시간" }

    This FST attaches measurement units (e.g., "킬로미터", "그램") to numeric expressions
    classified by the `cardinal`, `decimal`, or `fraction` subgraphs.

    Args:
        cardinal:  FST handling integer (cardinal) numbers.
        decimal:   FST handling decimal numbers (optional).
        fraction:  FST handling fractional numbers (optional).
        deterministic: If True, provides a single transduction path; otherwise allows multiple.
    """

    def __init__(
        self,
        cardinal: GraphFst,
        decimal: GraphFst = None,
        fraction: GraphFst = None,
        deterministic: bool = True,
    ):
        super().__init__(name="measure", kind="classify", deterministic=deterministic)

        # Numeric subgraphs
        graph_cardinal = cardinal.graph

        # Unit lexicon
        graph_unit = pynini.string_file(get_abs_path("data/measure/unit.tsv"))

        # Per-expression handling (e.g., km/h, m/s)
        opt_space = pynini.closure(delete_space, 0, 1)
        per = pynini.cross("/", "퍼") + opt_space + insert_space + graph_unit
        optional_per = pynini.closure(opt_space + insert_space + per, 0, 1)

        # Final unit FST produces either "<unit>" or "<unit> 퍼 <unit>"
        unit = pynutil.insert('units: "') + (graph_unit + optional_per | per) + pynutil.insert('"')

        minus_as_field = pynutil.insert('negative: "마이너스" ')
        consume_minus = pynini.cross("-", "") | pynini.cross("마이너스", "")

        # Optional minus field + removal of actual sign symbol or word
        optional_minus = pynini.closure(minus_as_field + consume_minus + opt_space, 0, 1)

        # Combine numeric and unit components
        pieces = []

        # 1) Cardinal form: e.g., "12kg"
        sub_cardinal = (
            pynutil.insert("cardinal { ")
            + pynutil.insert('integer: "')
            + graph_cardinal
            + delete_space
            + pynutil.insert('" } ')
            + unit
        )
        pieces.append(sub_cardinal)

        # 2) Decimal form: e.g., "12.5km"
        if decimal is not None:
            sub_decimal = (
                pynutil.insert("decimal { ")
                + optional_minus
                + decimal.just_decimal
                + delete_space
                + pynutil.insert(" } ")
                + unit
            )
            pieces.append(sub_decimal)

        # 3) Fraction form: e.g., "2/3m" or "삼분의 이 미터"
        if fraction is not None:
            sub_fraction = pynutil.insert("fraction { ") + fraction.graph + delete_space + pynutil.insert(" } ") + unit
            pieces.append(sub_fraction)

        # Union all supported numeric forms (cardinal | decimal | fraction)
        graph = pieces[0]
        for p in pieces[1:]:
            graph |= p

        # Final wrapping into tokens { measure { ... } }
        self.fst = self.add_tokens(graph).optimize()
