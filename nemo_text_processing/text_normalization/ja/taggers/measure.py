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

from nemo_text_processing.text_normalization.ja.graph_utils import GraphFst, delete_space
from nemo_text_processing.text_normalization.ja.utils import get_abs_path, load_labels


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying Japanese measure expressions.

    Examples:
        5kg -> measure { cardinal { integer: "五" } units: "キロ" preserve_order: true }
        0kg -> measure { cardinal { integer: "ゼロ" } units: "キロ" preserve_order: true }
        0.05m
        -> measure { decimal { integer_part: "零" fractional_part: "零五" } units: "メートル" preserve_order: true }
        60km/h -> measure { cardinal { integer: "時速六十" } units: "キロ" preserve_order: true }
        50m/s -> measure { cardinal { integer: "秒速五十" } units: "メートル" preserve_order: true }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        fraction: FractionFst
        deterministic: if True provides a single transduction option
    """

    def __init__(
        self,
        cardinal: GraphFst,
        decimal: GraphFst,
        fraction: GraphFst,
        deterministic: bool = True,
    ):
        super().__init__(name="measure", kind="classify", deterministic=deterministic)

        unit_path = get_abs_path("data/measure/unit.tsv")
        per_unit_path = get_abs_path("data/measure/per_unit.tsv")
        rate_numerator_path = get_abs_path("data/measure/rate_numerator.tsv")
        unit = pynini.string_file(unit_path)
        rate_numerator_labels = load_labels(rate_numerator_path)
        per_unit_labels = load_labels(per_unit_path)
        speed_configs = {
            (unit_spoken, per_spoken): prefix
            for unit_spoken, per_spoken, prefix in load_labels(get_abs_path("data/measure/speed.tsv"))
        }
        per_marker = load_labels(get_abs_path("data/measure/per_marker.tsv"))[0][0]

        slash = pynutil.delete("/") | pynutil.delete("／")

        general_per_unit = pynini.Fst()
        for unit_written, unit_spoken in load_labels(unit_path):
            for per_written, per_spoken in per_unit_labels:
                general_per_unit |= (
                    pynini.cross(unit_written, unit_spoken)
                    + delete_space
                    + slash
                    + delete_space
                    + pynutil.insert(per_marker)
                    + pynini.cross(per_written, per_spoken)
                )
        unit_graph = unit | general_per_unit

        unit_component = delete_space + pynutil.insert(' units: "') + unit_graph + pynutil.insert('"')

        optional_sign = (
            pynutil.insert('negative: "')
            + pynini.string_file(get_abs_path("data/numbers/sign.tsv"))
            + pynutil.insert('" ')
            + delete_space
        )

        cardinal_graph = (
            pynutil.insert("cardinal { ")
            + pynini.closure(optional_sign, 0, 1)
            + pynutil.insert('integer: "')
            + cardinal.just_cardinals
            + pynutil.insert('" }')
        )
        decimal_graph = (
            pynutil.insert("decimal { ")
            + pynini.closure(optional_sign, 0, 1)
            + decimal.just_decimal
            + pynutil.insert(" }")
        )
        fraction_graph = (
            pynutil.insert("fraction { ") + pynini.closure(optional_sign, 0, 1) + fraction.graph + pynutil.insert(" }")
        )

        number = cardinal_graph | decimal_graph | fraction_graph

        speed_graph = pynini.Fst()
        for unit_written, unit_spoken in rate_numerator_labels:
            for per_written, per_spoken in per_unit_labels:
                prefix = speed_configs.get((unit_spoken, per_spoken))
                if prefix is None:
                    continue
                speed_number = (
                    pynutil.insert("cardinal { ")
                    + pynini.closure(optional_sign, 0, 1)
                    + pynutil.insert(f'integer: "{prefix}')
                    + cardinal.just_cardinals
                    + pynutil.insert('" }')
                )
                speed_unit = (
                    delete_space
                    + pynutil.delete(unit_written)
                    + delete_space
                    + slash
                    + delete_space
                    + pynutil.delete(per_written)
                    + pynutil.insert(f' units: "{unit_spoken}"')
                )
                speed_graph |= speed_number + speed_unit + pynutil.insert(" preserve_order: true")

        general_graph = number + unit_component + pynutil.insert(" preserve_order: true")

        graph = pynutil.add_weight(speed_graph, -0.1) | general_graph

        self.fst = self.add_tokens(graph.optimize()).optimize()
