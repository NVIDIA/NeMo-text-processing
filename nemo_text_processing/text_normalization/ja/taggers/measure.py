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

from nemo_text_processing.text_normalization.ja.graph_utils import GraphFst, delete_space
from nemo_text_processing.text_normalization.ja.utils import get_abs_path, load_labels


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying Japanese measure expressions.

    Examples:
        5kg -> measure { cardinal { integer: "五" } units: "キロ" preserve_order: true }
        0kg -> measure { cardinal { integer: "ゼロ" } units: "キロ" preserve_order: true }
        0.05m -> measure { decimal { integer_part: "零" fractional_part: "零五" } units: "メートル" preserve_order: true }
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
        per_unit = pynini.string_file(per_unit_path)
        rate_numerator = pynini.string_file(rate_numerator_path)

        slash = pynutil.delete("/") | pynutil.delete("／")
        kilometer_rate_unit = rate_numerator @ pynini.cross("キロ", "")
        meter_rate_unit = rate_numerator @ pynini.cross("メートル", "")
        hour_per_unit = per_unit @ pynini.cross("時", "")
        second_per_unit = per_unit @ pynini.cross("秒", "")

        speed_kmh_inputs = {written for written, spoken in load_labels(rate_numerator_path) if spoken == "キロ"}
        speed_ms_inputs = {written for written, spoken in load_labels(rate_numerator_path) if spoken == "メートル"}
        general_per_unit = pynini.Fst()
        for unit_written, unit_spoken in load_labels(unit_path):
            for per_written, per_spoken in load_labels(per_unit_path):
                if (unit_written in speed_kmh_inputs and per_spoken == "時") or (
                    unit_written in speed_ms_inputs and per_spoken == "秒"
                ):
                    continue
                general_per_unit |= (
                    pynini.cross(unit_written, unit_spoken)
                    + delete_space
                    + slash
                    + delete_space
                    + pynutil.insert("毎")
                    + pynini.cross(per_written, per_spoken)
                )
        unit_graph = unit | general_per_unit

        speed_kmh_unit = (
            delete_space
            + kilometer_rate_unit
            + delete_space
            + slash
            + delete_space
            + hour_per_unit
            + pynutil.insert(' units: "キロ"')
        )
        speed_ms_unit = (
            delete_space
            + meter_rate_unit
            + delete_space
            + slash
            + delete_space
            + second_per_unit
            + pynutil.insert(' units: "メートル"')
        )

        unit_component = delete_space + pynutil.insert(' units: "') + unit_graph + pynutil.insert('"')

        optional_sign = (
            pynutil.insert('negative: "')
            + (pynini.cross("-", "マイナス") | pynini.accep("マイナス"))
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
            pynutil.insert("fraction { ")
            + pynini.closure(optional_sign, 0, 1)
            + fraction.graph
            + pynutil.insert(" }")
        )

        speed_kmh_number = (
            pynutil.insert("cardinal { ")
            + pynini.closure(optional_sign, 0, 1)
            + pynutil.insert('integer: "時速')
            + cardinal.just_cardinals
            + pynutil.insert('" }')
        )
        speed_ms_number = (
            pynutil.insert("cardinal { ")
            + pynini.closure(optional_sign, 0, 1)
            + pynutil.insert('integer: "秒速')
            + cardinal.just_cardinals
            + pynutil.insert('" }')
        )

        number = cardinal_graph | decimal_graph | fraction_graph

        speed_kmh_graph = (
            speed_kmh_number
            + speed_kmh_unit
            + pynutil.insert(" preserve_order: true")
        )
        speed_ms_graph = (
            speed_ms_number
            + speed_ms_unit
            + pynutil.insert(" preserve_order: true")
        )
        general_graph = number + unit_component + pynutil.insert(" preserve_order: true")

        graph = speed_kmh_graph | speed_ms_graph | general_graph

        self.fst = self.add_tokens(graph.optimize()).optimize()
