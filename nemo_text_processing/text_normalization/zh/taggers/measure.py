# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from nemo_text_processing.text_normalization.zh.graph_utils import GraphFst, insert_space
from nemo_text_processing.text_normalization.zh.utils import get_abs_path
from pynini.lib import pynutil


class Measure(GraphFst):
    '''
        1kg  -> tokens { measure { cardinal { integer: "一" } units: "千克" } }
    '''

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, deterministic: bool = True, lm: bool = False):
        super().__init__(name="measure", kind="classify", deterministic=deterministic)

        units_en = pynini.string_file(get_abs_path("data/measure/units_en.tsv"))
        units_zh = pynini.string_file(get_abs_path("data/measure/units_zh.tsv"))

        graph_cardinal = cardinal.just_cardinals
        integer_component = pynutil.insert("integer: \"") + graph_cardinal + pynutil.insert("\"")
        unit_component = pynutil.insert("units: \"") + (units_en | units_zh) + pynutil.insert("\"")
        graph_cardinal_measure = integer_component + insert_space + unit_component

        decimal = decimal.decimal
        graph_decimal = (
            decimal + insert_space + pynutil.insert("units: \"") + (units_en | units_zh) + pynutil.insert("\"")
        )

        graph_sign = (
            (pynutil.insert("negative: \"") + pynini.accep("负") + pynutil.insert("\""))
            | (pynutil.insert("negative: \"") + pynini.cross("負", "负") + pynutil.insert("\""))
            | (pynutil.insert("negative: \"") + pynini.cross("-", "负") + pynutil.insert("\""))
        )

        graph = pynini.closure(graph_sign + insert_space) + (graph_cardinal_measure | graph_decimal)

        self.fst = self.add_tokens(graph).optimize()
