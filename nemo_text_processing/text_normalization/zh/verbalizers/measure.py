# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.zh.graph_utils import NEMO_NOT_QUOTE, NEMO_SPACE, GraphFst, delete_space


class MeasureFst(GraphFst):
    '''
    tokens { measure { cardinal: "一" } units: "千克" } } ->  一千克
    '''

    def __init__(
        self, cardinal: GraphFst, decimal: GraphFst, fraction: GraphFst, deterministic: bool = True, lm: bool = False
    ):
        super().__init__(name="measure", kind="verbalize", deterministic=deterministic)

        cardinal = cardinal.numbers
        decimal = decimal.decimal_component
        sign_component = pynutil.delete("negative: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        unit_component = pynutil.delete("units: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")

        graph_cardinal = (
            pynutil.delete("cardinal { ") + cardinal + pynutil.delete(" } ") + delete_space + unit_component
        )

        graph_decimal = (
            pynutil.delete("decimal {")
            + pynini.closure(pynutil.delete(NEMO_SPACE))
            + decimal
            + pynini.closure(pynutil.delete(NEMO_SPACE))
            + pynutil.delete("}")
            + pynini.closure(pynutil.delete(NEMO_SPACE))
            + delete_space
            + unit_component
        )

        graph_fraction = (
            pynutil.delete("fraction {")
            + pynini.closure(pynutil.delete(NEMO_SPACE))
            + fraction.fraction
            + pynini.closure(pynutil.delete(NEMO_SPACE))
            + pynutil.delete("}")
            + pynini.closure(pynutil.delete(NEMO_SPACE))
            + delete_space
            + unit_component
        )

        graph_math_cardinal = pynutil.delete("cardinal { ") + cardinal + pynutil.delete(" } ")

        graph_measures = graph_decimal | graph_cardinal | graph_fraction
        graph_maths = graph_math_cardinal

        final_graph = graph_maths | graph_measures

        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()
