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
from nemo_text_processing.text_normalization.zh.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space
from pynini.lib import pynutil


class Measure(GraphFst):
    '''
        tokens { measure { cardinal: "一" } units: "千克" } } ->  一千克
    '''

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="measure", kind="verbalize", deterministic=deterministic)

        sign_component = pynutil.delete("negative: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        integer_component = pynutil.delete("integer: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        unit_component = pynutil.delete("units: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")

        cardinal_graph = integer_component + delete_space + unit_component

        decimal_graph = (
            pynutil.delete("integer_part: \"")
            + pynini.closure(NEMO_NOT_QUOTE)
            + pynutil.delete("\"")
            + pynutil.insert("点")
            + delete_space
            + pynutil.delete("fractional_part: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 0)
            + pynutil.delete("\"")
            + delete_space
            + pynutil.delete("units: \"")
            + pynini.closure(NEMO_NOT_QUOTE)
            + pynutil.delete("\"")
        )

        graph = pynini.closure(sign_component + delete_space) + (cardinal_graph | decimal_graph)

        self.fst = self.delete_tokens(graph).optimize()
