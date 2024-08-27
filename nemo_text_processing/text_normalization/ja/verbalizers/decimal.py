# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.ja.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space


class DecimalFst(GraphFst):
    """
    Finite state transducer for verbalizing decimal, e.g.
        decimal { integer_part: "零" fractional_part: "五" } -> 零点五
        decimal { positive: "-" integer_part: "零" fractional_part: "五" } -> マイナス零点五
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="decimal", kind="verbalize", deterministic=deterministic)

        graph_integer = pynutil.delete("integer_part: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        graph_fraction = (
            pynutil.delete("fractional_part: \"")
            + pynutil.insert("点")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        graph_optional_sign = pynini.closure(
            pynutil.delete("negative:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        graph_decimal_no_sign = graph_integer + pynutil.delete(" ") + graph_fraction
        graph_decimal = graph_decimal_no_sign | (graph_optional_sign + pynutil.delete(" ") + graph_decimal_no_sign)

        delete_tokens = self.delete_tokens(graph_decimal)
        self.fst = delete_tokens.optimize()
