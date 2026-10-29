# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright (c) 2026, Crane Labs. All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.bantu.verbalizers.cardinal import format_integer
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_UPPER, GraphFst, delete_space, insert_space


class MoneyFst(GraphFst):
    """
    Finite state transducer for verbalizing a resolved amount as "<amount> <ISO code>", e.g.
        money { integer_part: "20000" currency: "UGX" } -> 20,000 UGX

    Args:
        max_value: largest value the language profile allows
        thousands_separator: written separator between digit groups
    """

    def __init__(self, max_value: int, thousands_separator: str):
        super().__init__(name="money", kind="verbalize")
        number = format_integer(max_value, thousands_separator)
        code = pynini.closure(NEMO_UPPER, 3, 3)
        graph = (
            pynutil.delete("integer_part:")
            + delete_space
            + pynutil.delete('"')
            + number
            + pynutil.delete('"')
            + delete_space
            + pynutil.delete("currency:")
            + delete_space
            + pynutil.delete('"')
            + insert_space
            + code
            + pynutil.delete('"')
        )
        self.fst = self.delete_tokens(graph).optimize()
