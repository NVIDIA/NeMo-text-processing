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

from nemo_text_processing.inverse_text_normalization.bantu.taggers.number import DIGIT, integer_domain
from nemo_text_processing.text_normalization.en.graph_utils import GraphFst, delete_space


def format_integer(max_value: int, thousands_separator: str) -> "pynini.FstLike":
    """Groups digits in threes, e.g. 25000 -> 25,000. Rejects values outside 1..max_value."""
    first = pynini.closure(DIGIT, 1, 3)
    triple = pynini.closure(DIGIT, 3, 3)
    grouped = first + pynini.closure(pynutil.insert(thousands_separator) + triple)
    return (integer_domain(max_value) @ grouped).optimize()


class CardinalFst(GraphFst):
    """
    Finite state transducer for verbalizing a resolved cardinal, e.g.
        cardinal { integer: "25000" } -> 25,000

    Args:
        max_value: largest value the language profile allows
        thousands_separator: written separator between digit groups
    """

    def __init__(self, max_value: int, thousands_separator: str):
        super().__init__(name="cardinal", kind="verbalize")
        number = format_integer(max_value, thousands_separator)
        graph = pynutil.delete("integer:") + delete_space + pynutil.delete('"') + number + pynutil.delete('"')
        self.fst = self.delete_tokens(graph).optimize()
