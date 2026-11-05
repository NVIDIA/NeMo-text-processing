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

from nemo_text_processing.inverse_text_normalization.bantu.verbalizers.verbalize import build_verbalizer
from nemo_text_processing.inverse_text_normalization.bantu.verbalizers.word import WordFst
from nemo_text_processing.text_normalization.en.graph_utils import delete_extra_space, delete_space


def build_final_verbalizer(max_value: int, thousands_separator: str) -> pynini.Fst:
    """
    Verbalizes an entire sentence. Each language's VerbalizeFinalFst calls this.

    Args:
        max_value: largest value the language allows
        thousands_separator: written separator between digit groups
    """
    types = build_verbalizer(max_value, thousands_separator) | WordFst().fst
    graph = (
        pynutil.delete("tokens")
        + delete_space
        + pynutil.delete("{")
        + delete_space
        + types
        + delete_space
        + pynutil.delete("}")
    )
    graph = delete_space + pynini.closure(graph + delete_extra_space) + graph + delete_space
    return graph.optimize()
