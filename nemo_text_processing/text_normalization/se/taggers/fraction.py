# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Jim O'Regan for Språkbanken Tal
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

from nemo_text_processing.text_normalization.en.graph_utils import GraphFst, convert_space
from nemo_text_processing.text_normalization.se.utils import get_abs_path


class FractionFst(GraphFst):
    """Classifies fraction expressions documented by Sámediggi."""

    def __init__(self, cardinal=None, ordinal=None, deterministic: bool = True):
        super().__init__(name="name", kind="classify", deterministic=deterministic)

        graph = pynini.string_file(get_abs_path("data/numbers/fraction.tsv"))
        if not deterministic:
            graph |= pynini.string_file(get_abs_path("data/numbers/fraction_nd.tsv"))
        self.fst = pynutil.insert('name: "') + convert_space(graph) + pynutil.insert('"')
