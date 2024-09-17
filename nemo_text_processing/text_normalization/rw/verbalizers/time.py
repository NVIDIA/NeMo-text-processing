# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright (c) 2024, DIGITAL UMUGANDA
# Copyright 2015 and onwards Google, Inc.
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

from nemo_text_processing.text_normalization.rw.graph_utils import NEMO_CHAR, GraphFst, delete_space


class VerbalizeTimeFst(GraphFst):
    def __init__(self):
        super().__init__(name="time", kind="verbalize")
        hour = (
            pynutil.delete("hours:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_CHAR)
            + pynutil.delete("\"")
            + delete_space
            + pynutil.delete("minutes:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_CHAR)
            + pynutil.delete("\"")
        )

        graph = hour
        delete_tokens = self.delete_tokens(graph)

        self.fst = delete_tokens.optimize()
