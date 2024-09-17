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

from nemo_text_processing.text_normalization.rw.graph_utils import GraphFst
from nemo_text_processing.text_normalization.rw.utils import get_abs_path


class TimeFst(GraphFst):
    def __init__(self):
        super().__init__(name="time", kind="classify")

        hours = pynini.string_file(get_abs_path("data/time/hours.tsv"))

        minutes = pynini.string_file(get_abs_path("data/time/minutes.tsv"))

        final_graph = (
            pynutil.insert("hours:\"")
            + hours
            + pynutil.insert("\"")
            + pynutil.delete(":")
            + pynutil.insert(" minutes:\"")
            + minutes
            + pynutil.insert("\"")
        )
        final_graph = self.add_tokens(final_graph)

        self.fst = final_graph.optimize()
