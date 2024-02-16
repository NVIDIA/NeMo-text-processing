# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
from nemo_text_processing.inverse_text_normalization.zh.graph_utils import INPUT_LOWER_CASED, GraphFst
from nemo_text_processing.inverse_text_normalization.zh.utils import get_abs_path
from pynini.lib import pynutil


class WhiteListFst(GraphFst):
    """
    Finite state transducer for classifying whitelisted tokens
        e.g. 贵宾 -> tokens { name: "VIP" }
             美国研究生入学考试 -> { name: "GRE" }
             人力资源 -> { name: "HR" }
             工商管理学硕士 -> { name: "MBA" }
    This class has highest priority among all classifier grammars. Whitelisted tokens are defined and loaded from "data/whitelist.tsv".
    """

    def __init__(self, input_case: str = INPUT_LOWER_CASED, input_file: str = None):
        super().__init__(name="whitelist", kind="classify")

        whitelist = pynini.string_file(get_abs_path("data/whitelist.tsv"))
        graph = (pynutil.insert('name: "')) + (whitelist) + pynutil.insert('"')

        self.fst = graph.optimize()
