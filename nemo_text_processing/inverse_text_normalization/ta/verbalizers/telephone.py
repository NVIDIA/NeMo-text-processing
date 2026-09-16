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

from nemo_text_processing.inverse_text_normalization.ta.graph_utils import GraphFst
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, delete_space, insert_space


class TelephoneFst(GraphFst):
    """
    Finite state transducer for verbalizing telephone numbers, e.g.
        telephone { number_part: "9943206870" } -> 9943206870
        telephone { country_code: "+91" number_part: "9876543210" } -> +91 9876543210
        telephone { country_code: "+91" } -> +91
    """

    def __init__(self):
        super().__init__(name="telephone", kind="verbalize")

        country_code = pynutil.delete("country_code: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        number_part = pynutil.delete("number_part: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        self.graph = (pynini.closure(country_code + delete_space + insert_space, 0, 1) + number_part) | country_code
        self.fst = self.delete_tokens(self.graph).optimize()
