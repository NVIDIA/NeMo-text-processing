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
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, delete_preserve_order, delete_space


class OrdinalFst(GraphFst):
    """
    Finite state transducer for verbalizing ordinals, e.g.
        ordinal { integer: "5" morphosyntactic_features: "வது" preserve_order: true } -> 5வது
        ordinal { integer: "10" morphosyntactic_features: "ஆம்" preserve_order: true } -> 10ஆம்
    """

    def __init__(self):
        super().__init__(name="ordinal", kind="verbalize")

        integer = pynutil.delete("integer: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        marker = (
            pynutil.delete("morphosyntactic_features: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        )
        self.graph = integer + delete_space + marker + delete_preserve_order
        self.fst = self.delete_tokens(self.graph).optimize()
