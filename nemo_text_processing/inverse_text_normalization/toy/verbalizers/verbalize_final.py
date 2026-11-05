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

from nemo_text_processing.inverse_text_normalization.bantu.verbalizers.verbalize_final import build_final_verbalizer
from nemo_text_processing.inverse_text_normalization.toy.settings import SKELETON
from nemo_text_processing.text_normalization.en.graph_utils import GraphFst


class VerbalizeFinalFst(GraphFst):
    """
    Finite state transducer that verbalizes an entire sentence of the test language,
    e.g. tokens { name: "kasa" } tokens { cardinal { integer: "1000" } } -> kasa 1,000
    """

    def __init__(self):
        super().__init__(name="verbalize_final", kind="verbalize")
        self.fst = build_final_verbalizer(SKELETON.max_value, SKELETON.thousands_separator)
