# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.hi.graph_utils import NEMO_SIGMA, GraphFst


class PostProcessor(GraphFst):
    """
    Post-processing applied to the fully verbalized sentence, following the
    post-processor pattern used by other languages (e.g. the ja ITN grammar).

    Currently it removes the stray space the tokenizer inserts before sentence /
    clause punctuation, so an attached mark stays attached:
        e.g. "मेरा पेन नंबर है TBXQF4138W ." -> "मेरा पेन नंबर है TBXQF4138W."
    """

    def __init__(self):
        super().__init__(name="post_process", kind="verbalize")
        sentence_punct = pynini.union(".", "।", ",", "!", "?", ";", ":")
        remove_space_before_punct = pynini.cdrewrite(pynutil.delete(" "), "", sentence_punct, NEMO_SIGMA)
        self.fst = remove_space_before_punct.optimize()
