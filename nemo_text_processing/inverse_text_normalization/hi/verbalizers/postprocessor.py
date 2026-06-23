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

from nemo_text_processing.inverse_text_normalization.hi.graph_utils import (
    MIN_NEG_WEIGHT,
    NEMO_CHAR,
    NEMO_PUNCT,
    NEMO_SIGMA,
    GraphFst,
)


class PostProcessor(GraphFst):
    """
    Post-processing applied to the fully verbalized sentence, following the
    PostProcessingFst pattern used by other grammars (en TN, ja ITN): it removes
    the space the tokenizer leaves before a punctuation mark and after an opening
    bracket, e.g. "TBXQF4138W ." -> "TBXQF4138W." and "( AVIC )" -> "(AVIC)".
    """

    def __init__(self):
        super().__init__(name="post_process", kind="verbalize")

        punct = NEMO_PUNCT | pynini.union("।", "॥")
        allow_space_before = pynini.union("(", "{", "<", pynini.escape("["), "-", "&", '"', "'", "`")
        no_space_before = pynini.difference(punct, allow_space_before).optimize()
        brackets = pynini.union("(", "{", "<", pynini.escape("["))
        delete_space = pynutil.delete(" ")

        non_punct = pynini.difference(NEMO_CHAR, no_space_before).optimize()
        graph = pynini.closure(
            pynini.closure(non_punct)
            + pynini.closure(no_space_before | pynutil.add_weight(delete_space + no_space_before, MIN_NEG_WEIGHT))
            + pynini.closure(non_punct)
        ).optimize()

        no_space_after = pynini.cdrewrite(delete_space, brackets, NEMO_SIGMA, NEMO_SIGMA).optimize()
        self.fst = pynini.compose(graph, no_space_after).optimize()
