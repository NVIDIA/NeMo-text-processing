# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.ja.graph_utils import (
    NEMO_SIGMA,
    GraphFst,
    TO_LOWER,
    TO_UPPER,
)
from nemo_text_processing.text_normalization.ja.taggers.punctuation import PunctuationFst


class PostProcessor(GraphFst):
    """
    Optional postprocessing for Japanese TN.

    The default graph is an identity rewrite. Optional punctuation removal and
    ASCII case conversion are kept generic; OOV tagging needs a Japanese-specific
    character inventory and is intentionally not implemented here.
    """

    def __init__(
        self,
        remove_puncts: bool = False,
        to_upper: bool = False,
        to_lower: bool = False,
        tag_oov: bool = False,
    ):
        super().__init__(name="PostProcessor", kind="processor")

        if to_upper and to_lower:
            raise ValueError("to_upper and to_lower cannot both be enabled.")
        if tag_oov:
            raise ValueError("tag_oov is not supported for Japanese TN without a Japanese charset inventory.")

        graph = pynini.cdrewrite("", "", "", NEMO_SIGMA)
        if remove_puncts:
            remove_puncts_graph = pynutil.delete(pynini.union(*PunctuationFst().punct_marks))
            graph @= pynini.cdrewrite(remove_puncts_graph, "", "", NEMO_SIGMA).optimize()

        if to_upper:
            graph @= pynini.cdrewrite(TO_UPPER, "", "", NEMO_SIGMA).optimize()
        elif to_lower:
            conv_cases_graph = TO_LOWER
            graph @= pynini.cdrewrite(conv_cases_graph, "", "", NEMO_SIGMA).optimize()

        self.fst = graph.optimize()
