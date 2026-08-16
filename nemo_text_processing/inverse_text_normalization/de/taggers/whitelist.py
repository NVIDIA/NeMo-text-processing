# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.en.graph_utils import GraphFst, convert_space


class WhiteListFst(GraphFst):
    """
    Finite state transducer for classifying whitelisted tokens
        e.g. misses -> tokens { name: "Mrs." }
    Args:
        tn_whitelist_tagger: TN whitelist tagger
        input_file: path to a file with whitelist replacements (each line of the file: written_form\tspoken_form\n),
        e.g. nemo_text_processing/inverse_text_normalization/en/data/whitelist.tsv
    """

    def __init__(self, tn_whitelist_tagger: GraphFst, deterministic: bool = True, input_file: str = None):
        super().__init__(name="whitelist", kind="classify", deterministic=deterministic)

        if input_file:
            whitelist = pynini.string_file(input_file).invert()
        else:
            whitelist = pynini.invert(tn_whitelist_tagger.graph)

        graph = pynutil.insert("name: \"") + convert_space(whitelist) + pynutil.insert("\"")
        self.fst = graph.optimize()
