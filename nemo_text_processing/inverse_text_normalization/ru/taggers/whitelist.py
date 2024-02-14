# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from nemo_text_processing.text_normalization.ru.utils import get_abs_path


class WhiteListFst(GraphFst):
    """
    Finite state transducer for classifying whitelist, e.g.
        "квартира" -> telephone { number_part: "кв." }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
        input_file: path to a file with whitelist replacements (each line of the file: written_form\tspoken_form\n),
            e.g. nemo_text_processing/inverse_text_normalization/en/data/whitelist.tsv
    """

    def __init__(self, deterministic: bool = True, input_file: str = None):
        super().__init__(name="whitelist", kind="classify", deterministic=deterministic)

        if input_file:
            whitelist = pynini.string_file(input_file).invert()
        else:
            whitelist = pynini.string_file(get_abs_path("data/whitelist.tsv")).invert()
        graph = pynutil.insert("name: \"") + convert_space(whitelist) + pynutil.insert("\"")
        self.fst = graph.optimize()
