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
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, NEMO_SIGMA, GraphFst, delete_space
from pynini.lib import pynutil


class OrdinalFst(GraphFst):
    """
    Finite state transducer for verbalizing ordinal, e.g.
       ordinal { integer: "13" } -> 13th
    """

    def __init__(self):
        super().__init__(name="ordinal", kind="verbalize")
        graph = (
            pynutil.delete("integer:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        convert_eleven = pynini.cross("11", "11th")
        convert_twelve = pynini.cross("12", "12th")
        convert_thirteen = pynini.cross("13", "13th")
        convert_one = pynini.cross("1", "1st")
        convert_two = pynini.cross("2", "2nd")
        convert_three = pynini.cross("3", "3rd")
        convert_rest = pynutil.insert("th", weight=0.01)

        suffix = pynini.cdrewrite(
            convert_eleven
            | convert_twelve
            | convert_thirteen
            | convert_one
            | convert_two
            | convert_three
            | convert_rest,
            "",
            "[EOS]",
            NEMO_SIGMA,
        )
        graph = graph @ suffix
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
