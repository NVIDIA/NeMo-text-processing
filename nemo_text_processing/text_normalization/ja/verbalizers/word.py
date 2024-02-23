<<<<<<< HEAD
# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
=======
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
>>>>>>> 26af208 (verbalizers for numebrs)
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

<<<<<<< HEAD

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.ja.graph_utils import (
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_extra_space,
    delete_space,
)


class WordFst(GraphFst):
    '''
        tokens { char: "文字" } -> 文字
    '''

    def __init__(self, deterministic: bool = True):
        super().__init__(name="char", kind="verbalize", deterministic=deterministic)

        graph = pynutil.delete("name: \"") + NEMO_NOT_QUOTE + pynutil.delete("\"")
=======
import pynini
from nemo_text_processing.text_normalization.jp.graph_utils import NEMO_CHAR, NEMO_SIGMA, GraphFst, delete_space
from pynini.lib import pynutil


class WordFst(GraphFst):
    """
    Finite state transducer for verbalizing plain tokens
        e.g. tokens { name: "sleep" } -> sleep
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="word", kind="verbalize", deterministic=deterministic)
        chars = pynini.closure(NEMO_CHAR - " ", 1)
        char = pynutil.delete("name:") + delete_space + pynutil.delete("\"") + chars + pynutil.delete("\"")
        graph = char @ pynini.cdrewrite(pynini.cross(u"\u00A0", " "), "", "", NEMO_SIGMA)
>>>>>>> 26af208 (verbalizers for numebrs)

        self.fst = graph.optimize()
