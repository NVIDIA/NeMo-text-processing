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

<<<<<<< HEAD
<<<<<<< HEAD
from nemo_text_processing.text_normalization.ja.graph_utils import (
=======
from nemo_text_processing.text_normalization.zh.graph_utils import (
>>>>>>> 03c25c6 (updates for space removal)
=======
from nemo_text_processing.text_normalization.ja.graph_utils import (
>>>>>>> 88165c66 (updates)
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_extra_space,
    delete_space,
)


class WordFst(GraphFst):
    '''
        tokens { char: "文字" } -> 文字
    '''

<<<<<<< HEAD
<<<<<<< HEAD
    def __init__(self, deterministic: bool = True):
=======
    def __init__(self, deterministic: bool = True, lm: bool = False):
>>>>>>> 03c25c6 (updates for space removal)
=======
    def __init__(self, deterministic: bool = True):
>>>>>>> 88165c66 (updates)
        super().__init__(name="char", kind="verbalize", deterministic=deterministic)

        graph = pynutil.delete("name: \"") + NEMO_NOT_QUOTE + pynutil.delete("\"")

        self.fst = graph.optimize()
