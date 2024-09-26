<<<<<<< HEAD
<<<<<<<< HEAD:nemo_text_processing/inverse_text_normalization/ja/verbalizers/whitelist.py
=======
<<<<<<<< HEAD:nemo_text_processing/text_normalization/fr/verbalizers/__init__.py
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
========
>>>>>>> 59f46198ab4c8880c6a5fb88f3cbee9530156498
# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, GraphFst


class WhiteListFst(GraphFst):
    '''
<<<<<<< HEAD
<<<<<<< HEAD
    tokens { whitelist: "ATM" } -> A T M
=======
        tokens { whitelist: "ATM" } -> A T M
>>>>>>> 0a4a21c (Jp itn 20240221 (#141))
=======
    tokens { whitelist: "ATM" } -> A T M
>>>>>>> 59f46198ab4c8880c6a5fb88f3cbee9530156498
    '''

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="whitelist", kind="verbalize", deterministic=deterministic)

        whitelist = pynutil.delete("name: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        graph = whitelist
        self.fst = graph.optimize()
<<<<<<< HEAD
========
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


from pynini.lib import pynutil

from nemo_text_processing.text_normalization.zh.graph_utils import NEMO_NOT_QUOTE, GraphFst


class WordFst(GraphFst):
    '''
    tokens { char: "你" } -> 你
    '''

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="char", kind="verbalize", deterministic=deterministic)

        graph = pynutil.delete("name: \"") + NEMO_NOT_QUOTE + pynutil.delete("\"")
        self.fst = graph.optimize()
>>>>>>>> 59f46198ab4c8880c6a5fb88f3cbee9530156498:nemo_text_processing/text_normalization/zh/verbalizers/word.py
=======
>>>>>>>> 59f46198ab4c8880c6a5fb88f3cbee9530156498:nemo_text_processing/inverse_text_normalization/ja/verbalizers/whitelist.py
>>>>>>> 59f46198ab4c8880c6a5fb88f3cbee9530156498
