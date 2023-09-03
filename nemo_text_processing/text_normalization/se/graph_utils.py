# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2023, Jim O'Regan for Språkbanken Tal
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
from nemo_text_processing.text_normalization.en.graph_utils import delete_space
from pynini.lib import byte, pynutil

_ALPHA_UPPER = "АÁBCČDĐEFGHIJKLMNŊOPRSŠTŦUVZŽÆØÅÄÖ"
_ALPHA_LOWER = "аábcčdđefghijklmnŋoprsštŧuvzžæøåäö"

TO_LOWER = pynini.union(*[pynini.cross(x, y) for x, y in zip(_ALPHA_UPPER, _ALPHA_LOWER)])
TO_UPPER = pynini.invert(TO_LOWER)

SE_LOWER = pynini.union(*_ALPHA_LOWER).optimize()
SE_UPPER = pynini.union(*_ALPHA_UPPER).optimize()
SE_ALPHA = pynini.union(SE_LOWER, SE_UPPER).optimize()
SE_ALNUM = pynini.union(byte.DIGIT, SE_ALPHA).optimize()

bos_or_space = pynini.union("[BOS]", " ")
eos_or_space = pynini.union("[EOS]", " ")

ensure_space = pynini.cross(pynini.closure(delete_space, 0, 1), " ")


def make_spacer(deterministic=True):
    spacer = pynini.accep("")
    if not deterministic:
        spacer |= pynutil.insert(" ")
    return spacer
