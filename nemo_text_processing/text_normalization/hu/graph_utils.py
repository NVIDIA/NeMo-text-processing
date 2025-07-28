# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
# Copyright (c) 2023, Jim O'Regan
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
from pynini.lib import byte

from nemo_text_processing.text_normalization.en.graph_utils import delete_space, insert_space

_ALPHA_UPPER = "AÁBCDEÉFGHIÍJKLMNOÓÖŐPQRSTUÚÜŰVWXYZ"
_ALPHA_LOWER = "aábcdeéfghiíjklmnoóöőpqrstuúüűvwxyz"
_VOWELS = "AÁEÉIÍOÓÖŐUÚÜŰaáeéiíoóöőuúüű"

TO_LOWER = pynini.union(*[pynini.cross(x, y) for x, y in zip(_ALPHA_UPPER, _ALPHA_LOWER)])
TO_UPPER = pynini.invert(TO_LOWER)

HU_LOWER = pynini.union(*_ALPHA_LOWER).optimize()
HU_UPPER = pynini.union(*_ALPHA_UPPER).optimize()
HU_ALPHA = pynini.union(HU_LOWER, HU_UPPER).optimize()
HU_ALNUM = pynini.union(byte.DIGIT, HU_ALPHA).optimize()
HU_VOWELS = pynini.union(*[x for x in _VOWELS])

ensure_space = pynini.closure(delete_space, 0, 1) + insert_space

bos_or_space = pynini.union("[BOS]", " ")
eos_or_space = pynini.union("[EOS]", " ")
