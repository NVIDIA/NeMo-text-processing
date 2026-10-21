# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Attribution: Jim O’Regan for Språkbanken Tal
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
from pynini.lib import byte, pynutil

from nemo_text_processing.text_normalization.en.graph_utils import delete_space
from nemo_text_processing.text_normalization.se.utils import load_labels

CASE_KEYS = ["ess", "com_pl", "com_sg", "gen_sg", "gen_pl", "ill_pl", "ill_sg", "loc_sg", "nom_pl"]
CASE_KEYS_EXT = CASE_KEYS + ["nom_sg"]
CASE_ALIASES = {
    "acc_sg": "nom_sg",
    "acc_pl": "gen_pl",
    "loc_pl": "com_sg",
}


def load_case_forms(abs_path, extended=False):
    """Loads case forms from a TSV file."""
    keys = CASE_KEYS_EXT if extended else CASE_KEYS
    return {label[0]: label[1] for label in load_labels(abs_path) if label and label[0] in keys}


_ALPHA_UPPER = "AÁBCČDĐEFGHIJKLMNŊOPQRSŠTŦUVWXYZŽÆØÅÄÖ"
_ALPHA_LOWER = "aábcčdđefghijklmnŋopqrsštŧuvwxyzžæøåäö"

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
