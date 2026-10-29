# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright (c) 2026, Crane Labs. All rights reserved.
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

from pynini.lib import rewrite

from nemo_text_processing.inverse_text_normalization.bantu.taggers.tokenize_and_classify import ClassifyFst
from nemo_text_processing.inverse_text_normalization.bantu.verbalizers.verbalize_final import VerbalizeFinalFst
from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer
from nemo_text_processing.text_normalization.token_parser import TokenParser


def build_normalizer(profile) -> InverseNormalizer:
    """
    NeMo's own inverse-normalization runtime with the kernel's tagger and verbalizer
    built directly from a profile. The kernel is not a selectable language, so it is
    not constructed through InverseNormalizer(lang=...); the attributes set here
    mirror the end of InverseNormalizer.__init__.
    """
    normalizer = InverseNormalizer.__new__(InverseNormalizer)
    normalizer.lang = profile.language
    normalizer.tagger = ClassifyFst(profile)
    normalizer.verbalizer = VerbalizeFinalFst(profile)
    normalizer.parser = TokenParser()
    normalizer.max_number_of_permutations_per_split = 729  # InverseNormalizer default
    normalizer.post_processor = None
    return normalizer


def best_tags(normalizer: InverseNormalizer, text: str) -> set:
    """All equally best tagger outputs. The kernel must give exactly one for any input."""
    return set(rewrite.optimal_rewrites(text, normalizer.tagger.fst))
