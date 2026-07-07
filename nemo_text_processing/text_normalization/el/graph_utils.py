# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_SIGMA, NEMO_SPACE

# Greek numerals inflect for grammatical gender. The cardinal/ordinal grammars are
# built in the NEUTER citation form (the form used when counting, e.g. "τρία", "τέσσερα",
# "διακόσια"), which is the correct default for a standalone number. The helpers below
# rewrite a rendered neuter number string into its masculine or feminine equivalent so
# that consumers (measure, money, date, thousands multiplier) can enforce agreement.
#
# Only the units 1/3/4, the teens 13/14 and the hundreds series 200-900 have distinct
# gendered forms; 2 (δύο) and everything else are invariable. The rewrites are applied at
# word boundaries (start/end of string, spaces, or a closing quote) so that, e.g.,
# "τρία" -> "τρεις" fires as a full word but the substring inside "δεκατρία" does not.

# Contexts marking a full-word boundary within a number string.
_LEFT = pynini.union("[BOS]", NEMO_SPACE)
_RIGHT = pynini.union("[EOS]", NEMO_SPACE, '"')

# Neuter -> feminine (μία, τρεις, τέσσερις, ...αντες, διακόσιες, ...).
_NEUTER_TO_FEM = pynini.string_map(
    [
        ("ένα", "μία"),
        ("τρία", "τρεις"),
        ("τέσσερα", "τέσσερις"),
        ("δεκατρία", "δεκατρείς"),
        ("δεκατέσσερα", "δεκατέσσερις"),
        ("διακόσια", "διακόσιες"),
        ("τριακόσια", "τριακόσιες"),
        ("τετρακόσια", "τετρακόσιες"),
        ("πεντακόσια", "πεντακόσιες"),
        ("εξακόσια", "εξακόσιες"),
        ("εφτακόσια", "εφτακόσιες"),
        ("οχτακόσια", "οχτακόσιες"),
        ("εννιακόσια", "εννιακόσιες"),
    ]
)

# Neuter -> masculine (ένας, τρεις, τέσσερις, διακόσιοι, ...).
_NEUTER_TO_MASC = pynini.string_map(
    [
        ("ένα", "ένας"),
        ("τρία", "τρεις"),
        ("τέσσερα", "τέσσερις"),
        ("δεκατρία", "δεκατρείς"),
        ("δεκατέσσερα", "δεκατέσσερις"),
        ("διακόσια", "διακόσιοι"),
        ("τριακόσια", "τριακόσιοι"),
        ("τετρακόσια", "τετρακόσιοι"),
        ("πεντακόσια", "πεντακόσιοι"),
        ("εξακόσια", "εξακόσιοι"),
        ("εφτακόσια", "εφτακόσιοι"),
        ("οχτακόσια", "οχτακόσιοι"),
        ("εννιακόσια", "εννιακόσιοι"),
    ]
)

_FEM_REWRITE = pynini.cdrewrite(_NEUTER_TO_FEM, _LEFT, _RIGHT, NEMO_SIGMA).optimize()
_MASC_REWRITE = pynini.cdrewrite(_NEUTER_TO_MASC, _LEFT, _RIGHT, NEMO_SIGMA).optimize()


def shift_cardinal_gender_fem(fst: "pynini.FstLike") -> "pynini.FstLike":
    """Rewrites the (neuter) output of ``fst`` into its feminine form, e.g.
    "τρία" -> "τρεις", "διακόσια είκοσι τρία" -> "διακόσιες είκοσι τρεις"."""
    return fst @ _FEM_REWRITE


def shift_cardinal_gender_masc(fst: "pynini.FstLike") -> "pynini.FstLike":
    """Rewrites the (neuter) output of ``fst`` into its masculine form, e.g.
    "τρία" -> "τρεις", "διακόσια" -> "διακόσιοι", "ένα" -> "ένας"."""
    return fst @ _MASC_REWRITE
