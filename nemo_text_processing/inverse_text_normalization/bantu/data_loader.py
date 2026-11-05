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

"""Fills a language's profile skeleton with the word lists in its data folder.

Each language writes the shape of its grammar in settings.py: a CardinalProfile whose
word lists are empty (noun classes, places, counted scales with their head words and
switches). Its word lists live in tab-separated files under data/; every file is
optional, and a missing file means an empty list:

    numbers/units.tsv              form, value, class, where (any, atomic, final, teen)
    numbers/places.tsv             form, coefficient, power
    numbers/exact.tsv              form, value
    numbers/roles.tsv              form, value, role
    numbers/scale_words.tsv        scale plural, part (tens, ones, hundreds), form, value
    numbers/held_back.tsv          word
    numbers/spelling_variants.tsv  word, variant
    money/currency.tsv             phrase, code, class, roles (comma-separated, may be empty)
    money/cues.tsv                 word, kind (stem, word, not)

A "where" of any puts a unit form both inside compounds (atomic) and at the end of a
number (final); teen forms are whole teen words such as a contracted 15.
"""
import csv
import os
from collections import OrderedDict
from dataclasses import replace
from typing import Dict, List, Tuple

from nemo_text_processing.inverse_text_normalization.bantu.profile import (
    CardinalProfile,
    CurrencyContext,
    MoneyCues,
    with_spelling_variants,
)

WHERE = {
    "any": ("atomic_units", "final_units"),
    "atomic": ("atomic_units",),
    "final": ("final_units",),
    "teen": ("teen_variants",),
}
SCALE_PARTS = {"tens": "tens", "ones": "count_ones", "hundreds": "count_hundreds"}
CUE_KINDS = {"stem": "stems", "word": "words", "not": "not_cues"}


def read_rows(data_dir: str, name: str, columns: int, last_optional: bool = False) -> List[Tuple[str, List[str]]]:
    """
    Non-blank rows of one TSV as (location, row); location is "path:line" for error messages.
    Every cell must be non-empty, except the last one when last_optional is set.
    """
    path = os.path.join(data_dir, name)
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8", newline="") as tsv:
        for line, row in enumerate(csv.reader(tsv, delimiter="\t"), start=1):
            if not row:
                continue
            if len(row) != columns:
                raise ValueError(f"{path}:{line}: expected {columns} columns, found {len(row)}")
            required = row[:-1] if last_optional else row
            if not all(required):
                raise ValueError(f"{path}:{line}: empty field")
            rows.append((f"{path}:{line}", row))
    return rows


def lookup(table: Dict[str, str], key: str, location: str, what: str) -> str:
    if key not in table:
        raise ValueError(f"{location}: unknown {what} {key!r}")
    return table[key]


def load_profile(data_dir: str, skeleton: CardinalProfile) -> CardinalProfile:
    """
    Returns the skeleton with its word lists filled from data_dir.

    Args:
        data_dir: the language's data folder
        skeleton: the language's profile with empty word lists (from its settings.py)
    """
    if not os.path.isdir(data_dir):
        raise ValueError(f"no data folder at {data_dir}")
    contexts = {
        context.class_id: {"atomic_units": [], "final_units": [], "teen_variants": []} for context in skeleton.contexts
    }
    for location, (form, value, class_id, where) in read_rows(data_dir, "numbers/units.tsv", 4):
        lists = contexts.get(class_id)
        if lists is None:
            raise ValueError(f"{location}: unknown class {class_id!r}")
        for field in lookup(WHERE, where, location, "where"):
            lists[field].append((form, value))

    places = {str(place.power): [] for place in skeleton.places}
    for location, (form, value, power) in read_rows(data_dir, "numbers/places.tsv", 3):
        if power not in places:
            raise ValueError(f"{location}: unknown power {power!r}")
        places[power].append((form, value))

    scales = {scale.plural: {field: [] for field in SCALE_PARTS.values()} for scale in skeleton.counted_scales}
    for location, (plural, part, form, value) in read_rows(data_dir, "numbers/scale_words.tsv", 4):
        if plural not in scales:
            raise ValueError(f"{location}: unknown scale {plural!r}")
        scales[plural][lookup(SCALE_PARTS, part, location, "part")].append((form, value))

    roles = OrderedDict()
    for _, (form, value, role) in read_rows(data_dir, "numbers/roles.tsv", 3):
        roles.setdefault(role, []).append((form, value))

    # An amount after a currency agrees with a class whose numbers compound.
    compounding = {context.class_id for context in skeleton.contexts if context.compounds}
    currencies = []
    rows = read_rows(data_dir, "money/currency.tsv", 4, last_optional=True)
    for location, (phrase, code, class_id, currency_roles) in rows:
        if class_id not in compounding:
            raise ValueError(f"{location}: unknown or non-compounding class {class_id!r}")
        names = tuple(name for name in currency_roles.split(",") if name)
        for name in names:
            if name not in roles:
                raise ValueError(f"{location}: unknown role {name!r}")
        currencies.append(CurrencyContext(phrase, code, class_id, names))

    cues = {field: [] for field in CUE_KINDS.values()}
    for location, (word, kind) in read_rows(data_dir, "money/cues.tsv", 2):
        cues[lookup(CUE_KINDS, kind, location, "kind")].append(word)

    variants = OrderedDict()
    for _, (word, variant) in read_rows(data_dir, "numbers/spelling_variants.tsv", 2):
        variants.setdefault(word, []).append(variant)

    profile = replace(
        skeleton,
        contexts=tuple(
            replace(context, **{field: tuple(forms) for field, forms in contexts[context.class_id].items()})
            for context in skeleton.contexts
        ),
        places=tuple(replace(place, forms=tuple(places[str(place.power)])) for place in skeleton.places),
        counted_scales=tuple(
            replace(scale, **{field: tuple(forms) for field, forms in scales[scale.plural].items()})
            for scale in skeleton.counted_scales
        ),
        exact_boundaries=tuple(tuple(row) for _, row in read_rows(data_dir, "numbers/exact.tsv", 2)),
        roles=tuple((role, tuple(forms)) for role, forms in roles.items()),
        currencies=tuple(currencies),
        money_cues=MoneyCues(**{field: tuple(words) for field, words in cues.items()}),
        held_back_words=tuple(row[0] for _, row in read_rows(data_dir, "numbers/held_back.tsv", 1)),
    )
    if variants:
        profile = with_spelling_variants(profile, {word: tuple(v) for word, v in variants.items()})
    return profile
