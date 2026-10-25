# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""The settings a language supplies to the shared Bantu kernel: its number words, agreement and scales."""
from dataclasses import dataclass, replace
from typing import Dict, FrozenSet, List, Tuple

Forms = Tuple[Tuple[str, str], ...]


@dataclass(frozen=True)
class NounContext:
    class_id: str
    noun: str
    atomic_units: Forms
    final_units: Forms
    compounds: bool = True
    teen_variants: Forms = ()


@dataclass(frozen=True)
class Place:
    # Each form outputs its nonzero coefficient, not the complete place value.
    power: int
    forms: Forms
    remainder_join: str


@dataclass(frozen=True)
class CurrencyContext:
    # A spoken currency phrase sets the amount's agreement, like a counted noun.
    phrase: str
    code: str
    agreement_class: str
    roles: Tuple[str, ...] = ()  # extra readings licensed only after this phrase


@dataclass(frozen=True)
class CountedScale:
    """A scale noun followed by its count, e.g. Luganda mitwalo abiri = 20 x 10,000."""

    singular: str  # head counted once, e.g. mutwalo
    singular_count: str  # its agreeing "one", e.g. gumu
    plural: str  # e.g. mitwalo
    power: int  # 4 for ten-thousands
    count_class: str  # agreement class of unit words inside the count
    tens: Forms  # tens words inside a count, coefficient 1-9
    remainder_class: str  # agreement class of the amount after the scale
    contracted_teen_joins: Tuple[str, ...] = ()  # teen join before a vowel, e.g. n'
    implied_currency: str = ""  # payment policy: a bare amount means this currency
    count_ones: Forms = ()  # a "one" ending a count, e.g. kkumi na kamu (11)
    count_hundreds: Forms = ()  # hundreds inside a count, e.g. obukadde kikumi (100)
    # Heads with the augment (Luganda e-/a-/o-) count the noun before them; the bare
    # heads above are money.
    counting_plural: str = ""  # e.g. emitwalo, obukadde
    counting_ones: Tuple[str, ...] = ()  # e.g. akakadde kamu
    money_only_ones: Tuple[str, ...] = ()  # one, as money only: akakadde (not ebitabo akakadde)
    implied_anywhere: bool = False  # bare heads are money wherever they appear
    counting_money_at_start: bool = False  # augmented heads are money at the start of input
    counts_any_word: bool = False  # word + augmented head counts that word, listed or not
    after_counted_nouns: bool = False  # augmented heads count listed nouns, with agreement


@dataclass(frozen=True)
class MoneyCues:
    """Words that signal a payment just before an amount (e.g. Luganda money verbs)."""

    stems: Tuple[str, ...] = ()  # match anywhere inside a word
    words: Tuple[str, ...] = ()  # match only the whole word
    not_cues: Tuple[str, ...] = ()  # whole words containing a stem that are not cues


@dataclass(frozen=True)
class CardinalProfile:
    language: str
    contexts: Tuple[NounContext, ...]
    ten: str
    teen_join: str
    places: Tuple[Place, ...]
    exact_boundaries: Forms
    roles: Tuple[Tuple[str, Forms], ...]
    max_value: int
    currencies: Tuple[CurrencyContext, ...] = ()
    counted_scales: Tuple[CountedScale, ...] = ()
    money_cues: MoneyCues = MoneyCues()
    # Words that continue a number but are not handled yet (e.g. "and a half").
    # They join the number's run, so the whole phrase stays unchanged.
    held_back_words: Tuple[str, ...] = ()
    # Vowels, for contracting a join before a vowel-initial word (Luganda na + ebiri -> n'ebiri).
    vowels: str = "aeiou"

    def join_words(self) -> FrozenSet[str]:
        """Words that join number parts; they may also be ordinary words (e.g. locative mu)."""
        joins = [self.teen_join] + [place.remainder_join for place in self.places]
        return frozenset(word for join in joins for word in join.split())

    def number_words(self) -> FrozenSet[str]:
        """Every word that can be part of a number, including joins and role-only words."""
        surfaces = [self.ten, self.teen_join]
        for place in self.places:
            surfaces.append(place.remainder_join)
            surfaces.extend(form for form, _ in place.forms)
        for context in self.contexts:
            surfaces.extend(form for form, _ in context.atomic_units + context.final_units + context.teen_variants)
        surfaces.extend(form for form, _ in self.exact_boundaries)
        for _, forms in self.roles:
            surfaces.extend(form for form, _ in forms)
        for scale in self.counted_scales:
            surfaces.extend(
                (scale.singular, scale.singular_count, scale.plural) + scale.counting_ones + scale.money_only_ones
            )
            if scale.counting_plural:
                surfaces.append(scale.counting_plural)
            surfaces.extend(form for form, _ in scale.tens + scale.count_ones + scale.count_hundreds)
            surfaces.extend(scale_teen_words(self, scale))
        surfaces.extend(self.held_back_words)
        return frozenset(word for form in surfaces for word in form.split())


def scale_teen_words(profile: CardinalProfile, scale: CountedScale) -> List[str]:
    """Contracted teen words such as n'ebiri / nebiri (one word each)."""
    context = agreement_context(profile, scale.count_class)
    return [
        prefix + form
        for form, _ in context.final_units + scale.count_ones
        if form[0] in profile.vowels
        for prefix in scale.contracted_teen_joins
    ]


def agreement_context(profile: CardinalProfile, class_id: str) -> NounContext:
    context = next((c for c in profile.contexts if c.class_id == class_id and c.compounds), None)
    if context is None:
        raise ValueError(f"no compound agreement context for class {class_id}")
    return context


def invariant_context(profile: CardinalProfile) -> NounContext:
    """Unit forms that are the same in every class (Luganda 6-9), for unknown nouns.

    A number after a noun the profile does not list may only end in these, so
    nothing that must agree with that noun is guessed.
    """
    contexts = [c for c in profile.contexts if c.compounds]
    shared = set(contexts[0].final_units)
    for context in contexts[1:]:
        shared &= set(context.final_units)
    units = tuple(sorted(shared, key=lambda form: form[1]))
    return NounContext("invariant", "", units + ((profile.ten, "10"),), units)


def with_spelling_variants(profile: CardinalProfile, variants: Dict[str, Tuple[str, ...]]) -> CardinalProfile:
    """Accept extra spellings of whole words (e.g. abiiri for abiri) in every form."""

    def spell(forms: Forms) -> Forms:
        extra = []
        for form, value in forms:
            words = form.split()
            for index, word in enumerate(words):
                for variant in variants.get(word, ()):
                    extra.append((" ".join(words[:index] + [variant] + words[index + 1 :]), value))
        return forms + tuple(extra)

    return replace(
        profile,
        contexts=tuple(
            replace(
                c,
                atomic_units=spell(c.atomic_units),
                final_units=spell(c.final_units),
                teen_variants=spell(c.teen_variants),
            )
            for c in profile.contexts
        ),
        places=tuple(replace(p, forms=spell(p.forms)) for p in profile.places),
        roles=tuple((name, spell(forms)) for name, forms in profile.roles),
        counted_scales=tuple(
            replace(s, tens=spell(s.tens), count_hundreds=spell(s.count_hundreds)) for s in profile.counted_scales
        ),
    )
