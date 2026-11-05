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

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.bantu.graph_utils import (
    DIGIT,
    NONZERO,
    NONZERO_DIGITS,
    integer_domain,
    mapping,
    money_cue_word,
    run,
    token,
    word_token,
)
from nemo_text_processing.inverse_text_normalization.bantu.profile import (
    CardinalProfile,
    NounContext,
    invariant_context,
)
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_SPACE, GraphFst


def build_cardinal(profile: CardinalProfile, context: NounContext) -> pynini.Fst:
    """Compose fixed-width places, supply skipped zeros, then canonicalize once."""
    atomic = mapping(context.atomic_units)
    if not context.compounds:
        return (atomic @ integer_domain(profile.max_value)).optimize()
    units = mapping(context.final_units)
    lower = units
    expected_power = 1
    for place in profile.places:
        if place.power != expected_power:
            raise ValueError("places must be consecutive ascending decimal powers")
        if any(value not in NONZERO_DIGITS or len(value) != 1 for _, value in place.forms):
            raise ValueError("place coefficients must be single nonzero digits")
        # A present high place either consumes a smaller remainder or emits zeros.
        high = mapping(place.forms) + (
            pynutil.delete(place.remainder_join) + lower | pynutil.insert("0" * place.power)
        )
        if place.power == 1:
            high |= pynini.cross(profile.ten, "10")
            high |= pynini.cross(profile.ten, "1") + pynutil.delete(profile.teen_join) + units
            high |= mapping(context.teen_variants)
        lower = (pynutil.insert("0") + lower | high).optimize()
        expected_power += 1
    canonical = pynutil.delete(pynini.closure("0")) + NONZERO + pynini.closure(DIGIT)
    compounds = lower @ canonical
    # A final-one form need not be licensed as a standalone plural count.
    compounds @= NONZERO + pynini.closure(DIGIT, 1)
    result = atomic | compounds | mapping(profile.exact_boundaries)
    return (result @ integer_domain(profile.max_value)).optimize()


def build_role(profile: CardinalProfile, role: str) -> pynini.Fst:
    """Explicit role probe; not a standalone sentence or currency grammar."""
    for name, forms in profile.roles:
        if name == role:
            return mapping(forms)
    raise KeyError(role)


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying a counted number: a noun followed by a
    run of number words that agree with it. The noun stays a word; the number is
    fully resolved here, e.g.
        bakopa pideka vo baima -> tokens { name: "bakopa" } tokens { cardinal { integer: "21" } }

    Any other word followed by a counting scale (Luganda embuzi emitwalo ebiri,
    20,000 goats) is also a counted number. Here the count may only end in forms
    that are the same in every class, so no agreement with an unlisted noun is guessed.

    The graphs other classes reuse are kept as attributes:
        by_class: the agreeing number graph for each noun class that compounds
        roles: each role's graph (e.g. a currency-only reading), bounded by the profile maximum

    Args:
        profile: the language profile
        ordinary_word: any word that is not a number word
    """

    def __init__(self, profile: CardinalProfile, ordinary_word: pynini.Fst):
        super().__init__(name="cardinal", kind="classify")
        # Imported here: the scale builders use build_cardinal from this module.
        from nemo_text_processing.inverse_text_normalization.bantu.taggers.scale import (
            build_noun_scale_amount,
            build_scale_amount,
        )

        self.profile = profile
        numbers = [build_cardinal(profile, context) for context in profile.contexts]
        self.by_class = {}
        for context, number in zip(profile.contexts, numbers):
            if context.compounds:
                self.by_class.setdefault(context.class_id, number)
        self.roles = {}
        for name, _ in profile.roles:
            self.roles.setdefault(name, (build_role(profile, name) @ integer_domain(profile.max_value)).optimize())

        def cardinal_token(number):
            return token(self.add_tokens(pynutil.insert('integer: "') + number + pynutil.insert('"')))

        graphs = []
        for context, number in zip(profile.contexts, numbers):
            number = number | build_noun_scale_amount(profile, context)
            graphs.append(word_token(pynini.accep(context.noun)) + NEMO_SPACE + run(cardinal_token(number)))

        # Listed nouns are counted by their own cardinal above: a profile covers large
        # amounts after them with its places (Luganda abantu emitwalo ebiri) or with
        # after_counted_nouns. After a money cue or a currency word the amount is money,
        # handled by MoneyFst.
        excluded = [context.noun for context in profile.contexts]
        excluded += [currency.phrase.split()[-1] for currency in profile.currencies]
        excluded = pynini.union(*excluded, money_cue_word(profile)).optimize()
        any_noun = pynini.difference(ordinary_word, excluded).optimize()
        for scale in profile.counted_scales:
            if scale.counts_any_word:
                amount = build_scale_amount(profile, scale, "counting", invariant_context(profile))
                graphs.append(word_token(any_noun) + NEMO_SPACE + run(cardinal_token(amount)))

        self.fst = pynini.union(*graphs).optimize()
