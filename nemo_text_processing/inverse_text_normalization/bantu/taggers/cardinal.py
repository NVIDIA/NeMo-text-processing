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

from nemo_text_processing.inverse_text_normalization.bantu.graph_utils import money_cue_word, run, token, word_token
from nemo_text_processing.inverse_text_normalization.bantu.profile import CardinalProfile, invariant_context
from nemo_text_processing.inverse_text_normalization.bantu.taggers.number import build_cardinal
from nemo_text_processing.inverse_text_normalization.bantu.taggers.scale import (
    build_noun_scale_amount,
    build_scale_amount,
)
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_SPACE, GraphFst


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying a counted number: a noun followed by a
    run of number words that agree with it. The noun stays a word; the number is
    fully resolved here, e.g.
        bakopa pideka vo baima -> tokens { name: "bakopa" } tokens { cardinal { integer: "21" } }

    Any other word followed by a counting scale (Luganda embuzi emitwalo ebiri,
    20,000 goats) is also a counted number. Here the count may only end in forms
    that are the same in every class, so no agreement with an unlisted noun is guessed.

    Args:
        profile: the language profile
        ordinary_word: any word that is not a number word
    """

    def __init__(self, profile: CardinalProfile, ordinary_word: pynini.Fst):
        super().__init__(name="cardinal", kind="classify")

        def cardinal_token(number):
            return token(self.add_tokens(pynutil.insert('integer: "') + number + pynutil.insert('"')))

        graphs = []
        for context in profile.contexts:
            number = build_cardinal(profile, context) | build_noun_scale_amount(profile, context)
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
