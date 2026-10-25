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

"""Sentence-level tagging: numbers are resolved in context, over maximal runs of known number words.

Segmentation is internal to the compiled classifier, not a Python runtime pass.
Weights only prefer a normalization over leaving words unchanged. All numeric
readings have equal weight, so structure alone must make the reading unique;
the kernel tests check that every test input has exactly one best reading.
"""
import string

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.bantu.profile import CardinalProfile, invariant_context
from nemo_text_processing.inverse_text_normalization.bantu.taggers.cardinal import build_cardinal
from nemo_text_processing.inverse_text_normalization.bantu.taggers.money import build_amount
from nemo_text_processing.inverse_text_normalization.bantu.taggers.scale import (
    build_noun_scale_amount,
    build_scale_amount,
)

# Input words: lowercase ASCII letters (an apostrophe is allowed inside a word,
# e.g. Luganda n'ebiri), one space between words. Cased input and punctuation are
# handled by each language's own classifier, not by the kernel.
WORD = pynini.closure(pynini.union(*string.ascii_lowercase, "'"), 1).optimize()
# Internal markers on implied-currency amounts; removed before output.
IMPLIED = "#"  # allowed at the start of the input or right after a money cue
CUE_ONLY = "%"  # allowed only right after a money cue


def word_token(graph: pynini.Fst) -> pynini.Fst:
    return pynutil.insert('tokens { name: "') + graph + pynutil.insert('" }')


def cardinal_token(graph: pynini.Fst) -> pynini.Fst:
    return pynutil.insert('tokens { cardinal { integer: "') + graph + pynutil.insert('" } }')


def join_aware_runs(number_words, join_words):
    """Marks maximal runs of number words; a join word followed by an ordinary word ends the run.

    Joins such as Luganda mu also have nonnumeric uses ("in"). In
    "ente bbiri mu nnimiro" the run is "bbiri"; "mu nnimiro" stays as words.
    A run may still end on a join at the end of the input, where it is malformed
    and falls back whole (a cut-off amount is not guessed). A join between two
    number words still extends the run, so malformed internals fall back whole.
    """
    numeric_word = pynini.union(*sorted(number_words)).optimize()
    join_word = pynini.union(*sorted(join_words)).optimize()
    other_word = pynini.difference(WORD, numeric_word).optimize()
    plain_number = pynini.difference(numeric_word, join_word).optimize()
    any_word = (other_word | join_word).optimize()
    # Between or after runs: words and joins, containing at least one ordinary word.
    others = (pynini.closure(any_word + " ") + other_word + pynini.closure(" " + any_word)).optimize()
    numbers = numeric_word + pynini.closure(" " + numeric_word)
    before_others = pynini.closure(numeric_word + " ") + plain_number

    def marked(graph):
        return pynutil.insert("<") + graph + pynutil.insert(">")

    runs = pynini.closure(marked(before_others) + " " + others + " ") + (
        marked(before_others) + " " + others | marked(numbers)
    )
    segmenter = (others | pynini.closure(others + " ", 0, 1) + runs).optimize()
    return numeric_word, any_word, segmenter


def money_token(phrase: str, amount: pynini.Fst, code: str) -> pynini.Fst:
    return (
        pynutil.delete(phrase + " <")
        + pynutil.insert('tokens { money { integer_part: "')
        + amount
        + pynutil.insert(f'" currency: "{code}" }} }}')
        + pynutil.delete(">")
    )


def implied_money_token(amount: pynini.Fst, code: str, marker: str) -> pynini.Fst:
    """A bare amount whose currency is implied by policy (e.g. mitwalo -> UGX)."""
    return (
        pynutil.delete("<")
        + pynutil.insert(marker + 'tokens { money { integer_part: "')
        + amount
        + pynutil.insert(f'" currency: "{code}" }} }}')
        + pynutil.delete(">")
    )


def money_cue_word(profile: CardinalProfile) -> pynini.Fst:
    cues = profile.money_cues
    letters = pynini.closure(pynini.union(*string.ascii_lowercase, "'"))
    options = [letters + pynini.union(*cues.stems) + letters] if cues.stems else []
    options += [pynini.union(*cues.words)] if cues.words else []
    if not options:
        return pynini.Fst()
    cue = pynini.union(*options).optimize()
    if cues.not_cues:
        cue = pynini.difference(cue, pynini.union(*cues.not_cues).optimize())
    return cue.optimize()


def only_with_money_signal(profile: CardinalProfile) -> pynini.Fst:
    """Keep implied money only where something signals a payment, then drop markers.

    "nsasula mitwalo ebiri" (I pay ...) is money; "embuzi mitwalo ebiri" (goats)
    is left alone, because embuzi is not a money cue.
    """
    sigma = pynini.union(*(pynini.escape(chr(code)) for code in range(32, 127))).optimize()
    plain = pynini.difference(sigma, pynini.union(IMPLIED, CUE_ONLY)).optimize()
    after_cue = 'tokens { name: "' + money_cue_word(profile) + '" } ' + pynini.union(IMPLIED, CUE_ONLY)
    allowed = (pynini.closure(pynini.accep(IMPLIED), 0, 1) + pynini.closure(plain | after_cue)).optimize()
    strip = pynini.closure(plain | pynutil.delete(IMPLIED) | pynutil.delete(CUE_ONLY))
    return (allowed @ strip.optimize()).optimize()


def build_tagger(profile: CardinalProfile) -> pynini.Fst:
    numeric_word, other_word, segmenter = join_aware_runs(profile.number_words(), profile.join_words())
    fallback_run = (
        pynutil.delete("<")
        + word_token(numeric_word)
        + pynini.closure(" " + word_token(numeric_word))
        + pynutil.delete(">")
    )
    recognized = []
    for context in profile.contexts:
        recognized.append(
            word_token(pynini.accep(context.noun))
            + " "
            + pynutil.delete("<")
            + cardinal_token(build_cardinal(profile, context) | build_noun_scale_amount(profile, context))
            + pynutil.delete(">")
        )
    for currency in profile.currencies:
        recognized.append(money_token(currency.phrase, build_amount(profile, currency), currency.code))
    implied = []
    for scale in profile.counted_scales:
        if not scale.implied_currency:
            continue
        # Bare heads (mitwalo): money anywhere, or only with a signal.
        implied.append(
            implied_money_token(
                build_scale_amount(profile, scale, "bare"),
                scale.implied_currency,
                "" if scale.implied_anywhere else IMPLIED,
            )
        )
        # Augmented heads (emitwalo, akakadde): money only after a cue (or at the start).
        if scale.counting_plural or scale.counting_ones or scale.money_only_ones:
            implied.append(
                implied_money_token(
                    build_scale_amount(profile, scale, "counting_money"),
                    scale.implied_currency,
                    IMPLIED if scale.counting_money_at_start else CUE_ONLY,
                )
            )
    recognized.extend(implied)
    # Any other word + an augmented head counts that word, whether or not it is listed
    # (embuzi emitwalo ebiri = 20,000 goats). Money cues, currency words and listed
    # nouns are excluded: after a cue or a currency it is money, and listed nouns
    # already agree fully above.
    listed = [context.noun for context in profile.contexts]
    listed += [currency.phrase.split()[-1] for currency in profile.currencies]
    if profile.money_cues.stems or profile.money_cues.words:
        listed_or_cue = pynini.union(*listed, money_cue_word(profile))
    else:
        listed_or_cue = pynini.union(*listed)
    any_noun = pynini.difference(other_word, listed_or_cue.optimize()).optimize()
    for scale in profile.counted_scales:
        if scale.counts_any_word:
            amount = build_scale_amount(profile, scale, "counting", invariant_context(profile))
            recognized.append(
                word_token(any_noun) + " " + pynutil.delete("<") + cardinal_token(amount) + pynutil.delete(">")
            )
    chunks = pynini.union(
        *recognized,
        pynutil.add_weight(word_token(other_word), 1),
        pynutil.add_weight(fallback_run, 1),
    )
    sentence = chunks + pynini.closure(" " + chunks)
    tagger = segmenter @ sentence
    if implied:
        tagger @= only_with_money_signal(profile)
    return tagger.optimize()
