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

"""Helpers shared by the Bantu taggers: number-word runs, token formats and the money-signal filter.

The classifier reads the whole sentence at once. Maximal runs of number words are
marked first, so each run is read as a single number in the context of the word
before it (a counted noun, a currency phrase or a money cue). Weights only prefer a
normalization over leaving words unchanged. All numeric readings have equal weight,
so structure alone must make the reading unique; the kernel tests check that every
test input has exactly one best reading.
"""
import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.bantu.profile import CardinalProfile
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_CHAR,
    NEMO_NOT_SPACE,
    NEMO_SPACE,
    NEMO_WHITE_SPACE,
)

# Internal markers, removed before output. Control characters, so they never clash with input text.
RUN_START = "\x01"  # start of a run of number words
RUN_END = "\x02"  # end of a run of number words
IMPLIED = "\x03"  # implied-currency amount: allowed at the start of the input or right after a money cue
CUE_ONLY = "\x04"  # implied-currency amount: allowed only right after a money cue
MARKERS = pynini.union(RUN_START, RUN_END, IMPLIED, CUE_ONLY).optimize()

# Any input word: every character except whitespace. An apostrophe may be part of a
# number word (Luganda n'ebiri). Words with capitals, digits or punctuation are
# ordinary words and pass through unchanged.
WORD_CHAR = pynini.difference(NEMO_NOT_SPACE, MARKERS).optimize()
WORD = pynini.closure(WORD_CHAR, 1).optimize()
# Runs of whitespace between words become a single space.
SINGLE_SPACES = pynini.closure(WORD_CHAR | pynini.cross(pynini.closure(NEMO_WHITE_SPACE, 1), NEMO_SPACE)).optimize()


def word_token(graph: pynini.Fst) -> pynini.Fst:
    return pynutil.insert('tokens { name: "') + graph + pynutil.insert('" }')


def token(graph: pynini.Fst) -> pynini.Fst:
    """Wraps a class token, e.g. cardinal { ... }, as one item of the sentence."""
    return pynutil.insert("tokens { ") + graph + pynutil.insert(" }")


def run(graph: pynini.Fst) -> pynini.Fst:
    """Reads a whole marked run of number words with graph."""
    return pynutil.delete(RUN_START) + graph + pynutil.delete(RUN_END)


def join_aware_runs(number_words, join_words):
    """Marks maximal runs of number words; a join word followed by an ordinary word ends the run.

    Joins such as Luganda mu also have nonnumeric uses ("in"). In
    "ente bbiri mu nnimiro" the run is "bbiri"; "mu nnimiro" stays as words.
    A run may still end on a join at the end of the input, where it is malformed
    and falls back whole (a cut-off amount is not guessed). A join between two
    number words still extends the run, so malformed internals fall back whole.

    Returns (number word, ordinary word or join, segmenter).
    """
    numeric_word = pynini.union(*sorted(number_words)).optimize()
    join_word = pynini.union(*sorted(join_words)).optimize()
    other_word = pynini.difference(WORD, numeric_word).optimize()
    plain_number = pynini.difference(numeric_word, join_word).optimize()
    any_word = (other_word | join_word).optimize()
    # Between or after runs: words and joins, containing at least one ordinary word.
    others = (pynini.closure(any_word + NEMO_SPACE) + other_word + pynini.closure(NEMO_SPACE + any_word)).optimize()
    numbers = numeric_word + pynini.closure(NEMO_SPACE + numeric_word)
    before_others = pynini.closure(numeric_word + NEMO_SPACE) + plain_number

    def marked(graph):
        return pynutil.insert(RUN_START) + graph + pynutil.insert(RUN_END)

    runs = pynini.closure(marked(before_others) + NEMO_SPACE + others + NEMO_SPACE) + (
        marked(before_others) + NEMO_SPACE + others | marked(numbers)
    )
    segmenter = (others | pynini.closure(others + NEMO_SPACE, 0, 1) + runs).optimize()
    return numeric_word, any_word, segmenter


def money_cue_word(profile: CardinalProfile) -> pynini.Fst:
    """Words that signal a payment (e.g. Luganda money verbs); empty if the profile has none."""
    cues = profile.money_cues
    letters = pynini.closure(WORD_CHAR)
    options = [letters + pynini.union(*cues.stems) + letters] if cues.stems else []
    options += [pynini.union(*cues.words)] if cues.words else []
    if not options:
        return pynini.Fst()
    cue = pynini.union(*options).optimize()
    if cues.not_cues:
        cue = pynini.difference(cue, pynini.union(*cues.not_cues).optimize())
    return cue.optimize()


def only_with_money_signal(profile: CardinalProfile) -> pynini.Fst:
    """Keeps marked implied-currency amounts only where they are allowed, then drops the markers.

    An amount marked IMPLIED is money at the start of the input or right after a money
    cue; one marked CUE_ONLY only right after a money cue. Otherwise the tagging is
    rejected and another reading (the words unchanged, or a counted noun) is used.
    """
    plain = pynini.difference(NEMO_CHAR, pynini.union(IMPLIED, CUE_ONLY)).optimize()
    after_cue = 'tokens { name: "' + money_cue_word(profile) + '" }' + NEMO_SPACE + pynini.union(IMPLIED, CUE_ONLY)
    allowed = (pynini.closure(pynini.accep(IMPLIED), 0, 1) + pynini.closure(plain | after_cue)).optimize()
    strip = pynini.closure(plain | pynutil.delete(IMPLIED) | pynutil.delete(CUE_ONLY))
    return (allowed @ strip.optimize()).optimize()
