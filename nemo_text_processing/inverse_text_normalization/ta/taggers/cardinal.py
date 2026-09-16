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

from typing import Dict, List

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.ta.graph_utils import GraphFst
from nemo_text_processing.inverse_text_normalization.ta.utils import get_abs_path, load_rows
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_CHAR, NEMO_DIGIT, NEMO_SIGMA
from nemo_text_processing.text_normalization.ta.graph_utils import (
    MINUS_WORD,
    NEMO_TA_DIGIT,
    NEMO_TA_LETTER,
    PLUS_WORD,
    POINT_WORD,
    sequential,
)

# Colloquial (spoken/ASR) forms rewritten to the formal words the grammar knows.
_COLLOQUIAL = [
    ("ஒன்னு", "ஒன்று"),
    ("ஒண்ணு", "ஒன்று"),
    ("ரெண்டு", "இரண்டு"),
    ("மூணு", "மூன்று"),
    ("நாலு", "நான்கு"),
    ("அஞ்சு", "ஐந்து"),
    ("ஒம்பது", "ஒன்பது"),
    ("பூஜ்ஜியம்", "பூஜ்யம்"),
    ("பன்னெண்டு", "பன்னிரண்டு"),
    ("பன்னிரெண்டு", "பன்னிரண்டு"),
    ("அம்பது", "ஐம்பது"),
    ("ஐநூறு", "ஐந்நூறு"),
    # U+0BA9 TAMIL LETTER NNA spelling of 300 (முந்நூறு is the grammar's form).
    ("முன்னூறு", "முந்நூறு"),
    # U+0BA9 TAMIL LETTER NNA misspelling of 90 (தொண்ணூறு is the grammar's form).
    ("தொன்ணூறு", "தொண்ணூறு"),
    # 13 without its U+0BA9 TAMIL LETTER NNA (பதின்மூன்று is the grammar's form).
    ("பதிமூன்று", "பதின்மூன்று"),
    ("முன்னூற்று", "முந்நூற்று"),
]

# Colloquial/formal -த்தி tens joints normalized to the -த்து stems.
_TENS_JOINTS = [
    ("இருவத்தி", "இருபத்து"),
    ("இருபத்தி", "இருபத்து"),
    ("முப்பத்தி", "முப்பத்து"),
    ("நாப்பத்தி", "நாற்பத்து"),
    ("நாற்பத்தி", "நாற்பத்து"),
    ("அம்பத்தி", "ஐம்பத்து"),
    ("ஐம்பத்தி", "ஐம்பத்து"),
    ("அறுபத்தி", "அறுபத்து"),
    ("எழுபத்தி", "எழுபத்து"),
    ("எண்பத்தி", "எண்பத்து"),
    ("தொண்ணூத்தி", "தொண்ணூற்று"),
    ("தொண்ணூற்றி", "தொண்ணூற்று"),
    ("நூத்தி", "நூற்று"),
    ("இருநூத்தி", "இருநூற்று"),
    ("முன்னூத்தி", "முந்நூற்று"),
    ("முன்னூற்றி", "முந்நூற்று"),
    ("ஆயிரத்தி", "ஆயிரத்து"),
    ("ரெண்டாயிர", "இரண்டாயிர"),
    ("மூணாயிர", "மூன்றாயிர"),
]

# Compound linkers and scale-word spellings normalized to the grammar's own forms.
_SCALE_LINKS = [
    ("கோடியே", "கோடி"),
    ("இலட்சத்து", "இலட்சம்"),
    ("லட்சத்து", "இலட்சம்"),
    ("லட்சம்", "இலட்சம்"),
    ("ஓராயிரம்", "ஆயிரம்"),
    ("ஓர் ஆயிரம்", "ஆயிரம்"),
    ("ஒரு ஆயிரம்", "ஆயிரம்"),
]

_TENS_STEMS = [
    "இருபத்து",
    "முப்பத்து",
    "நாற்பத்து",
    "ஐம்பத்து",
    "அறுபத்து",
    "எழுபத்து",
    "எண்பத்து",
    "தொண்ணூற்று",
]

# Dependent vowel sign paired with the independent vowel it stands for, used both to fuse a
# spaced tens+digit pair and to split a solid one back apart.
_VOWEL_SIGNS = [("ொ", "ஒ"), ("ி", "இ"), ("ெ", "எ"), ("ே", "ஏ"), ("ை", "ஐ"), ("ா", "ஆ")]
_VOWELS = [vowel for _, vowel in _VOWEL_SIGNS]

# Every string in the language, kept away from the Tamil digits an ASCII-only grammar never
# writes.
_NO_NATIVE_DIGITS = pynini.closure(pynini.difference(NEMO_CHAR, NEMO_TA_DIGIT)).optimize()


def _boundary_rewrite(pairs: List) -> 'pynini.FstLike':
    """
    Word-boundary-anchored rewrite for the given (spoken, formal) pairs.
    """
    tau = pynini.union(*[pynini.cross(a, b) for a, b in pairs])
    edge = pynini.union("[BOS]", " ")
    right = pynini.union("[EOS]", " ")
    return pynini.cdrewrite(tau, edge, right, NEMO_SIGMA).optimize()


def _colloquial_chain() -> 'pynini.FstLike':
    """
    Normalizes spoken/colloquial number phrasing to the forms the TN grammar emits.
    """
    edge = pynini.union("[BOS]", " ")
    colloquial = _boundary_rewrite(_COLLOQUIAL)
    joints = pynini.cdrewrite(pynini.union(*[pynini.cross(a, b) for a, b in _TENS_JOINTS]), edge, "", NEMO_SIGMA)
    scale_links = _boundary_rewrite(_SCALE_LINKS)
    # Fused thousands split back to the spaced reading: அறுபதாயிரம் -> அறுபது ஆயிரம்.
    # தொள்ளாயிரம் (900) also contains ாயிரம், so a ள just before blocks the split.
    split_thousands = pynini.cdrewrite(
        pynini.union(pynini.cross("ாயிரத்து", "ு ஆயிரம்"), pynini.cross("ாயிரம்", "ு ஆயிரம்")),
        pynini.difference(NEMO_CHAR, pynini.accep("ள")),
        "",
        NEMO_SIGMA,
    )
    # Colloquial -ஞ்சு endings after த/ன read as -ைந்து (பதினஞ்சு -> பதினைந்து).
    nju = pynini.cdrewrite(pynini.cross("ஞ்சு", "ைந்து"), pynini.union("த", "ன"), pynini.union("[EOS]", " "), NEMO_SIGMA)
    # A -தி joint written solid onto a vowel-initial digit takes a ய glide (எண்பத்தியொன்று) or
    # fuses ந to ன (எண்பத்தினான்கு); split it back to the spaced reading so the joining stages
    # below can rebuild the grammar's own sandhi form.
    stems = pynini.union(*_TENS_STEMS)
    glide_split = pynini.union(*[pynini.cross(f"ய{sign}", f" {vowel}") for sign, vowel in _VOWEL_SIGNS])
    unfuse = (
        pynini.cdrewrite(glide_split, stems, "", NEMO_SIGMA)
        @ pynini.cdrewrite(pynini.cross("ன", " ந"), stems, "ா", NEMO_SIGMA)
        @ pynini.cdrewrite(pynutil.insert(" "), stems, pynini.union(*_VOWELS), NEMO_SIGMA)
    )

    # A spaced tens+digit pair joins into the fused sandhi form the grammar accepts:
    # consonant-initial digits join directly, vowel-initial digits merge the tens-final ு with
    # their initial vowel (நாற்பத்து ஒன்று -> நாற்பத்தொன்று).
    stems_lopped = pynini.union(*[stem[:-1] for stem in _TENS_STEMS])
    join_consonant = pynini.cdrewrite(pynutil.delete(" "), edge + stems, pynini.union("மூன்று", "நான்கு"), NEMO_SIGMA)
    vowel_merge = pynini.union(*[pynini.cross(f"ு {vowel}", sign) for sign, vowel in _VOWEL_SIGNS])
    join_vowel = pynini.cdrewrite(vowel_merge, edge + stems_lopped, "", NEMO_SIGMA)
    return (
        colloquial @ nju @ joints @ unfuse @ scale_links @ split_thousands @ join_consonant @ join_vowel
    ).optimize()


def _colloquial_domain() -> 'pynini.FstLike':
    """
    Strings some colloquial stage can rewrite; the chain is the identity on anything else, which
    the raw reading already covers.
    """
    triggers = (
        [spoken for spoken, _ in _COLLOQUIAL + _TENS_JOINTS + _SCALE_LINKS]
        + _TENS_STEMS
        + ["ஞ்சு", "ாயிரத்து", "ாயிரம்"]
        + [f"ு {vowel}" for vowel in _VOWELS]
    )
    return (pynini.closure(NEMO_CHAR) + pynini.union(*triggers) + pynini.closure(NEMO_CHAR)).optimize()


def _hundreds_split() -> 'pynini.FstLike':
    """
    Splits the spoken hundreds sandhi back into the spaced form: நூற்றிரண்டு -> நூற்று இரண்டு.
    """
    signs = ["ி", "ொ", "ெ", "ே", "ை", "ா"]
    vowels = ["இ", "ஒ", "எ", "ஏ", "ஐ", "ஆ"]
    stems = pynini.union("நூற்ற", "ஆயிரத்த")
    unmerge = pynini.union(*[pynini.cross(s, f"ு {v}") for s, v in zip(signs, vowels)])
    rewrite = pynini.cdrewrite(unmerge, stems, "", NEMO_SIGMA)
    # Restricted to strings that actually carry the sandhi, so this second reading of the input
    # costs a small composition instead of a whole extra copy of the grammar.
    domain = (pynini.closure(NEMO_CHAR) + stems + pynini.union(*signs) + pynini.closure(NEMO_CHAR)).optimize()
    return pynini.compose(domain, rewrite).optimize()


def spoken_pre_map() -> 'pynini.FstLike':
    """
    Every reading of the spoken words the cardinal tries: the words as spoken (preferred, since
    the colloquial rewrites would destroy the sandhi forms TN itself emits), the colloquial
    chain, and the hundreds sandhi split back apart. Each rewrite is restricted to the strings it
    can change, so it costs a small composition rather than a second copy of the number grammar.
    """
    raw = pynutil.add_weight(pynini.closure(NEMO_CHAR), -0.01)
    colloquial = pynini.compose(_colloquial_domain(), _colloquial_chain())
    return pynini.union(raw, colloquial, _hundreds_split()).optimize()


def ambiguous_words(condition: str) -> List:
    """
    The ``(word, reading)`` pairs of ``data/numbers/ambiguous.tsv`` whose admission condition is
    ``condition``: ``licensed`` words (ஒரு, ஓர், also the indefinite article) count as a number
    only inside a money or clock reading; ``standalone`` words (கால், அரை, முக்கால்) read as a
    fraction on their own but not before another Tamil word.
    """
    rows = load_rows(get_abs_path("data/numbers/ambiguous.tsv"), 3)
    return [(word, reading) for word, row_condition, reading, *_ in rows if row_condition == condition]


def licensed_words() -> 'pynini.FstLike':
    """
    ஒரு / ஓர் to 1: also the indefinite article, so a number only inside a money or clock reading.
    """
    return pynini.string_map(ambiguous_words("licensed")).optimize()


def optional_sign_field() -> 'pynini.FstLike':
    """
    Consumes a leading spoken sign word, emitting the ``negative``/``positive`` field.
    """
    negative = pynini.cross(MINUS_WORD + " ", "\"true\" ")
    positive = pynini.cross(PLUS_WORD + " ", "\"true\" ")
    return pynini.closure(pynutil.insert("negative: ") + negative | pynutil.insert("positive: ") + positive, 0, 1)


def half_form_rows() -> List[List[str]]:
    """
    Rows of ``data/numbers/half_forms.tsv`` (fused word, integer digits, fraction digits).
    """
    return load_rows(get_abs_path("data/numbers/half_forms.tsv"), 3)


def scale_word_rows() -> List[List[str]]:
    """
    Rows of ``data/numbers/scale_words.tsv`` (scale word, trailing zeros, expand|keep).
    """
    return load_rows(get_abs_path("data/numbers/scale_words.tsv"), 3)


def kept_scale_words() -> List[str]:
    """
    Scale words a written amount keeps as a word (5.5 லட்சம், ₹2.5 கோடி).
    """
    return [word for word, _, policy, *_ in scale_word_rows() if policy != "expand"]


def _scale_expanded(plain: 'pynini.FstLike') -> 'pynini.FstLike':
    """
    Multiplies out a scale word small enough for it, e.g. ஐந்து புள்ளி ஐந்து ஆயிரம் -> 5500.
    """
    words_by_zeros: Dict[int, List[str]] = {}
    for word, zeros, policy, *_ in scale_word_rows():
        if policy == "expand":
            words_by_zeros.setdefault(int(zeros), []).append(word)
    if not words_by_zeros:
        return pynini.Fst()

    half_rows = half_form_rows()
    point = pynutil.delete(" " + POINT_WORD + " ")
    graphs = []
    for zeros, words in words_by_zeros.items():
        tail = pynutil.delete(" " + pynini.union(*words))
        # The fractional digits shift left by the scale's zero count, so the padding inserted
        # after them follows the width that matched.
        shifted = pynini.union(
            *[(plain @ (NEMO_DIGIT**width)) + pynutil.insert("0" * (zeros - width)) for width in range(1, zeros + 1)]
        )
        digits_1_3 = pynini.closure(NEMO_DIGIT, 1, 3)
        graphs.append((plain @ pynini.difference(digits_1_3, pynini.accep("0"))) + point + shifted + tail)
        # A zero integer part is dropped, not kept as a leading zero, and an all-zero result
        # collapses to a single 0.
        drop_zero = pynutil.delete((plain @ pynini.accep("0")).project("input"))
        all_zeros = pynini.accep("0" * zeros)
        graphs.append(drop_zero + point + (shifted @ pynini.difference(NEMO_DIGIT**zeros, all_zeros)) + tail)
        graphs.append(drop_zero + point + (shifted @ pynini.cross("0" * zeros, "0")) + tail)
        # The fused half words scale the same way: ஒன்றரை ஆயிரம் -> 1500.
        graphs.append(
            pynini.union(
                *[
                    pynini.cross(f"{fused} {word}", str(int(ip + fp.ljust(zeros, "0"))))
                    for fused, ip, fp, *_ in half_rows
                    for word in words
                ]
            )
        )
    return pynini.union(*graphs).optimize()


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying spoken cardinals, e.g.
        இருபத்துமூன்று -> cardinal { integer: "23" }
        இரண்டாயிரத்து இருபத்துநான்கில் -> cardinal { integer: "2024ல்" }
        மைனஸ் நூற்று இருபது -> cardinal { negative: "true" integer: "120" }

    The spoken forms are the TN cardinal's own number grammar inverted, so the two directions
    cannot drift apart, read through a pre-map that normalizes colloquial phrasing (ரெண்டு,
    இருவத்தி மூணு, நூத்தி ஐம்பது) to the forms TN emits.

    Args:
        tn_cardinal: the Tamil TN CardinalFst, whose number grammar is inverted here
    """

    def __init__(self, tn_cardinal: 'GraphFst'):
        super().__init__(name="cardinal", kind="classify")

        # Every written form the TN grammar accepts, restricted to ASCII digits and inverted.
        inverted = pynini.invert(pynini.compose(_NO_NATIVE_DIGITS, tn_cardinal.itn_input_graph())).optimize()

        self.pre_map = spoken_pre_map()
        plain = self.read(inverted)
        # A decimal amount times a small scale word is one number: ஐந்து புள்ளி ஐந்து ஆயிரம் -> 5500.
        # The two readings are made sequential separately: determinizing their union re-times
        # every delayed output and blows up.
        scaled = _scale_expanded(plain)
        if scaled.num_states() > 0:
            scaled = sequential(scaled)
        self.words_to_digits = pynini.union(plain, scaled).optimize()
        self.words_to_digits_licensed = sequential(pynini.union(self.words_to_digits, licensed_words()))

        # A case suffix on the last number word is carried into the written form, spelled with
        # its independent vowel (ஐந்தால் -> 5ஆல்), never as a bare vowel sign.
        keep_suffix = pynini.closure(NEMO_DIGIT) + pynini.closure(NEMO_TA_LETTER)
        suffixed = (
            pynini.invert(
                pynini.compose(
                    _NO_NATIVE_DIGITS,
                    tn_cardinal.attach_case_suffix(tn_cardinal.readable_years(), include_vowel=False),
                )
            )
            @ keep_suffix
        ).optimize()
        self.words_to_digits_suffixed = self.read(suffixed)

        graph = self.words_to_digits | pynutil.add_weight(self.words_to_digits_suffixed, 0.1)
        graph = optional_sign_field() + pynutil.insert("integer: \"") + graph + pynutil.insert("\"")
        self.fst = self.add_tokens(graph).optimize()

    def read(self, lexicon: 'pynini.FstLike') -> 'pynini.FstLike':
        """
        Reads spoken words through the pre-map into ``lexicon``, input-deterministically.
        """
        return sequential(self.pre_map @ lexicon)
