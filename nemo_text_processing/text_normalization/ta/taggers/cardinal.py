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

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_CHAR, NEMO_SIGMA, insert_space
from nemo_text_processing.text_normalization.ta.graph_utils import (
    ASCII_TO_TA_DIGIT,
    GLUED_SUFFIXES,
    NEMO_ALL_DIGIT,
    NEMO_ALL_NON_ZERO,
    NEMO_TA_LETTER,
    TA_DIGITS,
    GraphFst,
    unweighted,
)
from nemo_text_processing.text_normalization.ta.utils import get_abs_path

# The joined spoken forms of 150-189 (நூற்றைம்பது) that ITN must read; TN itself emits the
# linking form (நூற்று ஐம்பது).
_JOINED_TENS = {
    5: ("நூற்றைம்பது", "நூற்றைம்பத்து"),
    6: ("நூற்றறுபது", "நூற்றறுபத்து"),
    7: ("நூற்றெழுபது", "நூற்றெழுபத்து"),
    8: ("நூற்றெண்பது", "நூற்றெண்பத்து"),
}

# Tails of the number words ending in -ஒன்று and the stem each takes when ஆயிரம் fuses onto
# it (இருபத்தொன்று ஆயிரம் -> இருபத்தோராயிரம்).
ONE_TAILS = (("ற்றொன்று", "ற்றோரா"), ("தொன்று", "தோரா"), ("ஒன்று", "ஓரா"))


def _ending(u: str, m: str) -> 'pynini.FstLike':
    """
    Rewrite of a final U+0BC1 TAMIL VOWEL SIGN U to ``u`` and of a final ம் to ``m``.
    """
    return pynini.union(pynini.cross("ு", u), pynini.cross("ம்", m))


_DATIVE = pynini.union("க்கு", "க்குள்", "க்கும்")
_OPTIONAL_I = pynutil.delete(pynini.closure("இ", 0, 1))

# Written case suffix and the rewrite of the number word's ending it calls for (None leaves
# the word as it is).
CASE_SUFFIXES = (
    (pynutil.delete(pynini.union("ல்", "இல்")), _ending("ில்", "த்தில்")),
    (pynutil.delete("த்தில்"), pynini.cross("ம்", "த்தில்")),
    (_DATIVE, pynini.difference(NEMO_CHAR, "்")),
    (_DATIVE, pynini.cross("ம்", "த்து")),
    (pynini.union("கள்", "களில்"), None),
    (pynini.cross("உம்", "ம்"), _ending("ு", "மு")),
    (pynutil.delete("ஆ") + pynini.accep("க"), _ending("ா", "மா")),
    (pynutil.delete("ஆ") + pynini.accep("ல்"), _ending("ா", "த்தா")),
    (pynutil.delete("ஓ") + pynini.accep("டு"), _ending("ோ", "த்தோ")),
    (pynutil.delete("உ") + pynini.accep("டன்"), _ending("ு", "த்து")),
    (pynutil.delete("ஐ"), _ending("ை", "த்தை")),
    (_OPTIONAL_I + pynini.accep("ன்"), _ending("ி", "த்தி")),
    (_OPTIONAL_I + pynini.accep("லிருந்து"), _ending("ி", "த்தி")),
    (pynini.accep("தான்"), None),
)
# Suffixes written with a bare vowel sign (5ால், 100ும்); TN reads them, ITN writes the
# independent-vowel spelling above instead.
SIGN_SPELLED_SUFFIXES = (
    (pynini.accep("ால்"), _ending("", "த்த")),
    (pynini.accep("ும்"), _ending("", "ம")),
)

# The adjectival stem replaces the cardinal's final -உ (or -ம்) with -ஆ: ஐந்து -> ஐந்தா,
# ஆயிரம் -> ஆயிரமா; then a written ordinal marker follows.
ORDINAL_STEM = NEMO_SIGMA + pynini.union(pynini.cross("ு", "ா"), pynini.cross("ம்", "மா"))
# ஆம் and the clipped ம் spell the same ordinal (28ஆம், 28ம்); any inflected tail after வத-
# is carried over (3ஆவதாக -> மூன்றாவதாக, 5வதுக்கு -> ஐந்தாவதுக்கு).
ORDINAL_MARKERS = pynini.union(
    pynutil.delete(pynini.union("வது", "ஆவது")) + pynutil.insert("வது"),
    pynutil.delete(pynini.union("ஆம்", "ம்")) + pynutil.insert("ம்"),
    pynutil.delete(pynini.closure("ஆ", 0, 1)) + pynini.accep("வத") + pynini.closure(NEMO_TA_LETTER, 1),
).optimize()


def _dual_script(table: 'pynini.FstLike') -> 'pynini.FstLike':
    """
    A table keyed by Tamil digits, also readable from ASCII digits.
    """
    return pynini.union(table, pynini.closure(ASCII_TO_TA_DIGIT) @ table).optimize()


def _digit(index: int) -> 'pynini.FstLike':
    """
    The digit ``index`` in either script.
    """
    return pynini.union(str(index), TA_DIGITS[index])


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g.
        -௨௩ -> cardinal { negative: "true" integer: "இருபத்துமூன்று" }
        2024ல் -> cardinal { integer: "இரண்டாயிரத்து இருபத்துநான்கில்" }
        007 -> cardinal { integer: "பூஜ்யம் பூஜ்யம் ஏழு" }

    Numbers up to the crore range are read as words; longer digit runs and leading-zero runs
    read digit by digit. The Indian (12,34,567) and international (1,234,567) grouping commas
    are accepted, both closing with a 3-digit group; 1,5 and 15,06 are not groupings.

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        digit = _dual_script(pynini.string_file(get_abs_path("data/numbers/digit.tsv")))
        zero = _dual_script(pynini.string_file(get_abs_path("data/numbers/zero.tsv")))
        teens_ties = _dual_script(pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv")))
        hundred = _dual_script(pynini.string_file(get_abs_path("data/numbers/hundred.tsv")))
        hundreds_exact = _dual_script(pynini.string_file(get_abs_path("data/numbers/hundreds_exact.tsv")))
        hundreds_combined = _dual_script(pynini.string_file(get_abs_path("data/numbers/hundreds_combined.tsv")))
        self.digit = digit
        self.zero = zero
        # The spoken zero, which the decimal and money grammars insert on their own.
        self.zero_word = pynini.shortestpath(zero.copy().project("output")).string()

        teens_and_ties = pynutil.add_weight(teens_ties, -0.1)
        zero_delete = pynutil.add_weight(pynutil.delete(_digit(0)), -0.1)

        def zeros(count: int) -> 'pynini.FstLike':
            return pynini.closure(zero_delete, count, count) if count else pynini.accep("")

        def scale(head, word: str, n_zeros: int, subs) -> 'pynini.FstLike':
            """
            Exact multiple (all trailing zeros) or the head plus a remainder.
            """
            suffix = pynutil.insert(word)
            graph = head + zeros(n_zeros) + suffix
            for count, sub in subs:
                graph |= head + suffix + zeros(count) + insert_space + sub
            return graph.optimize()

        # 100-199: நூறு, else the linking நூற்று; 150-189 join the hundred onto the tens.
        hundred_link = pynutil.insert(" நூற்று") + insert_space
        hundreds = (
            hundred
            | pynutil.delete(_digit(1) + _digit(0)) + hundred_link + digit
            | pynutil.delete(_digit(1)) + hundred_link + teens_ties
        )
        # TN keeps the linking form (150 -> நூற்று ஐம்பது); the joined forms are spoken variants
        # that ITN must read, so they stay in the graph a step behind.
        for k, (exact, stem) in _JOINED_TENS.items():
            joined = pynini.cross(_digit(1) + _digit(k) + _digit(0), exact)
            joined |= pynutil.delete(_digit(1) + _digit(k)) + pynutil.insert(stem) + insert_space + digit
            hundreds |= pynutil.add_weight(joined, 0.01)
        # 200-999: the exact hundreds, else the joined stem; 900 links as தொள்ளாயிரத்து.
        link = hundreds_combined | pynutil.delete(_digit(9)) + pynutil.insert(" தொள்ளாயிரத்து")
        hundreds |= hundreds_exact
        hundreds |= link + pynutil.delete(_digit(0)) + insert_space + digit
        hundreds |= link + insert_space + teens_ties
        hundreds = hundreds.optimize()

        below_thousand = [(2, digit), (1, teens_ties), (0, hundreds)]
        thousands = scale(digit, " ஆயிரம்", 3, below_thousand)
        ten_thousands = scale(teens_and_ties, " ஆயிரம்", 3, below_thousand)
        below_lakh = [(4, digit), (3, teens_ties), (2, hundreds), (1, thousands), (0, ten_thousands)]
        lakhs = scale(digit, " இலட்சம்", 5, below_lakh)
        ten_lakhs = scale(teens_and_ties, " இலட்சம்", 5, below_lakh)
        below_crore = [(6, digit), (5, teens_ties), (4, hundreds), (3, thousands)]
        below_crore += [(2, ten_thousands), (1, lakhs), (0, ten_lakhs)]
        crores = scale(digit, " கோடி", 7, below_crore)
        ten_crores = scale(teens_and_ties, " கோடி", 7, below_crore)

        # A leading zero is read out: 05 -> பூஜ்யம் ஐந்து.
        digit_word = (digit | zero).optimize()
        leading_zero = pynutil.add_weight(zero + insert_space + digit_word, 0.5)
        number = pynini.union(
            digit,
            zero,
            teens_and_ties,
            hundreds,
            thousands,
            ten_thousands,
            lakhs,
            ten_lakhs,
            crores,
            ten_crores,
            leading_zero,
        ).optimize()

        # Spacing is normalized inside the graph itself (the hundreds insert a leading space),
        # so inversion for ITN sees exactly the strings TN emits.
        squeeze = pynini.cdrewrite(pynini.cross(pynini.closure(" ", 2), " "), "", "", NEMO_SIGMA)
        strip_leading = pynini.cdrewrite(pynutil.delete(pynini.closure(" ", 1)), "[BOS]", "", NEMO_SIGMA)
        self.raw_graph = (number @ squeeze @ strip_leading).optimize()

        # Sandhi: after a stem ending ற்று, a ப/த-initial word doubles its consonant and joins,
        # e.g. நூற்று பத்து -> நூற்றுப்பத்து (110).
        sandhi = pynini.cdrewrite(
            pynini.union(pynini.cross(" ப", "ப்ப"), pynini.cross(" த", "த்த")), "ற்று", "", NEMO_SIGMA
        )
        # Scale-word style: exactly one thousand is bare ஆயிரம்; a counting prefix before a
        # scale word is ஒரு, not ஒன்று (ஒரு இலட்சம், ஒரு கோடி).
        exact_end = pynini.union("[EOS]", " கோடி")
        drop_one_exact = pynini.cdrewrite(pynini.cross("ஒன்று ஆயிரம்", "ஆயிரம்"), "[BOS]", exact_end, NEMO_SIGMA)
        drop_one_rest = pynini.cdrewrite(pynini.cross("ஒன்று ஆயிரம்", "ஆயிரத்து"), "[BOS]", " ", NEMO_SIGMA)
        oru_scales = pynini.cdrewrite(
            pynini.cross("ஒன்று ", "ஒரு "), "[BOS]", pynini.union("இலட்சம்", "கோடி"), NEMO_SIGMA
        )

        # Thousands fuse with the number word in front of them: இரண்டு ஆயிரம் -> இரண்டாயிரம், and
        # with a remainder இரண்டாயிரத்து (2024 -> இரண்டாயிரத்து இருபத்துநான்கு). Every multiplier
        # ends in U+0BC1 TAMIL VOWEL SIGN U, which the fusion replaces, except the -ஒன்று words,
        # whose tails take ஓர்; those go first so the generic rule never sees them. A compound
        # multiplier fuses on its last component, so no left context is imposed.
        def fuse(tail: str, right) -> 'pynini.FstLike':
            one_words = pynini.string_map([(f"{word} ஆயிரம்", f"{stem}{tail}") for word, stem in ONE_TAILS])
            rest = pynini.cross("ு ஆயிரம்", f"ா{tail}")
            return pynini.cdrewrite(one_words, "", right, NEMO_SIGMA) @ pynini.cdrewrite(rest, "", right, NEMO_SIGMA)

        # A scale word takes its oblique linking form when more of the number follows and its
        # nominative form when the number ends there: 200000 -> இரண்டு இலட்சம், but
        # 250000 -> இரண்டு இலட்சத்து ஐம்பதாயிரம். Thousands already do this above.
        oblique_scales = pynini.cdrewrite(
            pynini.union(pynini.cross("இலட்சம்", "இலட்சத்து"), pynini.cross("கோடி", "கோடியே")), "", " ", NEMO_SIGMA
        )
        # Kept as three stages: ITN must keep accepting the plainer spoken variants
        # (ஒரு இலட்சம் ஐம்பது ஆயிரம்) that TN itself no longer emits.
        self.style_scales = (sandhi @ drop_one_exact @ drop_one_rest @ oru_scales).optimize()
        self.style_fused = (self.style_scales @ fuse("யிரம்", exact_end) @ fuse("யிரத்து", " ")).optimize()
        # The bare digit reading, which ITN inverts; the grouping commas below are TN input only.
        self.number_graph = (self.raw_graph @ self.style_fused @ oblique_scales).optimize()

        # Grouping commas are deleted before the digits are read. A grouping opens with a
        # non-zero digit, so 00,000 is two zero runs and a comma.
        delete_comma = pynutil.delete(",")
        two, three = NEMO_ALL_DIGIT**2, NEMO_ALL_DIGIT**3
        head = NEMO_ALL_NON_ZERO + pynini.closure(NEMO_ALL_DIGIT, 0, 1)
        indian = head + pynini.closure(delete_comma + two) + delete_comma + three
        international = head + pynini.closure(NEMO_ALL_DIGIT, 0, 1) + pynini.closure(delete_comma + three, 1)
        grouped = pynini.union(indian, international).optimize()
        self.final_graph = pynini.union(self.number_graph, grouped @ self.number_graph).optimize()

        # Digit-by-digit fallback for shapes the number grammar rejects, e.g. leading-zero runs
        # (007) and digit strings beyond the crore range. A valid grouping beyond that range
        # (12,34,56,78,901) reads digit by digit as one token, so it is penalised only enough to
        # lose to any real number reading, not to a split at its commas.
        digit_by_digit = (digit_word + pynini.closure(insert_space + digit_word, 1)).optimize()
        self.digit_by_digit = digit_by_digit
        grouped_digit_by_digit = pynutil.add_weight(grouped @ digit_by_digit, -18.0)

        # Case-suffixed numbers, e.g. 2024ல் -> ...இருபத்துநான்கில்.
        self.suffixed_graph = self.attach_case_suffix(self.final_graph)

        # A sign is a field, so the verbalizer renders it and ITN can invert it.
        optional_sign = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\" ")
            | pynutil.insert("positive: ") + pynini.cross("+", "\"true\" "),
            0,
            1,
        )
        integer = (
            self.final_graph
            | pynutil.add_weight(self.suffixed_graph, 0.1)
            | pynutil.add_weight(self.digit_by_digit | grouped_digit_by_digit, 20.0)
        )
        graph = optional_sign + pynutil.insert("integer: \"") + integer + pynutil.insert("\"")
        self.fst = self.add_tokens(graph).optimize()

        # Every case or ordinal suffix that may stay glued to a digit; the tokenizer splits any
        # other Tamil word off a digit.
        ordinal_tail = pynini.union("வத", "ஆவத") + pynini.closure(NEMO_TA_LETTER, 1)
        self.known_suffixes = pynini.union(*GLUED_SUFFIXES, ordinal_tail).optimize()

    def attach_case_suffix(self, graph: 'pynini.FstLike', include_vowel: bool = True) -> 'pynini.FstLike':
        """
        Accepts a written case suffix after ``graph`` and attaches it to the last spoken word.

        Args:
            graph: a digits-to-words transducer
            include_vowel: if False, leave out the suffixes that are a bare vowel sign
        """
        rows = CASE_SUFFIXES + (SIGN_SPELLED_SUFFIXES if include_vowel else ())
        return pynini.union(
            *[(graph if ending is None else graph @ (NEMO_SIGMA + ending)) + written for written, ending in rows]
        ).optimize()

    def ordinal_graph(self, graph: 'pynini.FstLike') -> 'pynini.FstLike':
        """
        Reads ``graph`` followed by the written ordinal marker and an optional inflected tail.
        """
        return ((graph @ ORDINAL_STEM) + ORDINAL_MARKERS).optimize()

    def readable_years(self) -> 'pynini.FstLike':
        """
        The number readings ITN inverts, unweighted: the zero-deletion and teens bonuses are
        TN's own preferences and would otherwise decide ITN token boundaries.
        """
        return unweighted(self.number_graph)

    def itn_input_graph(self) -> 'pynini.FstLike':
        """
        Every spoken form ITN inverts: the styled number and the plainer variants TN itself no
        longer emits, minus the leading-zero pair, which ITN must read as the digit run 0 1
        rather than 01. Unweighted, because the zero-deletion and teens bonuses are TN's own
        preferences and would otherwise decide ITN token boundaries.
        """
        not_leading_zero = pynini.difference(NEMO_SIGMA, pynini.accep("பூஜ்யம் ") + NEMO_SIGMA)
        variants = pynini.union(
            self.raw_graph,
            self.raw_graph @ self.style_scales,
            self.raw_graph @ self.style_fused,
            self.number_graph,
        )
        return unweighted(variants @ not_leading_zero)
