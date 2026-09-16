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

from typing import Callable

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.ta.graph_utils import GraphFst
from nemo_text_processing.inverse_text_normalization.ta.taggers.cardinal import (
    CardinalFst,
    half_form_rows,
    kept_scale_words,
    optional_sign_field,
)
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_CHAR, NEMO_DIGIT, delete_space, insert_space
from nemo_text_processing.text_normalization.ta.graph_utils import POINT_WORD, TA_ARAI, TA_KAAL, TA_MUKKAL

# Fraction digits of the quarter words, and the same quantity read as clock minutes.
QUARTER_FRACTION = {TA_KAAL: "25", TA_ARAI: "5", TA_MUKKAL: "75"}
FRACTION_MINUTES = {"5": "30", "25": "15", "75": "45"}


def half_form_graph(
    number: 'pynini.FstLike', prefix: str, infix: str, suffix: Callable[[str], str]
) -> 'pynini.FstLike':
    """
    Maps a fused -ரை half word (இருபத்தைந்தரை) to ``prefix INT infix suffix("5")``.

    ``data/numbers/half_forms.tsv`` lists these only up to பத்தரை; the rule is regular, so this
    covers the rest (TN writes the fused form for any integer ending in -உ).
    """
    stem = ((pynini.closure(NEMO_CHAR) + pynini.cross(TA_ARAI[1:], "ு")) @ number).optimize()
    return (pynutil.insert(prefix) + stem + pynutil.insert(infix + suffix(QUARTER_FRACTION[TA_ARAI]))).optimize()


def quarter_form_graph(
    number: 'pynini.FstLike', prefix: str, infix: str, suffix: Callable[[str], str]
) -> 'pynini.FstLike':
    """
    Maps an -ே linked quarter phrase (பத்தே கால், ஒன்றேகால்) to ``prefix INT infix suffix(frac)``.
    """
    stem = ((pynini.closure(NEMO_CHAR) + pynini.cross("ே", "ு")) @ number).optimize()
    optional_space = pynini.closure(pynutil.delete(" "), 0, 1)
    return pynini.union(
        *[
            pynutil.insert(prefix)
            + stem
            + pynutil.insert(infix)
            + optional_space
            + pynutil.delete(word)
            + pynutil.insert(suffix(fraction))
            for word, fraction in QUARTER_FRACTION.items()
        ]
    ).optimize()


def decimal_fused(small: 'pynini.FstLike') -> 'pynini.FstLike':
    """
    The fused fractional words beyond the table as decimal fields: பத்தே கால் -> 10.25,
    இருபத்தைந்தரை -> 25.5. ``small`` is the spoken integer part, at most three digits.
    """

    def suffix(fraction: str) -> str:
        return f" fractional_part: \"{fraction}\""

    return pynini.union(
        quarter_form_graph(small, "integer_part: \"", "\"", suffix),
        half_form_graph(small, "integer_part: \"", "\"", suffix),
    ).optimize()


def money_fused(short: 'pynini.FstLike') -> 'pynini.FstLike':
    """
    The fused fractional words as a digit amount with a point (ஒன்றரை -> 1.5, பத்தே கால் ->
    10.25), for an amount before a scale word or a currency word.
    """
    tabulated = pynini.union(*[pynini.cross(word, f"{ip}.{fp}") for word, ip, fp, *_ in half_form_rows()])
    quarters = quarter_form_graph(short, "", ".", lambda fraction: fraction)
    return pynini.union(tabulated, quarters).optimize()


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying spoken decimals, e.g.
        பன்னிரண்டு புள்ளி ஐந்து -> decimal { integer_part: "12" fractional_part: "5" }
        ஒன்று புள்ளி இரண்டு ஐந்து லட்சம் -> decimal { integer_part: "1" fractional_part: "25" quantity: "லட்சம்" }
        ஒன்றரை -> decimal { integer_part: "1" fractional_part: "5" }
        பத்தே கால் -> decimal { integer_part: "10" fractional_part: "25" }

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: CardinalFst):
        super().__init__(name="decimal", kind="classify")

        # Fractional digits are spoken one to three at a time; a scale word after the fraction
        # is a quantity, never more digits (ஐந்து புள்ளி ஐந்து லட்சம் -> 5.5 லட்சம்).
        short = cardinal.words_to_digits @ pynini.closure(NEMO_DIGIT, 1, 3)
        digit_by_digit = short + pynini.closure(delete_space + short)
        point = pynutil.delete(POINT_WORD)

        integer_part = pynutil.insert("integer_part: \"") + cardinal.words_to_digits + pynutil.insert("\"")
        fractional_part = pynutil.insert("fractional_part: \"") + digit_by_digit + pynutil.insert("\"")

        optional_sign = optional_sign_field()

        # A kept scale word after the fraction stays in the token, so the last fractional digit is
        # never read as its multiplier (ஒன்று புள்ளி இரண்டு ஐந்து லட்சம் -> 1.25 லட்சம்).
        quantity = pynini.closure(
            pynutil.insert(" quantity: \"")
            + pynutil.delete(" ")
            + pynini.union(*kept_scale_words())
            + pynutil.insert("\"")
            + pynutil.add_weight(pynini.accep(""), -0.2),
            0,
            1,
        )
        graph = (
            optional_sign
            + integer_part
            + delete_space
            + point
            + delete_space
            + insert_space
            + fractional_part
            + quantity
        )

        # Dotted chains round-trip: ஒன்று புள்ளி இரண்டு புள்ளி மூன்று -> 1.2.3.
        chain_fraction = (
            pynutil.insert("fractional_part: \"")
            + digit_by_digit
            + pynini.closure(pynini.cross(f" {POINT_WORD} ", ".") + digit_by_digit, 1)
            + pynutil.insert("\"")
        )
        graph |= pynutil.add_weight(
            optional_sign + integer_part + delete_space + point + delete_space + insert_space + chain_fraction, -0.1
        )

        # Fused fractional words: ஒன்றரை -> 1.5, பத்தரை -> 10.5; the table, and the regular
        # -ரை / -ே readings beyond it. Bare half/quarter nouns stay words.
        graph |= pynini.union(
            *[
                pynini.cross(word, f"integer_part: \"{ip}\" fractional_part: \"{fp}\"")
                for word, ip, fp, *_ in half_form_rows()
            ]
        )
        # A fused fraction's integer part is at most three digits, so the fused readings compose a
        # bounded slice rather than the whole number grammar.
        graph |= decimal_fused(short.optimize())

        self.fst = self.add_tokens(graph).optimize()
