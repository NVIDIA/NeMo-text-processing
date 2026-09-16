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
from nemo_text_processing.inverse_text_normalization.ta.taggers.cardinal import (
    CardinalFst,
    kept_scale_words,
    optional_sign_field,
)
from nemo_text_processing.inverse_text_normalization.ta.taggers.decimal import money_fused
from nemo_text_processing.inverse_text_normalization.ta.utils import get_abs_path, load_rows
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, NEMO_SIGMA, delete_space
from nemo_text_processing.text_normalization.ta.graph_utils import MONEY_SUFFIXES, POINT_WORD, RANGE_WORD
from nemo_text_processing.text_normalization.ta.utils import get_abs_path as tn_abs_path


def _minor_unit_rows(major_to_symbol: Dict[str, str]) -> List[List[str]]:
    """
    Every minor-unit word TN can emit, paired with its major currency's symbol.

    The rows are derived from the TN ``data/money/major_minor_currencies.tsv`` the TN money
    verbalizer reads, so the two directions cannot drift apart; ``data/money/minor_units.tsv``
    adds only what that pairing cannot give (a plural TN never emits, the everyday காசு).
    """
    rows = [
        [minor, major_to_symbol[major]]
        for major, minor, *_ in load_rows(tn_abs_path("data/money/major_minor_currencies.tsv"), 2)
        if major in major_to_symbol
    ]
    seen = {tuple(row) for row in rows}
    rows += [row[:2] for row in load_rows(get_abs_path("data/money/minor_units.tsv"), 2) if tuple(row[:2]) not in seen]
    return rows


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying spoken money, e.g.
        ஐம்பது ரூபாய் -> money { integer_part: "50" currency: "₹" }
        ஐம்பது ரூபாய் ஐம்பது பைசா -> money { integer_part: "50" currency: "₹" fractional_part: "50" }
        ஐந்து கோடி ரூபாய் -> money { integer_part: "5 கோடி" currency: "₹" }
        ஐம்பது ரூபாய்க்கு -> money { integer_part: "50" currency: "₹" morphosyntactic_features: "க்கு" }

    Reads ``data/money/currency.tsv`` (spoken currency word -> symbol) and ``data/money/minor_units.tsv``.

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: CardinalFst):
        super().__init__(name="money", kind="classify")

        major_rows = [row[:2] for row in load_rows(get_abs_path("data/money/currency.tsv"), 2)]
        minor_rows = _minor_unit_rows(dict(major_rows))
        currency = pynini.string_map(major_rows)
        minor = pynini.string_map(minor_rows)
        # A minor unit belongs to one major currency: பைசா is rupees, சென்ட் is dollars. Grouping
        # them by symbol keeps ஐந்து டாலர் ஐம்பது பைசா from reading as $5.50.
        majors_by_symbol: Dict[str, List[str]] = {}
        minors_by_symbol: Dict[str, List[str]] = {}
        for word, symbol in major_rows:
            majors_by_symbol.setdefault(symbol, []).append(word)
        for word, symbol in minor_rows:
            minors_by_symbol.setdefault(symbol, []).append(word)

        # A case suffix on the currency or minor-unit word is carried into the written form
        # (₹50க்கு, ₹50.50க்கு).
        optional_suffix = pynini.closure(
            pynutil.insert(" morphosyntactic_features: \"") + pynini.union(*MONEY_SUFFIXES) + pynutil.insert("\""),
            0,
            1,
        )
        currency_field = pynutil.insert(" currency: \"") + currency + pynutil.insert("\"") + optional_suffix

        amount_words = cardinal.words_to_digits_licensed
        range_words = cardinal.words_to_digits + pynini.cross(" " + RANGE_WORD + " ", "-") + cardinal.words_to_digits
        integer_part = (
            pynutil.insert("integer_part: \"")
            + (amount_words | pynutil.add_weight(range_words, -0.5))
            + pynutil.insert("\"")
        )
        # A range takes no minor unit, so the minor paths embed the amount only once each.
        amount_part = pynutil.insert("integer_part: \"") + amount_words + pynutil.insert("\"")
        # A lone fractional digit is a tens value in paise (ஐந்து பைசா -> .05).
        two_digits = pynini.union(NEMO_DIGIT + NEMO_DIGIT, pynutil.insert("0") + NEMO_DIGIT)
        fractional_part = pynutil.insert(" fractional_part: \"") + (amount_words @ two_digits) + pynutil.insert("\"")

        graph = integer_part + delete_space + currency_field
        for symbol, minor_words in minors_by_symbol.items():
            if symbol not in majors_by_symbol:
                continue
            graph |= (
                amount_part
                + delete_space
                + pynutil.insert(f" currency: \"{symbol}\"")
                + pynutil.delete(pynini.union(*majors_by_symbol[symbol]))
                + delete_space
                + fractional_part
                + delete_space
                + pynutil.delete(pynini.union(*minor_words))
                + optional_suffix
            )

        # Currency word first: ரூபாய் ஐம்பது -> ₹50.
        graph |= (
            pynutil.insert("currency: \"")
            + currency
            + pynutil.insert("\"")
            + delete_space
            + pynutil.insert(" ")
            + integer_part
            + pynutil.insert(" preserve_order: true")
        )

        # Quantity-word money keeps the written idiom: ஐந்து கோடி ரூபாய் -> ₹5 கோடி, இரண்டு புள்ளி ஐந்து
        # லட்சம் ரூபாய் -> ₹2.5 லட்சம். Expanded scale words (ஆயிரம்) are digits. Two scale words
        # stack in the written idiom too: ஒரு லட்சம் கோடி ரூபாய் -> ₹1 லட்சம் கோடி.
        kept = kept_scale_words()
        quantity_written = pynini.union(*kept)
        stacked = pynini.union(*kept) + " " + quantity_written
        quantity_written = pynini.union(quantity_written, pynutil.add_weight(stacked, -0.1))
        short = cardinal.words_to_digits @ pynini.closure(NEMO_DIGIT, 1, 2)
        frac_digits = short + pynini.closure(delete_space + short)
        point = pynini.cross(pynini.accep(" ") + POINT_WORD + " ", ".")
        # The amount before a kept scale word holds no kept scale word itself: ஐந்து கோடி ஐம்பது
        # லட்சம் ரூபாய் is one number (₹55000000), not ₹50000050 லட்சம்.
        no_kept = pynini.difference(NEMO_SIGMA, NEMO_SIGMA + pynini.union(*kept) + NEMO_SIGMA)
        amount_digits = pynini.compose(no_kept, amount_words) + pynini.closure(point + frac_digits, 0, 1)
        # A fused half word is an amount too (ஒன்றரை லட்சம் ரூபாய் -> ₹1.5 லட்சம், ஒன்றரை ரூபாய் -> ₹1.50).
        fused_amount = money_fused((cardinal.words_to_digits_licensed @ pynini.closure(NEMO_DIGIT, 1, 3)).optimize())
        amount_digits |= fused_amount
        split_point = (
            pynini.closure(NEMO_DIGIT, 1)
            + pynini.cross(".", "\" fractional_part: \"")
            + (NEMO_DIGIT + NEMO_DIGIT | NEMO_DIGIT + pynutil.insert("0"))
        )
        graph |= (
            pynutil.insert("integer_part: \"")
            + (fused_amount @ split_point)
            + pynutil.insert("\"")
            + delete_space
            + currency_field
        )
        quantity_amount = (
            pynutil.insert("integer_part: \"")
            + amount_digits
            + pynini.accep(" ")
            + quantity_written
            + pynutil.insert("\"")
        )
        graph |= pynutil.add_weight(quantity_amount + delete_space + currency_field, -1.0)

        # Minor-unit-only amounts: ஐம்பது பைசா -> ₹0.50, ஐம்பது சென்ட் -> $0.50.
        graph |= (
            pynutil.insert("integer_part: \"0\"")
            + fractional_part
            + delete_space
            + pynutil.insert(" currency: \"")
            + minor
            + pynutil.insert("\"")
            + optional_suffix
        )

        # A spoken sign folds into the amount: மைனஸ் ஐந்நூறு ரூபாய் -> -₹500.
        self.fst = self.add_tokens(optional_sign_field() + graph).optimize()
