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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, GraphFst, insert_space
from nemo_text_processing.text_normalization.en.utils import load_labels
from nemo_text_processing.text_normalization.ta.graph_utils import (
    CURRENCY_SYMBOLS,
    MONEY_SUFFIXES,
    NEMO_ALL_DIGIT,
    NEMO_TA_DIGIT,
    NEMO_TA_NON_ZERO,
    NEMO_TA_ZERO,
    POINT_WORD,
    RANGE_WORD,
)
from nemo_text_processing.text_normalization.ta.taggers.decimal import quantity_words
from nemo_text_processing.text_normalization.ta.utils import get_abs_path


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money, e.g.
        ₹50 -> money { currency_maj: "ரூபாய்" integer_part: "ஐம்பது" }
        ₹50.50 -> money { currency_maj: "ரூபாய்" integer_part: "ஐம்பது" fractional_part: "ஐம்பது" currency_min: "centiles" }
        ₹5 கோடி -> money { currency_maj: "ரூபாய்" integer_part: "ஐந்து கோடி" }
        ₹150க்கு -> money { currency_maj: "ரூபாய்" integer_part: "நூற்று ஐம்பது" morphosyntactic_features: "க்கு" }

    The ``centiles`` placeholder is resolved by the verbalizer to the minor currency word, and a
    case suffix written on the amount travels as ``morphosyntactic_features`` for the verbalizer
    to attach to the currency word. Reads ``data/money/currency.tsv`` (symbol or code -> word)
    and ``data/numbers/quantity_words.tsv``.

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="classify", deterministic=deterministic)

        currency_rows = [r for r in load_labels(get_abs_path("data/money/currency.tsv")) if len(r) >= 2]
        currency_graph = pynini.string_map([(k, v) for k, v, *_ in currency_rows]).optimize()
        rupee_word = dict((k, v) for k, v, *_ in currency_rows)["₹"]
        spaced, short, native = quantity_words()

        cardinal_graph = cardinal.final_graph

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\"") + insert_space, 0, 1
        )
        currency_major = pynutil.insert("currency_maj: \"") + currency_graph + pynutil.insert("\"")
        optional_space = pynini.closure(pynini.accep(" "), 0, 1)
        # Every branch that reads the symbol first shares this head, and with it one copy of
        # the amount. The weight that ranks a branch therefore rides on its tail: on the head it
        # would make the shared prefixes differ and the copies would not merge.
        currency_prefix = optional_graph_negative + currency_major + optional_space + insert_space
        open_amount = pynutil.insert("integer_part: \"")
        close_amount = pynutil.insert("\"")

        range_word = f" {RANGE_WORD} "
        range_amount = cardinal_graph + pynini.cross("-", range_word) + cardinal_graph
        integer = open_amount + cardinal_graph + pynutil.add_weight(close_amount, -0.1)
        integer_range = open_amount + range_amount + pynutil.add_weight(close_amount, -0.05)

        # ₹50.5 means 50 paise: a lone fractional digit is scaled by ten before lookup.
        one_digit_padded = pynini.union(NEMO_DIGIT + pynutil.insert("0"), NEMO_TA_DIGIT + pynutil.insert(NEMO_TA_ZERO))
        # .05 is five paise: a leading zero in the minor unit is dropped.
        zero_lead = pynini.union(pynutil.delete("0") + NEMO_DIGIT, pynutil.delete(NEMO_TA_ZERO) + NEMO_TA_DIGIT)
        two_digits = pynini.union(pynini.difference(NEMO_DIGIT, "0") + NEMO_DIGIT, NEMO_TA_NON_ZERO + NEMO_TA_DIGIT)
        fraction_digits = pynini.union(one_digit_padded, zero_lead, two_digits).optimize()
        fraction = pynutil.insert("fractional_part: \"") + (fraction_digits @ cardinal_graph) + pynutil.insert("\"")
        currency_minor = pynutil.insert("currency_min: \"centiles\"")
        minor_amount = optional_space + pynini.cross(".", " ") + fraction + insert_space + currency_minor

        optional_slash_dash = pynini.closure(
            pynutil.add_weight(pynini.closure(pynini.accep(" "), 0, 1) + pynutil.delete("/-"), -0.1), 0, 1
        )
        # A trailing .00 minor part is silent (₹1,999.00 -> ...ரூபாய்).
        delete_zero_frac = pynutil.delete(
            pynini.union(".00", "." + NEMO_TA_ZERO + NEMO_TA_ZERO, ".0", "." + NEMO_TA_ZERO)
        )

        # ₹5 கோடி style: the amount carries a scale word and the currency reads after it. English
        # scale words and the shorthands L/cr/K/M/B are spoken natively (₹2 lakh, ₹15L, $50M);
        # two scale words may stack (₹1 லட்சம் கோடி).
        quantity_word = (
            pynini.accep(" ") + spaced | pynutil.delete(pynini.closure(" ", 0, 1)) + insert_space + short
        ) + pynini.closure(pynini.accep(" ") + native, 0, 1)
        single_frac_digit = NEMO_ALL_DIGIT @ cardinal_graph
        point_word = f" {POINT_WORD} "
        amount_with_point = cardinal_graph + pynini.closure(
            pynini.cross(".", point_word) + (cardinal.digit_by_digit | single_frac_digit), 0, 1
        )
        # ₹5-10 கோடி reads as a range amount.
        amount_with_point |= amount_with_point + pynini.cross("-", range_word) + amount_with_point

        # ₹50, ₹50.50 and ₹1,999.00 all read the same integer amount.
        after_integer = (
            optional_slash_dash
            | minor_amount + optional_slash_dash
            | pynutil.add_weight(delete_zero_frac + optional_slash_dash, -0.1)
        )
        after_amount = pynutil.add_weight(quantity_word + close_amount + optional_slash_dash, -0.2)
        # ₹150க்கு: a case suffix on the amount is carried as a field and attached to the currency
        # word by the verbalizer; it may also follow a scale word. The verbalizer joins the suffix
        # onto ரூபாய் with sandhi, which wants the independent-vowel spelling of a glued ல்.
        written_suffix = pynini.union(*MONEY_SUFFIXES) | pynini.cross("ல்", "இல்")
        case_suffix = pynutil.insert(" morphosyntactic_features: \"") + written_suffix + pynutil.insert("\"")
        after_amount |= pynutil.add_weight(pynini.closure(quantity_word, 0, 1) + close_amount + case_suffix, -0.1)

        # ₹50.123: three or more minor digits are not paise; read as a decimal amount.
        long_fraction = pynini.compose(pynini.closure(NEMO_ALL_DIGIT, 3), cardinal.digit_by_digit)
        graph_long_fraction = (
            currency_prefix
            + open_amount
            + cardinal_graph
            + pynini.cross(".", point_word)
            + long_fraction
            + pynutil.add_weight(close_amount, 0.2)
        )

        # 50/- with no symbol is rupees.
        graph_slash_rupee = (
            pynutil.insert(f"currency_maj: \"{rupee_word}\"")
            + insert_space
            + integer
            + optional_space
            + pynutil.add_weight(pynutil.delete("/-"), -0.1)
        )

        # ₹.50 reads as paise only (symbol currencies only: Rs./ரூ. own the dot).
        symbol_currency = pynini.compose(pynini.union(*CURRENCY_SYMBOLS), currency_graph)
        currency_symbol_major = pynutil.insert("currency_maj: \"") + symbol_currency + pynutil.insert("\"")
        graph_bare_paise = (
            currency_symbol_major
            + optional_space
            + insert_space
            + pynutil.insert(f"integer_part: \"{cardinal.zero_word}\"")
            + pynini.cross(".", " ")
            + fraction
            + insert_space
            + pynutil.add_weight(currency_minor, -0.1)
        )

        # ₹-500: the sign may follow the symbol.
        negative_after_currency = (
            currency_major
            + optional_space
            + pynutil.insert(" negative: ")
            + pynini.cross("-", "\"true\"")
            + optional_space
            + insert_space
            + integer
            + pynutil.add_weight(optional_slash_dash, 0.1)
        )

        # The amount may also stand before the currency word (50 ரூபாய், 50.50 ரூபாய்).
        graph_major_only_suffix = (
            optional_graph_negative + integer + insert_space + optional_space + currency_major + optional_slash_dash
        )
        graph_major_and_minor_suffix = (
            optional_graph_negative
            + integer
            + optional_space
            + pynini.cross(".", " ")
            + fraction
            + optional_space
            + insert_space
            + currency_minor
            + insert_space
            + currency_major
            + optional_slash_dash
        )

        graph_currencies = (
            currency_prefix + integer + after_integer
            | currency_prefix + integer_range + optional_slash_dash
            | currency_prefix + open_amount + amount_with_point + after_amount
            | graph_long_fraction
            | graph_slash_rupee
            | graph_bare_paise
            | negative_after_currency
            | pynutil.add_weight(graph_major_only_suffix | graph_major_and_minor_suffix, 0.5)
        )

        self.fst = self.add_tokens(graph_currencies.optimize())
