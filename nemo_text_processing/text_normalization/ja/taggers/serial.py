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

from nemo_text_processing.text_normalization.ja.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_UPPER,
    TO_UPPER,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.ja.utils import get_abs_path


class SerialFst(GraphFst):
    """
    Finite state transducer for classifying compact serial/model identifiers.

    Examples:
        B2A23C -> name: "ビー 二 エー 二三 シー"
        MIG-25/235212-asdg
        -> name: "エムアイジー ハイフン 二五 スラッシュ 二三五二一二 ハイフン エーエスディージー"
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="serial", kind="classify", deterministic=deterministic)

        uppercase_letters = pynini.string_file(get_abs_path("data/latin/letters.tsv"))
        letters = (NEMO_UPPER | TO_UPPER) @ uppercase_letters
        letter_input = pynini.project(letters, "input")
        digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")) | pynini.string_file(
            get_abs_path("data/numbers/zero_maru.tsv")
        )
        insert_letter_digit_space = pynini.cdrewrite(pynutil.insert(" "), letter_input, NEMO_DIGIT, NEMO_SIGMA)
        insert_digit_letter_space = pynini.cdrewrite(pynutil.insert(" "), NEMO_DIGIT, letter_input, NEMO_SIGMA)
        alnum_spacing = insert_letter_digit_space @ insert_digit_letter_space
        raw_alnum = pynini.closure(letter_input | NEMO_DIGIT, 1)
        raw_letter_leading_alnum = (
            letter_input
            + pynini.closure(letter_input | NEMO_DIGIT)
            + NEMO_DIGIT
            + pynini.closure(letter_input | NEMO_DIGIT)
        )
        currency_prefix_input = pynini.project(
            pynini.string_file(get_abs_path("data/money/currency_prefix.tsv")), "input"
        )
        currency_code_number = currency_prefix_input + pynini.closure(NEMO_DIGIT, 1)
        raw_letter_leading_alnum = pynini.difference(raw_letter_leading_alnum, currency_code_number)
        raw_short_digit_leading_alnum = NEMO_DIGIT + (pynini.accep("x") | pynini.accep("X"))
        raw_mixed_alnum = raw_letter_leading_alnum | raw_short_digit_leading_alnum
        alnum_reader = pynini.closure(letters | digit | pynini.accep(" "), 1)
        alnum = (raw_mixed_alnum @ alnum_spacing @ alnum_reader).optimize()

        delimiter = insert_space + pynini.string_file(get_abs_path("data/serial/delimiter.tsv")) + insert_space
        unit_input = pynini.project(pynini.string_file(get_abs_path("data/measure/unit.tsv")), "input")
        numeric_measure_segment = pynini.closure(NEMO_DIGIT, 1) + unit_input
        raw_alnum_segment = pynini.difference(raw_alnum, numeric_measure_segment)
        raw_alnum_with_letter = (
            pynini.closure(letter_input | NEMO_DIGIT) + letter_input + pynini.closure(letter_input | NEMO_DIGIT)
        )
        raw_alnum_with_letter = pynini.difference(raw_alnum_with_letter, numeric_measure_segment)
        segment = (raw_alnum_segment @ alnum_spacing @ alnum_reader).optimize()
        segment_with_letter = (raw_alnum_with_letter @ alnum_spacing @ alnum_reader).optimize()
        delimited = segment_with_letter + pynini.closure(
            delimiter + segment, 1
        ) | segment + delimiter + segment_with_letter + pynini.closure(delimiter + segment)

        special_word = pynini.string_file(get_abs_path("data/serial/words.tsv"))
        covid_style = special_word + pynutil.delete("-") + insert_space + (NEMO_DIGIT**2 @ cardinal.just_cardinals)

        model_cue = pynini.string_file(get_abs_path("data/serial/model_cues.tsv"))
        model_number = model_cue + insert_space + delimited

        graph = pynutil.add_weight(covid_style, -0.1) | model_number | delimited | alnum
        self.fst = (pynutil.insert('name: "') + graph.optimize() + pynutil.insert('"')).optimize()
