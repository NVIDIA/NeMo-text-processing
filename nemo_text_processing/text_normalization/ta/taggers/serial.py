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

from typing import List

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_ALPHA,
    NEMO_UPPER,
    TO_UPPER,
    GraphFst,
    convert_space,
    insert_space,
)
from nemo_text_processing.text_normalization.en.utils import load_labels
from nemo_text_processing.text_normalization.ta.graph_utils import NEMO_ALL_DIGIT, NEMO_ALL_NON_ZERO
from nemo_text_processing.text_normalization.ta.utils import get_abs_path

# Upper-case runs this long or shorter are spelled letter by letter (5G, PAN, KA 01 AB 1234); a
# longer run is an English word or acronym the voice reads on its own (COVID-19).
MAX_SPELLED_RUN = 4


def letter_names() -> 'pynini.FstLike':
    """
    One Latin letter, either case, to its spoken name from ``data/serial/letters.tsv``.
    """
    table = pynini.string_file(get_abs_path("data/serial/letters.tsv"))
    return pynini.union(table, TO_UPPER @ table).optimize()


def digit_words() -> 'pynini.FstLike':
    """
    One digit in either script to its word from ``data/telephone/number.tsv``.
    """
    return pynini.string_file(get_abs_path("data/telephone/number.tsv")).optimize()


def unit_letters() -> List[str]:
    """
    The single upper-case abbreviations in ``data/measure/unit.tsv`` (C, K, W, A, V ...).
    """
    rows = load_labels(get_abs_path("data/measure/unit.tsv"))
    return [row[0] for row in rows if len(row) >= 2 and len(row[0]) == 1 and row[0].isupper()]


def spelled(letter: 'pynini.FstLike', lower: int, upper: int) -> 'pynini.FstLike':
    """
    ``lower`` to ``upper`` letters read one name at a time, space-separated.
    """
    return letter + pynini.closure(insert_space + letter, lower - 1, upper - 1)


class SerialFst(GraphFst):
    """
    Finite state transducer for classifying alphanumeric codes, e.g.
        5G -> tokens { name: "ஐந்து ஜி" }
        COVID-19 -> tokens { name: "COVID பத்தொன்பது" }
        ABCDE1234F -> tokens { name: "ABCDE ஆயிரத்து இருநூற்று முப்பத்துநான்கு எஃப்" }
        KA 01 AB 1234 -> tokens { name: "கே ஏ பூஜ்யம் ஒன்று ஏ பி ஒன்று இரண்டு மூன்று நான்கு" }

    A code mixes Latin letters and digits, glued or joined by ``-`` or ``/``, and holds at least
    one upper-case letter; a digit group of one to four digits without a leading zero reads as a
    cardinal, any other group digit by digit. A hyphen chain of three or more digit groups that
    is neither a date nor a telephone number (1-800-555) is a code too. Dimensions (5x3) and a
    digit run glued to a unit abbreviation (170C) are left alone.

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="serial", kind="classify", deterministic=deterministic)

        any_digit = NEMO_ALL_DIGIT
        letter = letter_names()
        digit = digit_words()

        digit_by_digit = digit + pynini.closure(insert_space + digit)
        short_shape = NEMO_ALL_NON_ZERO + pynini.closure(any_digit, 0, 3)
        number = pynini.union(
            short_shape @ cardinal.final_graph,
            pynini.difference(pynini.closure(any_digit, 1), short_shape) @ digit_by_digit,
        )

        letters = pynini.union(
            spelled(letter, 1, MAX_SPELLED_RUN),
            pynini.difference(pynini.closure(NEMO_ALPHA, 1), pynini.closure(NEMO_UPPER, 1, MAX_SPELLED_RUN)),
        )

        # Letters and digits alternate inside a glued piece, so a digit run is never split.
        glued = pynini.union(
            letters
            + pynini.closure(insert_space + number + insert_space + letters)
            + pynini.closure(insert_space + number, 0, 1),
            number
            + pynini.closure(insert_space + letters + insert_space + number)
            + pynini.closure(insert_space + letters, 0, 1),
        )
        separator = pynutil.delete(pynini.union("-", "/")) + insert_space
        # Optimized before it is composed with the shape filters below: the alternation of
        # letters and numbers leaves tens of thousands of states that determinize to a few
        # thousand, and composing the unoptimized form explodes.
        code = (glued + pynini.closure(separator + glued)).optimize()

        # An Indian vehicle plate: state, district, series and a four-digit number, with or
        # without spaces or hyphens (KA 01 AB 1234, TN-09-AB-1234, MH12DE1433). Its groups always
        # read digit by digit, so the shape is taken out of the general code reading below,
        # which would otherwise read a four-digit group as a cardinal.
        one_or_two = pynini.closure(any_digit, 1, 2)
        gap = pynini.closure(pynini.union(" ", "-"), 0, 1)
        plate_shape = (
            NEMO_UPPER**2
            + gap
            + one_or_two
            + gap
            + pynini.closure(NEMO_UPPER, 1, 3)
            + gap
            + pynini.closure(any_digit, 4, 4)
        ).optimize()

        code_chars = pynini.closure(pynini.union(NEMO_ALPHA, any_digit, "-", "/"))
        digits = pynini.closure(any_digit, 1)
        # A digit run closing in a unit abbreviation belongs to the measure class, which reads it
        # only when a space separates the two (170 C); glued, it is left as written.
        measure_shape = digits + pynini.union(*unit_letters())
        shape = pynini.difference(
            pynini.intersect(code_chars + any_digit + code_chars, code_chars + NEMO_UPPER + code_chars),
            pynini.union(digits + pynini.union("x", "X") + digits, plate_shape, measure_shape),
        ).optimize()
        mixed = (shape @ code).optimize()

        # 1-800-555, 1-2-3: three or more groups of up to four digits, fewer than ten digits in
        # all (a telephone number has ten) and not a date.
        group = pynini.closure(any_digit, 1, 4)
        date = pynini.union(
            one_or_two + "-" + one_or_two + "-" + pynini.closure(any_digit, 2, 4),
            pynini.closure(any_digit, 4, 4) + "-" + one_or_two + "-" + one_or_two,
        )
        at_most_nine = pynini.closure(pynini.closure("-") + any_digit, 0, 9) + pynini.closure("-")
        chain_shape = pynini.difference(
            pynini.intersect(group + pynini.closure("-" + group, 2), at_most_nine), date
        ).optimize()
        chain = (
            chain_shape @ (number + pynini.closure(pynutil.delete("-") + insert_space + number, 2)).optimize()
        ).optimize()

        plate_sep = pynini.closure(pynutil.delete(pynini.union(" ", "-")), 0, 1) + insert_space
        plate = (
            plate_shape
            @ (
                spelled(letter, 2, 2)
                + plate_sep
                + (one_or_two @ digit_by_digit)
                + plate_sep
                + spelled(letter, 1, 3)
                + plate_sep
                + (pynini.closure(any_digit, 4, 4) @ digit_by_digit)
            ).optimize()
        ).optimize()

        # A short upper-case cue before a code is spelled with it: PNR 4X7K9M, PAN ABCDE1234F.
        cued = spelled(letter, 2, MAX_SPELLED_RUN) + pynutil.delete(" ") + insert_space + mixed

        graph = pynini.union(
            pynutil.add_weight(mixed, 0.2),
            pynutil.add_weight(chain, 0.3),
            pynutil.add_weight(plate, 0.1),
            pynutil.add_weight(cued, 0.1),
        )
        self.graph = convert_space(graph).optimize()
        self.fst = (pynutil.insert("name: \"") + self.graph + pynutil.insert("\"")).optimize()
