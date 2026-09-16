# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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

import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List
from unicodedata import category

import pynini
from pynini import Far
from pynini.export import export
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, NEMO_SIGMA, delete_space

# Tamil digits occupy U+0BE6 TAMIL DIGIT ZERO to U+0BEF TAMIL DIGIT NINE.
TA_DIGITS = "".join(chr(0x0BE6 + i) for i in range(10))
NEMO_TA_ZERO = TA_DIGITS[0]
NEMO_TA_DIGIT = pynini.union(*TA_DIGITS).optimize()
NEMO_TA_NON_ZERO = pynini.union(*TA_DIGITS[1:]).optimize()
# Combined Tamil and ASCII digits for graphs that read both scripts.
NEMO_ALL_DIGIT = pynini.union(NEMO_TA_DIGIT, NEMO_DIGIT).optimize()
NEMO_ALL_ZERO = pynini.union(NEMO_TA_ZERO, "0").optimize()
NEMO_ALL_NON_ZERO = pynini.union(NEMO_TA_NON_ZERO, pynini.difference(NEMO_DIGIT, "0")).optimize()

# The Tamil block U+0B80-U+0BFF; the letters are the block minus its digits.
NEMO_TA_BLOCK = pynini.union(*[chr(i) for i in range(0x0B80, 0x0C00)]).optimize()
NEMO_TA_LETTER = pynini.difference(NEMO_TA_BLOCK, NEMO_TA_DIGIT).optimize()

# Digit bridges between the two scripts, one digit at a time and over a whole run.
TA_TO_ASCII_DIGIT = pynini.string_map([(n, str(i)) for i, n in enumerate(TA_DIGITS)]).optimize()
ASCII_TO_TA_DIGIT = pynini.invert(TA_TO_ASCII_DIGIT).optimize()
# A run of digits in either script to ASCII, and a run of ASCII digits to Tamil.
TO_ASCII_DIGITS = pynini.closure(pynini.union(TA_TO_ASCII_DIGIT, NEMO_DIGIT)).optimize()
TO_TA_DIGITS = pynini.closure(ASCII_TO_TA_DIGIT).optimize()

MINUS_WORD = "மைனஸ்"
PLUS_WORD = "பிளஸ்"
# Spoken between the bounds of a range (10-20 -> பத்து முதல் இருபது).
RANGE_WORD = "முதல்"
POINT_WORD = "புள்ளி"
# Read between the parts of a non-idiomatic fraction: 5/77 -> ஐந்து கீழ் எழுபத்தேழு.
FRACTION_WORD = "கீழ்"

# Fractional-hour words used by the time and fraction grammars.
TA_KAAL = "கால்"
TA_ARAI = "அரை"
TA_MUKKAL = "முக்கால்"

# Day-part words TN fronts before a clock time (காலை பத்து மணி) and ITN reads back.
DAY_PARTS = ("காலை", "அதிகாலை", "மதியம்", "நண்பகல்", "மாலை", "இரவு", "முற்பகல்", "பிற்பகல்")
DAY_PART_ABBREVIATIONS = {
    "மு.ப.": "முற்பகல்",
    "மு.ப": "முற்பகல்",
    "பி.ப.": "பிற்பகல்",
    "பி.ப": "பிற்பகல்",
}
AM_WORD = "முற்பகல்"
PM_WORD = "பிற்பகல்"

# Vulgar fraction signs as spoken numerator and denominator words; the fraction verbalizer
# speaks the pair as its everyday word (ஒன்று/இரண்டு -> அரை).
VULGAR_PAIRS = {"½": ("ஒன்று", "இரண்டு"), "¼": ("ஒன்று", "நான்கு"), "¾": ("மூன்று", "நான்கு")}

# Case suffixes written glued to a money amount (₹150க்கு), attached to the currency word
# by the money verbalizer; a glued ல் is spelled இல் in the field, the form its sandhi takes.
MONEY_SUFFIXES = ("க்கு", "க்கும்", "க்குள்", "ஆக", "ஆல்", "இல்")

# Every case or ordinal suffix that may be written glued to a digit (2024ல், 100க்கு,
# 5வது); any other Tamil word glued to a digit is split off by the tokenizer.
GLUED_SUFFIXES = (
    "ல்",
    "இல்",
    "க்கு",
    "க்கும்",
    "க்குள்",
    "கள்",
    "களில்",
    "உம்",
    "ும்",
    "ஆக",
    "ஆல்",
    "ால்",
    "ஓடு",
    "உடன்",
    "ஐ",
    "ன்",
    "இன்",
    "லிருந்து",
    "இலிருந்து",
    "த்தில்",
    "த்துக்கு",
    "தான்",
    "ஆம்",
    "ம்",
    "ஆவது",
    "வது",
    "ஆவதாக",
    "வதாக",
)

# Currency symbols the money grammars read; also what may precede a re-fed written amount.
CURRENCY_SYMBOLS = "₹$£€¥₩₺৳₦"

MIN_NEG_WEIGHT = -0.0001
MIN_POS_WEIGHT = 0.0001


@lru_cache(maxsize=None)
def punctuation_code_points() -> List[str]:
    """
    Every Unicode punctuation code point, computed once per process on first use.

    The scan is ~1.1 M category lookups, so it is deferred: a process that only loads a
    compiled grammar from the FAR cache never pays for it.
    """
    return [chr(i) for i in range(sys.maxunicode + 1) if category(chr(i)).startswith("P")]


def rank(weight: float) -> 'pynini.FstLike':
    """
    A weight-carrying epsilon for the tail of a union branch: at the head the same weight
    would keep the branch's prefix from merging with its neighbours'.

    Args:
        weight: weight of the epsilon arc
    """
    return pynutil.insert("", weight)


def unweighted(fst: 'pynini.FstLike') -> 'pynini.FstLike':
    """
    Drops every arc weight, leaving only the consuming grammar's own weights to rank paths.

    Args:
        fst: input fst
    """
    return pynini.arcmap(fst.optimize(), map_type="rmweight").optimize()


def sequential(fst: 'pynini.FstLike') -> 'pynini.FstLike':
    """
    Input-deterministic form of an acyclic transducer, for grammars that read spoken words.

    An inverted TN grammar emits its digits before consuming any input (the TN side deleted
    them), so composing a string with it explores the whole digit skeleton at every word
    start, in every tagger that embeds it. Determinizing on the input delays each output
    until the input that decides it has been read, so composition explores one path per
    input prefix. The language, outputs and weights are unchanged.

    Args:
        fst: an acyclic transducer; several outputs for one input are kept as alternatives

    Raises:
        ValueError: if ``fst`` is cyclic, because determinization may then not terminate
    """
    acyclic = pynini.ACYCLIC
    if fst.properties(acyclic, True) != acyclic:
        raise ValueError("sequential() needs an acyclic transducer.")
    return pynini.determinize(fst, det_type="nonfunctional").optimize()


def generator_main(file_name: str, graphs: Dict[str, 'pynini.FstLike']):
    """
    Exports graph as OpenFst finite state archive (FAR) file with given file name and rule name.

    Args:
        file_name: exported file name
        graphs: Mapping of a rule name and Pynini WFST graph to be exported
    """
    exporter = export.Exporter(file_name)
    for rule, graph in graphs.items():
        exporter[rule] = graph.optimize()
    exporter.close()
    logging.info(f'Created {file_name}')


class GraphFst:
    """
    Base class for all grammar fsts.

    Args:
        name: name of grammar class
        kind: either 'classify' or 'verbalize'
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, name: str, kind: str, deterministic: bool = True):
        self.name = name
        self.kind = kind
        self._fst = None
        self.deterministic = deterministic

        self.far_path = Path(os.path.dirname(__file__) + '/grammars/' + kind + '/' + name + '.far')
        if self.far_exist():
            self._fst = Far(self.far_path, mode="r", arc_type="standard", far_type="default").get_fst()

    def far_exist(self) -> bool:
        """
        Returns true if FAR can be loaded
        """
        return self.far_path.exists()

    @property
    def fst(self) -> 'pynini.FstLike':
        return self._fst

    @fst.setter
    def fst(self, fst):
        self._fst = fst

    def add_tokens(self, fst) -> 'pynini.FstLike':
        """
        Wraps class name around to given fst

        Args:
            fst: input fst

        Returns:
            Fst: fst
        """
        return pynutil.insert(f"{self.name} {{ ") + fst + pynutil.insert(" }")

    def delete_tokens(self, fst) -> 'pynini.FstLike':
        """
        Deletes class name wrap around output of given fst

        Args:
            fst: input fst

        Returns:
            Fst: fst
        """
        res = (
            pynutil.delete(f"{self.name}")
            + delete_space
            + pynutil.delete("{")
            + delete_space
            + fst
            + delete_space
            + pynutil.delete("}")
        )
        return res @ pynini.cdrewrite(pynini.cross(" ", " "), "", "", NEMO_SIGMA)
