# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Portuguese (PT) text normalization graph utilities.

Self-contained module with no dependency on en.graph_utils. Provides character/digit
symbols (NEMO_*), space helpers (delete_space, insert_space, delete_extra_space),
GraphFst base class, generator_main for FAR export, and PT-specific helpers
(filter_cardinal_punctuation, shift_cardinal_gender_pt).
"""

import os
import string
from pathlib import Path
from typing import Dict

import pynini
from pynini import Far
from pynini.export import export
from pynini.lib import byte, pynutil, utf8

from nemo_text_processing.utils.logging import logger

# ---- Character/digit symbols (same semantics as EN) ----
NEMO_CHAR = utf8.VALID_UTF8_CHAR
NEMO_DIGIT = byte.DIGIT
NEMO_LOWER = pynini.union(*string.ascii_lowercase).optimize()
NEMO_UPPER = pynini.union(*string.ascii_uppercase).optimize()
NEMO_ALPHA = pynini.union(NEMO_LOWER, NEMO_UPPER).optimize()
NEMO_SPACE = " "
NEMO_WHITE_SPACE = pynini.union(" ", "\t", "\n", "\r", "\u00a0").optimize()
NEMO_NOT_QUOTE = pynini.difference(NEMO_CHAR, pynini.accep('"')).optimize()
NEMO_SIGMA = pynini.closure(NEMO_CHAR)

delete_space = pynutil.delete(pynini.closure(NEMO_WHITE_SPACE))
insert_space = pynutil.insert(" ")
delete_extra_space = pynini.cross(
    pynini.closure(NEMO_WHITE_SPACE, 1), " "
).optimize()


def generator_main(file_name: str, graphs: Dict[str, "pynini.FstLike"]) -> None:
    """
    Export one or more graphs to an OpenFst Finite State Archive (FAR) file.

    Args:
        file_name: path to the output .far file.
        graphs: mapping of rule names to FST graphs to export.
    """
    exporter = export.Exporter(file_name)
    for rule, graph in graphs.items():
        exporter[rule] = graph.optimize()
    exporter.close()
    logger.info(f"Created {file_name}")


class GraphFst:
    """
    Base class for all Portuguese text normalization grammar FSTs.

    Args:
        name: name of the grammar (e.g. "cardinal", "decimal").
        kind: either "classify" or "verbalize".
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization).
    """

    def __init__(self, name: str, kind: str, deterministic: bool = True):
        self.name = name
        self.kind = kind
        self._fst = None
        self.deterministic = deterministic

        self.far_path = Path(
            os.path.dirname(os.path.abspath(__file__)) + "/grammars/" + kind + "/" + name + ".far"
        )
        if self.far_exist():
            self._fst = Far(
                self.far_path, mode="r", arc_type="standard", far_type="default"
            ).get_fst()

    def far_exist(self) -> bool:
        return self.far_path.exists()

    @property
    def fst(self) -> "pynini.FstLike":
        return self._fst

    @fst.setter
    def fst(self, fst):
        self._fst = fst

    def add_tokens(self, fst) -> "pynini.FstLike":
        return pynutil.insert(f"{self.name} {{ ") + fst + pynutil.insert(" }")

    def delete_tokens(self, fst) -> "pynini.FstLike":
        res = (
            pynutil.delete(f"{self.name}")
            + delete_space
            + pynutil.delete("{")
            + delete_space
            + fst
            + delete_space
            + pynutil.delete("}")
        )
        return res @ pynini.cdrewrite(
            pynini.cross("\u00a0", " "), "", "", NEMO_SIGMA
        )


# ---- PT-specific (Brazilian: 1.000.000 or 1 000 000) ----
cardinal_separator = pynini.string_map([".", " "])


def filter_cardinal_punctuation(fst: "pynini.FstLike") -> "pynini.FstLike":
    """
    Parse digit groups separated by cardinal_separator (e.g. 1.000.000) then apply fst.

    Args:
        fst: FST that maps digit string to verbalized cardinal.

    Returns:
        Composed FST that accepts digit strings with optional thousand separators.
    """
    exactly_three = NEMO_DIGIT**3
    up_to_three = pynini.closure(NEMO_DIGIT, 1, 3)
    cardinal_string = pynini.closure(NEMO_DIGIT, 1)
    cardinal_string |= (
        up_to_three
        + pynutil.delete(cardinal_separator)
        + pynini.closure(exactly_three + pynutil.delete(cardinal_separator))
        + exactly_three
    )
    return cardinal_string @ fst


def shift_cardinal_gender_pt(fst: "pynini.FstLike") -> "pynini.FstLike":
    """
    Apply Portuguese masculine-to-feminine conversion for cardinal strings, e.g.
        "um" -> "uma", "dois" -> "duas", "duzentos" -> "duzentas".

    Args:
        fst: FST producing masculine cardinal verbalization.

    Returns:
        FST that produces feminine form when composed with the same input.
    """
    fem_ones = pynini.cdrewrite(
        pynini.cross("um", "uma"),
        "",
        pynini.union(NEMO_SPACE, pynini.accep("[EOS]"), pynini.accep('"')),
        NEMO_SIGMA,
    )
    fem_twos = pynini.cdrewrite(
        pynini.cross("dois", "duas"),
        "",
        pynini.union(NEMO_SPACE, pynini.accep("[EOS]"), pynini.accep('"')),
        NEMO_SIGMA,
    )
    fem_hundreds = pynini.cdrewrite(
        pynini.cross("entos", "entas"),
        pynini.union(
            "duz", "trez", "quatroc", "quinh", "seisc", "setec", "oitoc", "novec"
        ),
        pynini.union(NEMO_SPACE, pynini.accep("[EOS]"), pynini.accep('"')),
        NEMO_SIGMA,
    )
    return fst @ fem_ones @ fem_twos @ fem_hundreds
