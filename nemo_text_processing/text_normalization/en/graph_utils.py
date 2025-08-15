# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import os
import string
from pathlib import Path
from typing import Dict

import pynini
from pynini import Far
from pynini.examples import plurals
from pynini.export import export
from pynini.lib import byte, pynutil, utf8

from nemo_text_processing.text_normalization.en.utils import get_abs_path, load_labels
from nemo_text_processing.utils.logging import logger

NEMO_CHAR = utf8.VALID_UTF8_CHAR

NEMO_DIGIT = byte.DIGIT
NEMO_LOWER = pynini.union(*string.ascii_lowercase).optimize()
NEMO_UPPER = pynini.union(*string.ascii_uppercase).optimize()
NEMO_ALPHA = pynini.union(NEMO_LOWER, NEMO_UPPER).optimize()
NEMO_ALNUM = pynini.union(NEMO_DIGIT, NEMO_ALPHA).optimize()
NEMO_VOWELS = pynini.union(*"aeiouAEIOU").optimize()
NEMO_CONSONANTS = pynini.union(*"BCDFGHJKLMNPQRSTVWXYZbcdfghjklmnpqrstvwxyz").optimize()
NEMO_HEX = pynini.union(*string.hexdigits).optimize()
NEMO_NON_BREAKING_SPACE = "\u00a0"
NEMO_SPACE = " "
NEMO_WHITE_SPACE = pynini.union(" ", "\t", "\n", "\r", "\u00a0").optimize()
NEMO_NOT_SPACE = pynini.difference(NEMO_CHAR, NEMO_WHITE_SPACE).optimize()
NEMO_NOT_QUOTE = pynini.difference(NEMO_CHAR, r'"').optimize()

NEMO_PUNCT = pynini.union(*map(pynini.escape, string.punctuation)).optimize()
NEMO_GRAPH = pynini.union(NEMO_ALNUM, NEMO_PUNCT).optimize()

NEMO_SIGMA = pynini.closure(NEMO_CHAR)
NEMO_LOWER_NOT_A = pynini.union(
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
).optimize()

delete_space = pynutil.delete(pynini.closure(NEMO_WHITE_SPACE))
delete_space_or_punct = NEMO_PUNCT | delete_space
delete_zero_or_one_space = pynutil.delete(pynini.closure(NEMO_WHITE_SPACE, 0, 1))
insert_space = pynutil.insert(" ")
delete_extra_space = pynini.cross(pynini.closure(NEMO_WHITE_SPACE, 1), " ")
delete_preserve_order = pynini.closure(
    pynutil.delete(" preserve_order: true")
    | (pynutil.delete(' field_order: "') + NEMO_NOT_QUOTE + pynutil.delete('"'))
)


# Common string literals; expand as you see fit.
username_string = "username"
double_quotes = '"'
domain_string = "domain"
protocol_string = "protocol"
slash = "/"
double_slash = "//"
triple_slash = "///"
file = "file"
period = "."
at = "@"
colon = ":"
https = "https"
http = "http"
www = "www"


suppletive = pynini.string_file(get_abs_path("data/suppletive.tsv"))
# _v = pynini.union("a", "e", "i", "o", "u")
_c = pynini.union(
    "b",
    "c",
    "d",
    "f",
    "g",
    "h",
    "j",
    "k",
    "l",
    "m",
    "n",
    "p",
    "q",
    "r",
    "s",
    "t",
    "v",
    "w",
    "x",
    "y",
    "z",
)
_ies = NEMO_SIGMA + _c + pynini.cross("y", "ies")
_es = NEMO_SIGMA + pynini.union("s", "sh", "ch", "x", "z") + pynutil.insert("es")
_s = NEMO_SIGMA + pynutil.insert("s")

graph_plural = plurals._priority_union(
    suppletive,
    plurals._priority_union(_ies, plurals._priority_union(_es, _s, NEMO_SIGMA), NEMO_SIGMA),
    NEMO_SIGMA,
).optimize()

SINGULAR_TO_PLURAL = graph_plural
PLURAL_TO_SINGULAR = pynini.invert(graph_plural)
TO_LOWER = pynini.union(*[pynini.cross(x, y) for x, y in zip(string.ascii_uppercase, string.ascii_lowercase)])
TO_UPPER = pynini.invert(TO_LOWER)
MIN_NEG_WEIGHT = -0.0001
MIN_POS_WEIGHT = 0.0001
INPUT_CASED = "cased"
INPUT_LOWER_CASED = "lower_cased"
MINUS = pynini.union("minus", "Minus").optimize()


def capitalized_input_graph(
    graph: "pynini.FstLike",
    original_graph_weight: float = None,
    capitalized_graph_weight: float = None,
) -> "pynini.FstLike":
    """
    Allow graph input to be capitalized, e.g. for ITN)

    Args:
        graph: FstGraph
        original_graph_weight: weight to add to the original `graph`
        capitalized_graph_weight: weight to add to the capitalized graph
    """
    capitalized_graph = pynini.compose(TO_LOWER + NEMO_SIGMA, graph).optimize()

    if original_graph_weight is not None:
        graph = pynutil.add_weight(graph, weight=original_graph_weight)

    if capitalized_graph_weight is not None:
        capitalized_graph = pynutil.add_weight(capitalized_graph, weight=capitalized_graph_weight)

    graph |= capitalized_graph
    return graph


def generator_main(file_name: str, graphs: Dict[str, "pynini.FstLike"]):
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
    logger.info(f"Created {file_name}")


def generate_far_filename(
    language: str,
    mode: str,  # "tn" or "itn"
    cache_dir: str,
    operation: str,  # "tokenize" or "verbalize"
    deterministic: bool = False,
    project_input: bool = False,
    input_case: str = INPUT_LOWER_CASED,
    whitelist_file: str = "",
) -> str:
    """
    Generate FAR filename based on parameters.

    Args:
        language: Language code (e.g., "en")
        mode: Either "tn" or "itn"
        cache_dir: Directory for cache files
        operation: Either "tokenize" or "verbalize"
        deterministic: If True, append "deterministic" to filename
        project_input: If True, append "projecting" to filename
        input_case: Input case handling, append if INPUT_CASED
        whitelist_file: Whitelist filename to include

    Returns:
        Complete path to FAR file
    """
    filename_parts = [language, mode]

    if deterministic:
        filename_parts.append("deterministic")

    if project_input:
        filename_parts.append("projecting")

    if input_case == INPUT_CASED:
        filename_parts.append(input_case)

    if whitelist_file:
        filename_parts.append(Path(whitelist_file).stem)

    filename_parts.append(operation)

    filename = "_".join(filename_parts) + ".far"
    return os.path.join(cache_dir, filename)


def get_plurals(fst):
    """
    Given singular returns plurals

    Args:
        fst: Fst

    Returns plurals to given singular forms
    """
    return SINGULAR_TO_PLURAL @ fst


def get_singulars(fst):
    """
    Given plural returns singulars

    Args:
        fst: Fst

    Returns singulars to given plural forms
    """
    return PLURAL_TO_SINGULAR @ fst


def convert_space(fst) -> "pynini.FstLike":
    """
    Converts space to nonbreaking space.
    Used only in tagger grammars for transducing token values within quotes, e.g. name: "hello kitty"
    This is making transducer significantly slower, so only use when there could be potential spaces within quotes, otherwise leave it.

    Args:
        fst: input fst

    Returns output fst where breaking spaces are converted to non breaking spaces
    """
    return fst @ pynini.cdrewrite(pynini.cross(NEMO_SPACE, NEMO_NON_BREAKING_SPACE), "", "", NEMO_SIGMA)


def string_map_cased(input_file: str, input_case: str = INPUT_LOWER_CASED):
    labels = load_labels(input_file)

    if input_case == INPUT_CASED:
        additional_labels = []
        for written, spoken, *weight in labels:
            written_capitalized = written[0].upper() + written[1:]
            additional_labels.extend(
                [
                    [
                        written_capitalized,
                        spoken.capitalize(),
                    ],  # first letter capitalized
                    [
                        written_capitalized,
                        spoken.upper().replace(" AND ", " and "),
                    ],  # # add pairs with the all letters capitalized
                ]
            )

            spoken_no_space = spoken.replace(" ", "")
            # add abbreviations without spaces (both lower and upper case), i.e. "BMW" not "B M W"
            if len(spoken) == (2 * len(spoken_no_space) - 1):
                logger.debug(f"This is weight {weight}")
                if len(weight) == 0:
                    additional_labels.extend(
                        [
                            [written, spoken_no_space],
                            [written_capitalized, spoken_no_space.upper()],
                        ]
                    )
                else:
                    additional_labels.extend(
                        [
                            [written, spoken_no_space, weight[0]],
                            [written_capitalized, spoken_no_space.upper(), weight[0]],
                        ]
                    )
        labels += additional_labels

    whitelist = pynini.string_map(labels).invert().optimize()
    return whitelist


class GraphFst:
    """
    Base class for all grammar fsts.

    Args:
        name: name of grammar class
        kind: either 'classify' or 'verbalize'
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, name: str, kind: str, deterministic: bool = True, project_input: bool = False):
        self.name = name
        self.kind = kind
        self._fst = None
        self.deterministic = deterministic
        self.project_input = project_input

        self.far_path = Path(os.path.dirname(__file__) + "/grammars/" + kind + "/" + name + ".far")
        if self.far_exist():
            self._fst = Far(self.far_path, mode="r", arc_type="standard", far_type="default").get_fst()

    def far_exist(self) -> bool:
        """
        Returns true if FAR can be loaded
        """
        return self.far_path.exists()

    @property
    def fst(self) -> "pynini.FstLike":
        return self._fst

    @fst.setter
    def fst(self, fst):
        self._fst = fst

    def add_tokens(self, fst) -> "pynini.FstLike":
        """
        Wraps class name around to given fst

        Args:
            fst: input fst

        Returns:
            Fst: fst
        """
        if self.project_input:
            return pynutil.insert('input: "') + fst.project('input') + pynutil.insert('"')
        return pynutil.insert(f"{self.name} {{ ") + fst + pynutil.insert(" }")

    def delete_tokens(self, fst) -> "pynini.FstLike":
        """
        Deletes class name wrap around output of given fst

        Args:
            fst: input fst.
            project: if True, adds input projection with brackets

        """
        if self.project_input:
            # Match input content: either NEMO_NOT_QUOTE or NEMO_NOT_QUOTE followed by a single quote
            # This handles cases like '12"' where input ends with a quote character
            input_content = pynini.union(pynini.closure(NEMO_NOT_QUOTE, 1), pynini.closure(NEMO_NOT_QUOTE, 1) + "\"")
            input_projection = (
                pynutil.delete(" input: \"")
                + pynutil.insert(r"\[")
                + input_content
                + pynutil.insert(r"\]")
                + pynutil.delete("\"")
            )
            input_projection = pynini.closure(input_projection, 0, 1)

            # Wrap main output in brackets when projecting input
            bracketed_fst = pynutil.insert(r"\[") + fst + pynutil.insert(r"\]")
            fst = bracketed_fst + input_projection

        res = (
            pynutil.delete(f"{self.name}")
            + delete_space
            + pynutil.delete("{")
            + delete_space
            + fst
            + delete_space
            + pynutil.delete("}")
        )
        return res @ pynini.cdrewrite(pynini.cross("\u00a0", " "), "", "", NEMO_SIGMA)
