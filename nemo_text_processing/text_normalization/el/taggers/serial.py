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
from pynini.examples import plurals
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.el.utils import get_abs_path, load_labels
from nemo_text_processing.text_normalization.en.graph_utils import (
    MIN_NEG_WEIGHT,
    MIN_POS_WEIGHT,
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_NOT_SPACE,
    NEMO_SIGMA,
    GraphFst,
    convert_space,
)


def _leading_zero_graph(cardinal: GraphFst) -> "pynini.FstLike":
    return pynini.compose(pynini.accep("0") + pynini.closure(NEMO_DIGIT), cardinal.single_digits_graph).optimize()


def _build_serial_graph(
    num_graph: "pynini.FstLike",
    delimiter: "pynini.FstLike",
    alphas: "pynini.FstLike",
) -> "pynini.FstLike":
    letter_num = alphas + delimiter + num_graph
    num_letter = pynini.closure(num_graph + delimiter, 1) + alphas
    next_alpha_or_num = pynini.closure(delimiter + (alphas | num_graph))
    next_alpha_or_num |= pynini.closure(
        delimiter
        + num_graph
        + plurals._priority_union(pynini.accep(" "), pynutil.insert(" "), NEMO_SIGMA).optimize()
        + alphas
    )

    serial_graph = letter_num + next_alpha_or_num
    serial_graph |= num_letter + next_alpha_or_num
    serial_graph |= num_graph + delimiter + num_graph + delimiter + num_graph + pynini.closure(delimiter + num_graph)

    symbols = [x[0] for x in load_labels(get_abs_path("data/whitelist/symbol.tsv"))]
    symbols = pynini.union(*symbols)
    serial_graph |= pynini.compose(NEMO_SIGMA + symbols + NEMO_SIGMA, num_graph + delimiter + num_graph)

    serial_graph = pynutil.add_weight(serial_graph, MIN_POS_WEIGHT)
    serial_graph |= (
        pynini.closure(NEMO_NOT_SPACE, 1) + (pynini.cross("^2", " squared") | pynini.cross("^3", " cubed")).optimize()
    )

    serial_graph = (
        pynini.closure((serial_graph | num_graph | alphas) + delimiter)
        + serial_graph
        + pynini.closure(delimiter + (serial_graph | num_graph | alphas))
    )
    return serial_graph.optimize()


class SerialFst(GraphFst):
    """
    Finite state transducer for classifying serial numbers in Greek,
        e.g.
            "A320" -> tokens { name: "άλφα τριακόσια είκοσι" }
            "H800" -> tokens { name: "έιτς οχτακόσια" }
            "a320b" -> tokens { name: "άλφα τρία δύο μηδέν βήτα" }

    Args:
        cardinal: Greek cardinal tagger instance.
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
        lm: whether to use for hybrid LM
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True, lm: bool = False):
        super().__init__(name="serial", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.graph_no_tokens

        # ---- Symbol support ----
        symbols_graph = pynini.string_file(get_abs_path("data/whitelist/symbol.tsv")).optimize()
        symbols_graph |= pynini.cross("#", "hash")

        if deterministic:
            num_graph_pure = (
                pynini.compose(NEMO_DIGIT ** (1, 3), cardinal_graph)  # 1-3 digits → cardinal speech
                | pynini.compose(NEMO_DIGIT ** (4, ...), cardinal.single_digits_graph)  # 4+ digits → single-digit
                | _leading_zero_graph(cardinal)
            ).optimize()

            num_graph_alnum = (
                pynini.compose(NEMO_DIGIT, cardinal_graph)  # single digit → cardinal
                | pynini.compose(NEMO_DIGIT**2, cardinal_graph)  # two digits → cardinal
                | pynutil.add_weight(
                    pynini.compose(NEMO_DIGIT + pynini.closure("0", 1), cardinal_graph),
                    MIN_NEG_WEIGHT,  # X0 → cardinal
                )
                | pynini.compose(
                    pynini.difference(NEMO_DIGIT**3, NEMO_DIGIT + NEMO_DIGIT + "00"),
                    cardinal.single_digits_graph,  # 3-digit not ending in 00 → single-digit
                )
                | pynini.compose(NEMO_DIGIT ** (4, ...), cardinal.single_digits_graph)  # 4+ digits → single-digit
                | _leading_zero_graph(cardinal)
            ).optimize()

            num_graph_slash = (
                pynini.compose(NEMO_DIGIT ** (1, 4), cardinal_graph)
                | pynini.compose(NEMO_DIGIT ** (5, ...), cardinal.single_digits_graph)
                | _leading_zero_graph(cardinal)
            ).optimize()

        else:
            num_graph_pure = cardinal.final_graph
            num_graph_alnum = cardinal.final_graph
            num_graph_slash = cardinal.final_graph

        num_graph_pure |= symbols_graph
        num_graph_alnum |= symbols_graph

        if not self.deterministic and not lm:
            num_graph_pure |= cardinal.single_digits_graph
            num_graph_pure |= pynini.compose(num_graph_pure, NEMO_SIGMA + pynutil.delete("hundred ") + NEMO_SIGMA)
            num_graph_pure |= pynutil.add_weight(
                NEMO_DIGIT**2 @ cardinal.graph_hundred_component_at_least_one_none_zero_digit,
                weight=MIN_POS_WEIGHT,
            )
            num_graph_alnum = num_graph_pure

        # add space between letter and digit/symbol
        symbols = [x[0] for x in load_labels(get_abs_path("data/whitelist/symbol.tsv"))]
        symbols = pynini.union(*symbols)
        digit_symbol = NEMO_DIGIT | symbols

        graph_with_space = pynini.compose(
            pynini.cdrewrite(pynutil.insert(" "), NEMO_ALPHA | symbols, digit_symbol, NEMO_SIGMA),
            pynini.cdrewrite(pynutil.insert(" "), digit_symbol, NEMO_ALPHA | symbols, NEMO_SIGMA),
        )
        graph_with_space = pynini.compose(
            graph_with_space,
            pynini.cdrewrite(pynutil.insert(" "), NEMO_ALPHA, NEMO_ALPHA, NEMO_SIGMA),
        )

        # ---- Greek letter names ----
        alphas = _get_alpha_map()

        # serial graph with delimiter
        delimiter = pynini.accep("-") | pynini.accep("/") | pynini.accep(" ")
        if not deterministic:
            delimiter |= pynini.cross("-", " dash ") | pynini.cross("/", " slash ")

        serial_graph = _build_serial_graph(num_graph_pure, delimiter, alphas)
        serial_graph_alnum = _build_serial_graph(num_graph_alnum, delimiter, alphas)

        # Rule 3: tokens that contain only digits and slashes (e.g. 31/31/100, 123/261788/2021).
        slash_digit_token = (
            pynini.closure(NEMO_DIGIT, 1) + pynini.accep("/") + pynini.closure(NEMO_DIGIT | pynini.accep("/"), 0)
        )
        slash_serial = pynini.compose(
            slash_digit_token,
            pynini.closure(num_graph_slash + pynini.accep("/"), 1) + num_graph_slash,
        ).optimize()
        serial_graph |= pynutil.add_weight(slash_serial, MIN_NEG_WEIGHT)

        serial_graph |= pynini.compose(graph_with_space, serial_graph_alnum.optimize()).optimize()
        serial_graph = pynini.compose(pynini.closure(NEMO_NOT_SPACE, 2), serial_graph).optimize()

        # exclude patterns like "import/export"
        serial_graph = pynini.compose(
            pynini.difference(
                NEMO_SIGMA,
                pynini.closure(NEMO_ALPHA, 1) + pynini.accep("/") + pynini.closure(NEMO_ALPHA, 1),
            ),
            serial_graph,
        )

        self.graph = serial_graph.optimize()
        graph = pynutil.insert("name: \"") + convert_space(self.graph).optimize() + pynutil.insert("\"")
        self.fst = graph.optimize()


def _get_alpha_map():
    letter_names = pynini.string_map(
        [
            ("a", "άλφα"),
            ("A", "άλφα"),
            ("b", "βήτα"),
            ("B", "βήτα"),
            ("c", "γάμα"),
            ("C", "γάμα"),
            ("d", "δέλτα"),
            ("D", "δέλτα"),
            ("e", "έψιλον"),
            ("E", "έψιλον"),
            ("f", "εφ"),
            ("F", "εφ"),
            ("g", "τζι"),
            ("G", "τζι"),
            ("h", "έιτς"),
            ("H", "έιτς"),
            ("i", "γιώτα"),
            ("I", "γιώτα"),
            ("j", "τζέι"),
            ("J", "τζέι"),
            ("k", "κάπα"),
            ("K", "κάπα"),
            ("l", "ελ"),
            ("L", "ελ"),
            ("m", "εμ"),
            ("M", "εμ"),
            ("n", "εν"),
            ("N", "εν"),
            ("o", "όμικρον"),
            ("O", "όμικρον"),
            ("p", "πι"),
            ("P", "πι"),
            ("q", "κιου"),
            ("Q", "κιου"),
            ("r", "αρ"),
            ("R", "αρ"),
            ("s", "ες"),
            ("S", "ες"),
            ("t", "ταυ"),
            ("T", "ταυ"),
            ("u", "ύψιλον"),
            ("U", "ύψιλον"),
            ("v", "βι"),
            ("V", "βι"),
            ("w", "νταμπλγιου"),
            ("W", "νταμπλγιου"),
            ("x", "χι"),
            ("X", "χι"),
            ("y", "γουάι"),
            ("Y", "γουάι"),
            ("z", "ζέτα"),
            ("Z", "ζέτα"),
        ]
    ).optimize()
    return pynini.closure(letter_names, 1)
