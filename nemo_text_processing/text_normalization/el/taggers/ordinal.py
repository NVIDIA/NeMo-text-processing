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

from nemo_text_processing.text_normalization.el.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_SIGMA, GraphFst, insert_space

# Word-final boundary for the ending rewrites (space between compound components, or end of string).
_WORD_END = pynini.union(" ", "[EOS]")

# Greek ordinals are written as a digit followed by a gendered/case ending, e.g. "1ος", "1η",
# "1ο", "1ου", "21ές". We build the masculine-nominative form once ("εικοστός πρώτος") and rewrite
# every component's ending to the target gender/case. The masculine base ends either in an
# unaccented "ος" (accent on the stem: πρώτος, δεύτερος, ...) or an accented "ός" (accent on the
# ending: εικοστός, εκατοστός, ...); both patterns are handled in parallel so the accent lands
# correctly (πρώτος -> πρώτη but εικοστός -> εικοστή).
#
# (written ending, (target for stem-accented "ος", target for ending-accented "ός")) or None for
# the identity (masculine nominative) case.
_ENDINGS = [
    ("ος", None),  # masculine nominative singular
    ("η", ("η", "ή")),  # feminine nominative singular
    ("ο", ("ο", "ό")),  # neuter nom/acc singular, masculine accusative singular
    ("ον", ("ο", "ό")),  # masculine accusative singular (archaic -ον)
    ("ου", ("ου", "ού")),  # genitive singular (masculine/neuter)
    ("ης", ("ης", "ής")),  # feminine genitive singular
    ("οι", ("οι", "οί")),  # masculine nominative plural
    ("ες", ("ες", "ές")),  # feminine nominative/accusative plural
    ("α", ("α", "ά")),  # neuter nominative/accusative plural
    ("ων", ("ων", "ών")),  # genitive plural (all genders)
    ("ους", ("ους", "ούς")),  # masculine accusative plural
]


def _ending_rewrite(unaccented: str, accented: str) -> "pynini.FstLike":
    mapping = pynini.string_map([("ος", unaccented), ("ός", accented)])
    return pynini.cdrewrite(mapping, "", _WORD_END, NEMO_SIGMA)


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinals in Greek, e.g.
        1ος -> ordinal { integer: "πρώτος" }
        21η -> ordinal { integer: "εικοστή πρώτη" }
        3ου -> ordinal { integer: "τρίτου" }

    Supports ordinals 1-999 in every gender/case (nominative, genitive, accusative;
    singular and plural).

    Args:
        cardinal: CardinalFst (unused for now, kept for interface symmetry)
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst = None, deterministic: bool = True):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        ord_digit = pynini.string_file(get_abs_path("data/ordinal/digit.tsv"))  # 1-9
        ord_teen = pynini.string_file(get_abs_path("data/ordinal/teen.tsv"))  # 10-19
        ord_ties = pynini.string_file(get_abs_path("data/ordinal/ties.tsv"))  # 2-9 -> tens ordinal
        ord_hundreds = pynini.string_file(get_abs_path("data/ordinal/hundreds.tsv"))  # 1-9 -> hundreds ordinal

        # tens 10-99 (masculine nominative)
        tens_exact = ord_ties + pynutil.delete("0")
        tens_compound = ord_ties + insert_space + ord_digit
        graph_tens = ord_teen | tens_exact | tens_compound

        two_digit_rem = (pynutil.delete("0") + insert_space + ord_digit) | (insert_space + graph_tens)
        hundreds_exact = ord_hundreds + pynutil.delete("00")
        hundreds_rem = ord_hundreds + two_digit_rem
        graph_hundreds = hundreds_exact | hundreds_rem

        number_to_masc = (ord_digit | graph_tens | graph_hundreds).optimize()
        self.number_to_masc = number_to_masc
        # neuter ordinal forms, exposed for the fraction grammar (denominators)
        self.graph_neuter_sg = (number_to_masc @ _ending_rewrite("ο", "ό")).optimize()
        self.graph_neuter_pl = (number_to_masc @ _ending_rewrite("α", "ά")).optimize()

        graph = None
        for suffix, target in _ENDINGS:
            if target is None:
                inflected = number_to_masc
            else:
                inflected = number_to_masc @ _ending_rewrite(target[0], target[1])
            path = inflected + pynutil.delete(suffix)
            graph = path if graph is None else (graph | path)

        self.graph = graph.optimize()

        final_graph = pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")
        self.final_graph = final_graph
        self.fst = self.add_tokens(final_graph).optimize()
