# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_CHAR,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    delete_space,
)
from nemo_text_processing.text_normalization.es.graph_utils import roman_to_int, strip_accent
from nemo_text_processing.text_normalization.es.utils import get_abs_path
from pynini.lib import pynutil

digit = pynini.invert(pynini.string_file(get_abs_path("data/ordinals/digit.tsv")))
teens = pynini.invert(pynini.string_file(get_abs_path("data/ordinals/teen.tsv")))
twenties = pynini.invert(pynini.string_file(get_abs_path("data/ordinals/twenties.tsv")))
ties = pynini.invert(pynini.string_file(get_abs_path("data/ordinals/ties.tsv")))
hundreds = pynini.invert(pynini.string_file(get_abs_path("data/ordinals/hundreds.tsv")))
gender_suffix = pynini.string_file(get_abs_path("data/ordinals/gender_suffix.tsv"))


def get_one_to_one_thousand(cardinal: 'pynini.FstLike') -> 'pynini.FstLike':
    """
    Produces an acceptor for verbalizations of all numbers from 1 to 1000. Needed for ordinals and fractions.

    Args:
        cardinal: CardinalFst

    Returns:
        fst: A pynini.FstLike object
    """
    numbers = pynini.string_map([str(_) for _ in range(1, 1000)]) @ cardinal
    return pynini.project(numbers, "output").optimize()


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinal
        	"21.º" -> ordinal { integer: "vigésimo primero" morphosyntactic_features: "gender_masc" }
    This class converts ordinal up to the millionth (millonésimo) order (exclusive).

    This FST also records the ending of the ordinal (called "morphosyntactic_features"):
    either as gender_masc, gender_fem, or apocope. Also introduces plural feature for non-deterministic graphs.

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="ordinal", kind="classify")
        cardinal_graph = cardinal.graph

        graph_digit = digit.optimize()
        graph_teens = teens.optimize()
        graph_ties = ties.optimize()
        graph_twenties = twenties.optimize()
        graph_hundreds = hundreds.optimize()

        if not deterministic:
            # Some alternative derivations
            graph_ties = graph_ties | pynini.cross("sesenta", "setuagésimo")

            graph_teens = graph_teens | pynini.cross("once", "decimoprimero")
            graph_teens |= pynini.cross("doce", "decimosegundo")

            graph_digit = graph_digit | pynini.cross("nueve", "nono")
            graph_digit |= pynini.cross("siete", "sétimo")

        graph_tens_component = (
            graph_teens
            | (graph_ties + pynini.closure(pynini.cross(" y ", NEMO_SPACE) + graph_digit, 0, 1))
            | graph_twenties
        )

        graph_hundred_component = pynini.union(
            graph_hundreds + pynini.closure(NEMO_SPACE + pynini.union(graph_tens_component, graph_digit), 0, 1),
            graph_tens_component,
            graph_digit,
        )

        # Need to go up to thousands for fractions
        self.one_to_one_thousand = get_one_to_one_thousand(cardinal_graph)

        thousands = pynini.cross("mil", "milésimo")

        graph_thousands = (
            strip_accent(self.one_to_one_thousand) + NEMO_SPACE + thousands
        )  # Cardinals become prefix for thousands series. Snce accent on the powers of ten we strip accent from leading words
        graph_thousands @= pynini.cdrewrite(delete_space, "", "milésimo", NEMO_SIGMA)  # merge as a prefix
        graph_thousands |= thousands

        self.multiples_of_thousand = (cardinal_graph @ graph_thousands).optimize()

        if (
            not deterministic
        ):  # Formally the words preceding the power of ten should be a prefix, but some maintain word boundaries.
            graph_thousands |= (self.one_to_one_thousand @ graph_hundred_component) + NEMO_SPACE + thousands

        graph_thousands += pynini.closure(NEMO_SPACE + graph_hundred_component, 0, 1)

        ordinal_graph = graph_thousands | graph_hundred_component
        ordinal_graph = cardinal_graph @ ordinal_graph

        if not deterministic:
            # The 10's and 20's series can also be two words
            split_words = pynini.cross("decimo", "décimo ") | pynini.cross("vigesimo", "vigésimo ")
            split_words = pynini.cdrewrite(split_words, "", NEMO_CHAR, NEMO_SIGMA)
            ordinal_graph |= ordinal_graph @ split_words

        # If "octavo" is preceeded by the "o" within string, it needs deletion
        ordinal_graph @= pynini.cdrewrite(pynutil.delete("o"), "", "octavo", NEMO_SIGMA)

        self.graph = ordinal_graph.optimize()

        delete_period = pynini.closure(pynutil.delete("."), 0, 1)  # Sometimes the period is omitted
        convert_gender_suffix = delete_period + gender_suffix

        # Managing Romanization, excluding words that may be ambiguous
        roman_ordinals = roman_to_int(ordinal_graph)
        # exceptions = pynini.accep("vi") | pynini.accep("di") | pynini.accep("mi")
        exceptions = pynini.string_file(get_abs_path("data/ordinals/roman_exceptions.tsv"))
        graph_exception = pynini.project(exceptions, 'input')
        roman_ordinals = (pynini.project(roman_ordinals, "input") - graph_exception.arcsort()) @ roman_ordinals

        graph_roman = pynutil.insert("integer: \"") + roman_ordinals + pynutil.insert("\"")

        if not deterministic:
            # Introduce plural
            plural = pynini.closure(pynutil.insert("/plural"), 0, 1)
            convert_gender_suffix += plural

            # Romanizations have no morphology marker, so in non-deterministic case we provide option for all
            insert_morphology = pynutil.insert(convert_gender_suffix) + plural
            insert_morphology = (
                pynutil.insert(" morphosyntactic_features: \"") + insert_morphology + pynutil.insert("\"")
            )

            graph_roman += insert_morphology

        else:
            # We insert both genders as default
            graph_roman += pynutil.insert(" morphosyntactic_features: \"gender_masc\"") | pynutil.insert(
                " morphosyntactic_features: \"gender_fem\""
            )

        graph = (
            pynutil.insert("integer: \"")
            + ordinal_graph
            + pynutil.insert("\"")
            + pynutil.insert(" morphosyntactic_features: \"")
            + convert_gender_suffix
            + pynutil.insert("\"")
        )
        graph = pynini.union(graph, graph_roman)

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
