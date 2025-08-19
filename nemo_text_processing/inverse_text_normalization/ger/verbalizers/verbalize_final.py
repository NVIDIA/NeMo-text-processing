# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.ger.verbalizers.verbalize import (
    VerbalizeFst,
)
from nemo_text_processing.inverse_text_normalization.ger.utils import get_abs_path
from nemo_text_processing.inverse_text_normalization.ger.verbalizers.word import WordFst
from nemo_text_processing.inverse_text_normalization.ger.graph_utils import (
    GraphFst,
    delete_extra_space,
    delete_space,
    NEMO_SPACE,
    NEMO_DIGIT,
    NEMO_SIGMA,
)


class VerbalizeFinalFst(GraphFst):
    """
    Finite state transducer that verbalizes an entire sentence, e.g.
    tokens { name: "Es" } tokens { name: "ist" } tokens { name: "jetzt" } tokens { time { hours: "12" minutes: "30" } }  -> Es ist jetzt 12:30
    """

    def __init__(self):
        super().__init__(name="verbalize_final", kind="verbalize")
        verbalize = VerbalizeFst().fst
        word = WordFst().fst
        types = verbalize | word
        graph = (
            pynutil.delete("tokens")
            + delete_space
            + pynutil.delete("{")
            + delete_space
            + types
            + delete_space
            + pynutil.delete("}")
        )
        graph = (
            delete_space
            + pynini.closure(graph + delete_extra_space)
            + graph
            + delete_space
        )

        # The subgraphs below handle post-processing cases
        # Do not remove.

        ### MONEY ###

        # Removes the stray "1" between a decimal number and the magnitudes abbreviation in the MONEY graph
        get_powers_of_ten = pynini.string_file(
            get_abs_path("data/money/magnitudes.tsv")
        )
        remove_1 = pynini.cdrewrite(
            pynini.cross(" 1 ", NEMO_SPACE),
            NEMO_DIGIT,
            pynini.project(get_powers_of_ten, "output"),
            NEMO_SIGMA,
        )
        graph = graph @ remove_1
        graph = graph.optimize()

        # Formats the post-separator decimal strings
        get_currencies_major = pynini.string_file(
            get_abs_path("data/money/currency_major.tsv")
        )
        get_currencies_minor = pynini.string_file(
            get_abs_path("data/money/currency_minor.tsv")
        )

        currencies_major = pynini.project(get_currencies_major, "output")
        currencies_minor = pynini.project(get_currencies_minor, "output")
        currencies = currencies_major | currencies_minor

        format_decimal = pynini.cdrewrite(
            pynutil.delete(NEMO_SPACE),
            (pynini.accep(",") + pynini.closure(NEMO_DIGIT, 1)),
            (pynini.closure(NEMO_DIGIT, 1) + pynini.accep(NEMO_SPACE) + currencies),
            NEMO_SIGMA,
        )
        graph = graph @ format_decimal
        graph = graph.optimize()

        ### MEASURE ###

        # Reformats units
        reformat_units = pynini.string_map(
            [(" ° Celcius", "°C"), (" ° Fahrenheit", "°F"), (' "', '"'), (" °", "°")]
        )
        reformats_units = pynini.cdrewrite(reformat_units, "", "", NEMO_SIGMA)
        graph = graph @ reformats_units
        graph = graph.optimize()

        self.fst = graph
