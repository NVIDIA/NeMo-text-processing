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
from nemo_text_processing.inverse_text_normalization.ger.utils import get_abs_path
from nemo_text_processing.inverse_text_normalization.ger.graph_utils import (
    GraphFst,
    delete_space,
    NEMO_SPACE,
    NEMO_DIGIT,
    NEMO_ALPHA,
)


class CardinalFst(GraphFst):
    """
    Finite state transducer for verbalizing cardinal numbers.  Note that the verbalizer retains period-separated formatting.
        e.g. 'cardinal { negative: "-" integer: "1.234.512.102" }' -> -1.234.512.102
    """

    def __init__(self):
        super().__init__(name="cardinal", kind="verbalize")

        DE_chars = pynini.union(*"äöüÄÖÜß").optimize()

        # removes the 'negative:' label and leaves the optional '-' sign in place
        optional_minus = pynini.closure(
            pynutil.delete("negative:")
            + pynutil.delete(NEMO_SPACE)
            + pynutil.delete('"')
            + pynini.accep("-")
            + pynutil.delete('"')
            + pynutil.delete(NEMO_SPACE),
            0,
            1,
        )

        # handles all elements of a cardinal integer
        integer_chars = NEMO_DIGIT | pynini.accep(".")
        cardinal_components = NEMO_DIGIT | NEMO_ALPHA | DE_chars | pynini.accep(".")

        # removes the 'integer:' label
        just_integers = (
            pynutil.delete("integer:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(integer_chars, 1)
            + pynutil.delete('"')
            + delete_space
        )

        # handles the canonical representation with the first dozen normalized
        first_dozen_verbalized = (
            pynutil.delete("integer:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(cardinal_components, 1)
            + pynutil.delete('"')
            + delete_space
        )

        # Handles noun + number combinations, where the noun forces full denormalization
        # The nouns are implemented as a .tsv list
        nouns_forcing_denormalization = pynini.string_file(
            get_abs_path("data/measure/nouns_forcing_denormalization.tsv")
        )
        graph_forced_denormalization = (
            pynutil.delete("morphosyntactic_features: ")
            + pynutil.delete('"')
            + nouns_forcing_denormalization
            + pynutil.delete('"')
            + pynini.accep(NEMO_SPACE)
            + just_integers
        )

        graph = optional_minus + just_integers
        self.numbers = graph.optimize()
        first_dozen = (optional_minus + first_dozen_verbalized).optimize()
        self.first_dozen = first_dozen
        updated_cardinals = (first_dozen | graph_forced_denormalization).optimize()
        delete_tokens = self.delete_tokens(updated_cardinals)
        self.fst = delete_tokens.optimize()
