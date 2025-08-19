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
from nemo_text_processing.inverse_text_normalization.ger.graph_utils import (
    NEMO_SIGMA,
    NEMO_SPACE,
    delete_space,
    GraphFst,
)


class OrdinalFst(GraphFst):
    """
    WFST for classifying ordinal numerals:

        einundzwanzigste jahrhundert -> ordinal { integer: "21" morphosyntactic_features: "./jahrhundert" }

    This class handles all declined ordinal numeral forms.
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="ordinal", kind="classify")
        graph_cardinals = cardinal.graph_all_cardinals
        graph_first_dozen = cardinal.dozen

        # WFST for all ordinal suffixes across all cases in singular and plural.
        # The graph accepts the following morphemes: -te, -tem, -ten, -ter, -tes.
        # For cardinal numerals ending in 'g', an 's' is prepended to the suffix ( '' -> 's' / 'g' _ suffix).
        suffix = (
            pynini.accep("s").ques + pynini.accep("te") + pynini.union(*"nmrs").ques
        )

        # These are only applicable to the last digit in the string
        exceptions = pynini.cdrewrite(
            pynini.cross(("ers" + suffix), "eins")
            | pynini.cross(("drit" + suffix), "drei")
            | pynini.cross(("sieb" + suffix), "sieben")
            | pynini.cross(("ach" + suffix), "acht")
            | pynini.cross(("sech" + suffix), "sechs"),
            "",
            "[EOS]",
            NEMO_SIGMA,
        )
        graph_exceptions = NEMO_SIGMA @ exceptions
        # Removes any suffixes left after transcuding the exceptions
        remove_suffixes = graph_exceptions @ pynini.cdrewrite(
            pynini.cross(suffix, "").ques, "", "[EOS]", NEMO_SIGMA
        )

        graph_ordinals_raw = remove_suffixes @ graph_cardinals
        graph_first_dozen_raw = remove_suffixes @ graph_first_dozen

        # Appplies the tag
        graph_ordinals = (
            pynutil.insert('integer: "') + graph_ordinals_raw + pynutil.insert('"')
        )

        # Handles denormalized morphosyntax
        ordinal_ending = pynutil.insert(".")
        morphosyntax = pynutil.insert(' morphosyntactic_features: "') + ordinal_ending

        graph_ordinals += morphosyntax

        # Simplified ordinal graphs to be passed to other semiotic classes
        self.graph_ordinals = graph_ordinals_raw + ordinal_ending
        self.graph_ordinals_first_dozen = graph_first_dozen_raw + ordinal_ending

        # Applies special tokens
        special_tokens = pynini.accep("Jahrhundert") | pynini.accep("Jahrhunderte")

        graph_special_tokens = delete_space + pynutil.insert("/") + special_tokens
        graph_special_tokens = pynini.closure(graph_special_tokens, 0, 1)

        graph_ordinals += graph_special_tokens

        # Indicates the era B.C. / A.D.
        # This subgraph logically follows from the above (Jahrhundert = century)
        vor = pynini.cross("vor", "v.")
        nach = pynini.cross("nach", "n.")
        christ = pynini.cross("Christus", "Ch.")
        BC_AD = (vor | nach) + pynini.accep(NEMO_SPACE) + christ

        graph_ordinals += (pynini.accep(NEMO_SPACE) + BC_AD).ques
        graph_ordinals += pynutil.insert('"')

        # Builds and optimizes the graph
        graph_ordinals = self.add_tokens(graph_ordinals)
        self.fst = graph_ordinals.optimize()
