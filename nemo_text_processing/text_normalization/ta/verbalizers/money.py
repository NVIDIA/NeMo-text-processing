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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, NEMO_SIGMA, NEMO_SPACE, GraphFst
from nemo_text_processing.text_normalization.en.utils import load_labels
from nemo_text_processing.text_normalization.ta.graph_utils import MINUS_WORD, POINT_WORD
from nemo_text_processing.text_normalization.ta.utils import get_abs_path


class MoneyFst(GraphFst):
    """
    Finite state transducer for verbalizing money, e.g.
        money { integer_part: "பன்னிரண்டு" currency_maj: "ரூபாய்" } -> பன்னிரண்டு ரூபாய்
        money { integer_part: "பன்னிரண்டு" currency_maj: "ரூபாய்" fractional_part: "ஐம்பது" currency_min: "centiles" } -> பன்னிரண்டு ரூபாய் ஐம்பது பைசா
        money { currency_maj: "ரூபாய்" integer_part: "பூஜ்யம்" fractional_part: "ஐம்பது" currency_min: "centiles" } -> ஐம்பது பைசா
        money { integer_part: "ஐம்பது" currency_maj: "ரூபாய்" morphosyntactic_features: "ஆக" } -> ஐம்பது ரூபாயாக

    The ``centiles`` placeholder is resolved from ``data/money/major_minor_currencies.tsv``; a
    case suffix in ``morphosyntactic_features`` is joined onto the currency word with sandhi
    (ரூபாய் + ஆக -> ரூபாயாக, ரூபாய் + இல் -> ரூபாயில்).

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="money", kind="verbalize", deterministic=deterministic)

        major_minor_currencies = [
            r for r in load_labels(get_abs_path("data/money/major_minor_currencies.tsv")) if len(r) >= 2
        ]

        optional_suffix = pynini.closure(
            pynutil.delete(" morphosyntactic_features: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\""),
            0,
            1,
        )
        currency_major = (
            pynutil.delete("currency_maj: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
            + optional_suffix
        )

        # A whole-field ஒன்று, or ஒன்று heading a quantity phrase (ஒரு லட்சம்), reads as ஒரு, but not
        # before a decimal point (ஒன்று புள்ளி ஐந்து கோடி).
        not_point = pynini.difference(
            pynini.closure(NEMO_NOT_QUOTE, 1), pynini.accep(POINT_WORD) + pynini.closure(NEMO_NOT_QUOTE)
        )
        one_phrase = (pynini.accep("ஒன்று") + pynini.closure(" " + not_point, 0, 1)).optimize()
        one_as_oru = pynini.cross("ஒன்று", "ஒரு") + pynini.closure(" " + not_point, 0, 1) | pynini.difference(
            pynini.closure(NEMO_NOT_QUOTE, 1), one_phrase
        )
        integer_part = pynutil.delete("integer_part: \"") + one_as_oru + pynutil.delete("\"")
        fractional_part = pynutil.delete("fractional_part: \"") + one_as_oru + pynutil.delete("\"")

        # Major denomination only.
        graph_major_only = integer_part + pynini.accep(NEMO_SPACE) + currency_major

        major_minor_graphs = []
        minor_graphs = []
        for major, minor, *_ in major_minor_currencies:
            graph_major = pynutil.delete("currency_maj: \"") + pynini.accep(major) + pynutil.delete("\"")
            graph_minor = pynutil.delete("currency_min: \"") + pynini.cross("centiles", minor) + pynutil.delete("\"")
            major_minor_graphs.append(
                integer_part
                + pynini.accep(NEMO_SPACE)
                + graph_major
                + pynini.accep(NEMO_SPACE)
                + fractional_part
                + pynini.accep(NEMO_SPACE)
                + graph_minor
            )
            # Minor denomination only: the zero integer part and the major word are silent.
            minor_graphs.append(
                pynutil.delete("integer_part: \"பூஜ்யம்\"")
                + pynutil.delete(NEMO_SPACE)
                + pynutil.delete("currency_maj: \"")
                + pynutil.delete(major)
                + pynutil.delete("\"")
                + pynutil.delete(NEMO_SPACE)
                + fractional_part
                + pynini.accep(NEMO_SPACE)
                + graph_minor
            )

        graph = (
            graph_major_only
            | pynini.union(*major_minor_graphs)
            | pynutil.add_weight(pynini.union(*minor_graphs), -0.1)
        )

        optional_sign = pynini.closure(pynini.cross("negative: \"true\" ", f"{MINUS_WORD} "), 0, 1)
        graph = optional_sign + graph
        suffix_sandhi = pynini.cdrewrite(
            pynini.union(pynini.cross("்ஆ", "ா"), pynini.cross("்இ", "ி")), "", "", NEMO_SIGMA
        )
        graph = graph @ suffix_sandhi

        self.fst = self.delete_tokens(graph).optimize()
