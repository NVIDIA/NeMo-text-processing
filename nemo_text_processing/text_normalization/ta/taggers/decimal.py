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

from typing import Tuple

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import GraphFst, insert_space
from nemo_text_processing.text_normalization.en.utils import load_labels
from nemo_text_processing.text_normalization.ta.graph_utils import POINT_WORD, rank
from nemo_text_processing.text_normalization.ta.utils import get_abs_path


def quantity_words() -> Tuple['pynini.FstLike', 'pynini.FstLike', 'pynini.FstLike']:
    """
    The written scale ("quantity") words a number may carry, from ``data/numbers/quantity_words.tsv``
    (written, spoken, kind), grouped by how they attach to the number.

    Returns:
        spaced: native and English words that follow the number after a space (கோடி, lakh)
        short: shorthands that may be glued to the number (L, cr, K, M)
        native: native words only, for a second stacked scale word (₹1 லட்சம் கோடி)
    """
    by_kind = {"native": [], "english": [], "short": []}
    for written, spoken, kind, *_ in load_labels(get_abs_path("data/numbers/quantity_words.tsv")):
        by_kind[kind].append((written, spoken))
    native = pynini.string_map(by_kind["native"]).optimize()
    spaced = pynini.union(native, pynini.string_map(by_kind["english"])).optimize()
    short = pynini.string_map(by_kind["short"]).optimize()
    return spaced, short, native


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimals, e.g.
        -12.5006 கோடி -> decimal { negative: "true" integer_part: "பன்னிரண்டு" fractional_part: "ஐந்து பூஜ்யம் பூஜ்யம் ஆறு" quantity: "கோடி" }
        +5.5 -> decimal { positive: "true" integer_part: "ஐந்து" fractional_part: "ஐந்து" }
        1 கோடி -> decimal { integer_part: "ஒன்று" quantity: "கோடி" }
        .5 -> decimal { integer_part: "பூஜ்யம்" fractional_part: "ஐந்து" }
        1.2.3 -> decimal { integer_part: "ஒன்று" fractional_part: "இரண்டு புள்ளி மூன்று" }

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        graph_digit = cardinal.digit | cardinal.zero
        cardinal_graph = cardinal.final_graph

        # Digits in either script read one at a time (the fractional reading).
        self.graph = (graph_digit + pynini.closure(insert_space + graph_digit)).optimize()

        point = pynutil.delete(".")

        optional_sign = pynini.closure(
            (
                pynutil.insert("negative: ") + pynini.cross("-", "\"true\"")
                | pynutil.insert("positive: ") + pynini.cross("+", "\"true\"")
            )
            + insert_space,
            0,
            1,
        )

        self.graph_fractional = (
            pynutil.insert("fractional_part: \"")
            + (self.graph | cardinal.attach_case_suffix(self.graph) + rank(0.1))
            + pynutil.insert("\"")
        )
        self.graph_integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")

        final_graph_wo_sign = self.graph_integer + point + insert_space + self.graph_fractional

        # Bare-dot decimals: .5 reads as <zero> <point> <five>.
        bare_dot = (
            pynutil.insert(f"integer_part: \"{cardinal.zero_word}\"") + point + insert_space + self.graph_fractional
        )
        # Dotted chains (versions, IPs): every segment after the first reads digit-by-digit with
        # the point word between them.
        dotted_chain = (
            self.graph_integer
            + point
            + insert_space
            + pynutil.insert("fractional_part: \"")
            + self.graph
            + pynini.closure(pynini.cross(".", f" {POINT_WORD} ") + self.graph, 1)
            + pynutil.insert("\"")
        )
        final_graph_wo_sign |= pynutil.add_weight(bare_dot, 0.1)
        final_graph_wo_sign |= pynutil.add_weight(dotted_chain, 0.5)

        # A cardinal or decimal followed by a quantity word (5 லட்சம், 1.5 கோடி, 2 lakh).
        spaced, _, _ = quantity_words()
        quantity = pynutil.delete(" ") + insert_space + pynutil.insert("quantity: \"") + spaced + pynutil.insert("\"")
        with_quantity = self.graph_integer + quantity
        with_quantity |= final_graph_wo_sign + quantity

        self.final_graph_wo_negative = final_graph_wo_sign | with_quantity
        final_graph = optional_sign + self.final_graph_wo_negative
        self.fst = self.add_tokens(final_graph).optimize()
