# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, GraphFst
from nemo_text_processing.text_normalization.pl.taggers.cardinal import CASES
from nemo_text_processing.text_normalization.pl.taggers.ordinal import complete_paradigm
from nemo_text_processing.text_normalization.pl.utils import adjective_inflection, get_abs_path, load_labels


class DecimalFst(GraphFst):
    """Classifies Polish decimals with case-inflected fractional readings."""

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        separators = dict(load_labels(get_abs_path("data/decimal/separators.tsv")))
        denominators = {}
        for width, lemma in load_labels(get_abs_path("data/decimal/denominators.tsv")):
            forms = adjective_inflection(lemma)
            complete_paradigm(forms, complete=True)
            denominators[int(width)] = forms

        positive = (NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT)
        integer_input = pynini.union("0", positive)
        point = pynutil.delete(pynini.union(",", "."))
        optional_negative = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", '"true" '), 0, 1)

        self.graphs = {}
        self.digit_graphs = {}
        for case in CASES:
            integer = integer_input @ (cardinal.zero_all[f"sg_{case}"] | cardinal.graphs[f"mi_sg_{case}"])
            integer_field = pynutil.insert('integer_part: "') + integer + pynutil.insert('" ')
            named_fractions = []
            governed_case = "gen" if case in {"nom", "acc", "voc"} else case
            for width, denominator in denominators.items():
                one = pynini.cross(f"{1:0{width}d}", "1")
                fixed_width = NEMO_DIGIT**width
                nonzero = fixed_width - ("0" * width)
                strip_leading_zeros = fixed_width @ (
                    pynini.closure(pynutil.delete("0")) + (NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT)
                )
                few_input = NEMO_DIGIT ** (width - 1) + pynini.union("2", "3", "4")
                if width > 1:
                    few_input -= NEMO_DIGIT ** (width - 2) + pynini.union("12", "13", "14")
                few = (few_input @ strip_leading_zeros).optimize()
                many_input = nonzero - f"{1:0{width}d}" - few_input
                many = (many_input @ strip_leading_zeros).optimize()
                fraction = (one @ cardinal.graphs[f"f_sg_{case}"]) + pynutil.insert(" " + denominator[f"f_sg_{case}"])
                fraction |= (few @ cardinal.graphs[f"f_pl_{case}"]) + pynutil.insert(" " + denominator[f"f_pl_{case}"])
                fraction |= (many @ cardinal.graphs[f"f_pl_{case}"]) + pynutil.insert(
                    " " + denominator[f"f_pl_{governed_case}"]
                )
                named_fractions.append(fraction)

            named = (
                optional_negative
                + integer_field
                + pynutil.insert(f'separator: "{separators["named"]}" fractional_part: "')
                + point
                + pynini.union(*named_fractions)
                + pynutil.insert('"')
            )
            digits = (
                optional_negative
                + integer_field
                + pynutil.insert(f'separator: "{separators["digits"]}" fractional_part: "')
                + point
                + cardinal.single_digits_graph
                + pynutil.insert('"')
            )
            self.graphs[case] = named.optimize()
            self.digit_graphs[case] = digits.optimize()

        graph = self.graphs["nom"]
        if not deterministic:
            graph = pynini.union(*self.graphs.values(), *self.digit_graphs.values()).optimize()
        self.final_graph = graph
        self.fst = self.add_tokens(graph).optimize()
