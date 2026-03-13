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


from functools import reduce

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.pt.graph_utils import (
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_space,
    insert_space,
    filter_cardinal_punctuation,
)
from nemo_text_processing.text_normalization.pt.utils import get_abs_path, load_labels


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying Portuguese cardinals, e.g.
        "1000" -> cardinal { integer: "mil" }
        "2.000.000" -> cardinal { integer: "dois milhões" }
        "-5" -> cardinal { negative: "true" integer: "cinco" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        specials = {
            row[0]: row[1]
            for row in load_labels(get_abs_path("data/numbers/cardinal_specials.tsv"))
            if len(row) >= 2
        }
        connector_e = insert_space + pynutil.insert(specials["connector"]) + insert_space
        thousand = specials["thousand"]
        hundred_100 = specials["hundred_100"]
        hundred_1 = specials["hundred_1"]

        scale_rows = load_labels(get_abs_path("data/numbers/scales.tsv"))
        scales = [
            (row[0], row[1], int(row[2]))
            for row in scale_rows
            if len(row) >= 3 and row[2].strip().isdigit()
        ]

        _num = lambda p: pynini.string_file(get_abs_path(f"data/numbers/{p}"))
        zero, digit, teens, tens, hundreds = (
            _num("zero.tsv"), _num("digit.tsv"), _num("teens.tsv"), _num("tens.tsv"), _num("hundreds.tsv")
        )
        digits_no_one = (NEMO_DIGIT - "1") @ digit

        graph_tens = teens | (tens + (pynutil.delete("0") | (connector_e + digit)))
        self.tens = graph_tens.optimize()
        self.two_digit_non_zero = pynini.union(
            digit, graph_tens, (pynini.cross("0", NEMO_SPACE) + digit)
        ).optimize()

        graph_hundreds = hundreds + pynini.union(
            pynutil.delete("00"),
            (connector_e + graph_tens),
            (connector_e + digit),
        )
        graph_hundreds |= pynini.cross("100", hundred_100)
        graph_hundreds |= pynini.cross("1", hundred_1) + pynini.union(
            pynutil.delete("00"),
            (connector_e + graph_tens),
            (connector_e + pynutil.delete("0") + digit),
        )
        self.hundreds = graph_hundreds.optimize()

        h_comp_base = pynini.union(graph_hundreds, pynutil.delete("0") + graph_tens)
        h_comp = h_comp_base | (pynutil.delete("00") + digit)
        h_comp_no_one = h_comp_base | (pynutil.delete("00") + digits_no_one)

        pure_tens_input = pynini.union(*[pynini.accep(str(d * 10)) for d in range(1, 10)])
        graph_pure_tens_only = pure_tens_input @ graph_tens
        graph_compound_tens = (pynini.closure(NEMO_DIGIT, 2, 2) - pure_tens_input) @ graph_tens

        graph_pure_components = pynini.union(
            pynutil.delete("0") + graph_pure_tens_only,
            pynutil.delete("00") + digit,
            hundreds + pynutil.delete("00"),
            pynini.cross("100", hundred_100),
        )
        graph_compound_hundreds = pynini.union(
            pynini.cross("1", hundred_1)
            + pynini.union(
                (connector_e + graph_tens),
                (connector_e + pynutil.delete("0") + digit),
            ),
            hundreds + pynini.union(
                (connector_e + graph_tens),
                (connector_e + digit),
            ),
        )

        suffix_after_mil = pynini.union(
            pynutil.delete("000"),
            (connector_e + graph_pure_components),
            (insert_space + graph_compound_hundreds),
            (insert_space + pynutil.delete("0") + graph_compound_tens),
        )

        t_comp = pynini.union(
            pynutil.delete("000") + h_comp,
            h_comp_no_one + insert_space + pynutil.insert(thousand) + suffix_after_mil,
            pynini.cross("001", thousand) + suffix_after_mil,
        )
        t_comp_no_one = pynini.union(
            pynutil.delete("000") + h_comp_no_one,
            h_comp_no_one + insert_space + pynutil.insert(thousand)
            + ((insert_space + h_comp) | pynutil.delete("000")),
            pynini.cross("001", thousand) + ((insert_space + h_comp) | pynutil.delete("000")),
        )

        graph_large_scales = pynini.accep("")
        for one_label, plural_suffix, _ in reversed(scales):
            g = pynutil.add_weight(pynini.cross("000001", one_label), -0.001)
            g |= t_comp_no_one + pynutil.insert(plural_suffix)
            g |= pynutil.delete("000000")
            g += insert_space
            graph_large_scales += g

        # 9/12-digit: scale block + trailing (million+thousands, billion+9digits)
        scale_3_mil = self._scale_block_3(scales[0][0], scales[0][1], h_comp_no_one)
        scale_3_bi = self._scale_block_3(scales[1][0], scales[1][1], h_comp_no_one)
        graph_9 = self._build_scale_trailing_graph(scale_3_mil, t_comp, 6, 9)
        graph_12 = self._build_scale_trailing_graph(scale_3_bi, graph_9, 9, 12)
        pure_9, pure_12 = self._pure_inputs(9), self._pure_inputs(12)
        trail_9 = (pure_9 @ graph_9, (NEMO_DIGIT**9 - pure_9) @ graph_9)
        trail_12 = (pure_12 @ graph_12, (NEMO_DIGIT**12 - pure_12) @ graph_12)

        # Units 6 (u6): pure get "e" after scale; compound no "e"
        u6_one = pynini.cross("000001", "1") @ digit
        u6_pure = pynini.union(
            u6_one, pynini.cross("001000", thousand),
            pynini.cross("000010", "10") @ graph_tens, pynini.cross("000100", hundred_100),
            (pynini.cross("010000", "10") @ graph_tens) + insert_space + pynutil.insert(thousand),
            pynini.cross("100000", hundred_100) + insert_space + pynutil.insert(thousand),
        )
        u6_compound = (NEMO_DIGIT**6 - self._pure_inputs(6)) @ t_comp
        u6 = u6_pure | u6_compound
        z18 = pynini.accep("0" * 18)  # 18 zeros: branch no "e"
        smaller_e = (connector_e + u6_pure) | u6_compound | pynutil.delete("0" * 6)
        smaller = u6 | pynutil.delete("0" * 6)
        graph_24 = (
            ((NEMO_DIGIT**18 - z18) + NEMO_DIGIT**6) @ (graph_large_scales + smaller_e)
        ) | ((z18 + NEMO_DIGIT**6) @ (pynutil.delete(z18) + smaller))

        trail_by_z = {9: trail_9, 12: trail_12}
        magnitude_patterns = [
            self._build_magnitude_pattern(
                one_label, plural_suffix, magnitude_zeros, trail_by_z.get(magnitude_zeros),
                connector_e, insert_space, digit, graph_tens, graph_hundreds,
            )
            for one_label, plural_suffix, magnitude_zeros in scales
            if magnitude_zeros > 0
        ]

        pad = (NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT, 0)
        pad = pad @ pynini.cdrewrite(pynini.closure(pynutil.insert("0")), "[BOS]", "", NEMO_SIGMA) @ NEMO_DIGIT**24
        norm = pynini.cdrewrite(delete_space, "[BOS]", "", NEMO_SIGMA) @ pynini.cdrewrite(delete_space, "", "[EOS]", NEMO_SIGMA)
        norm = norm @ pynini.cdrewrite(pynini.cross(pynini.closure(NEMO_WHITE_SPACE, 2), NEMO_SPACE), NEMO_ALPHA, NEMO_ALPHA, NEMO_SIGMA)
        self.graph = reduce(lambda a, b: a | b, magnitude_patterns, pad @ graph_24 @ norm) | zero
        self.graph = filter_cardinal_punctuation(self.graph).optimize()

        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1
        )
        final_graph = (
            optional_minus_graph
            + pynutil.insert("integer: \"")
            + self.graph
            + pynutil.insert("\"")
        )
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

    def _scale_block_3(self, one_label, plural_suffix, component_no_one):
        """001->one_label, 000->'', else component+plural."""
        return pynini.union(
            pynini.cross("001", one_label),
            pynini.cross("000", ""),
            (NEMO_DIGIT**3 - pynini.accep("001") - pynini.accep("000"))
            @ (component_no_one + insert_space + pynutil.insert(plural_suffix)),
        )

    def _build_scale_trailing_graph(self, scale_3, sub_graph, trailing_len, total_len):
        """total_len digits = scale_3 + trailing; no trailing space when trailing all zeros."""
        zt, ztotal = "0" * trailing_len, "0" * total_len
        scale_nonzero = NEMO_DIGIT**3 - pynini.accep("000")
        branches = [
            (pynini.accep("000") + NEMO_DIGIT**trailing_len) @ (pynutil.delete("000") + sub_graph),
            (scale_nonzero + (NEMO_DIGIT**trailing_len - pynini.accep(zt))) @ (scale_3 + insert_space + sub_graph),
            (scale_nonzero + pynini.accep(zt)) @ (scale_3 + pynutil.delete(zt)),
            (pynini.accep("000") + pynini.accep(zt)) @ pynutil.delete(ztotal),
        ]
        return pynini.union(*branches)

    @staticmethod
    def _pure_inputs(num_digits):
        """Inputs 1, 10, 100, ... as num_digits-digit strings."""
        return pynini.union(
            *[pynini.accep(str(10**k).zfill(num_digits)) for k in range(0, num_digits)]
        )

    def _magnitude_graph(
        self, one_word, plural_suffix, zero_count, graph_digit, graph_tens, graph_hundreds,
        connector_e, insert_space, trailing_pair=None,
    ):
        """Round (1–3 digit + scale + zeros); optional trailing (e + pure | space + compound)."""
        zeros = "0" * zero_count
        round_pats = []
        trail_pats = [] if trailing_pair else None
        for L in (1, 2, 3):
            total = zero_count + L
            if L == 1:
                lead = pynini.cross("1", one_word) | (
                    (NEMO_DIGIT - "1") @ graph_digit + pynutil.insert(plural_suffix)
                )
            else:
                lead = (
                    pynini.closure(NEMO_DIGIT, L, L)
                    @ (graph_tens if L == 2 else graph_hundreds)
                    + pynutil.insert(plural_suffix)
                )
            lead_fst = NEMO_DIGIT**L @ lead
            round_pats.append(
                pynini.closure(NEMO_DIGIT, total, total) @ (lead_fst + pynutil.delete(zeros))
            )
            if trailing_pair:
                pure, compound = trailing_pair
                trail_part = (
                    NEMO_DIGIT**zero_count @ (connector_e + pure)
                    | NEMO_DIGIT**zero_count @ (insert_space + compound)
                )
                trail_pats.append(
                    pynini.closure(NEMO_DIGIT, total, total) @ (lead_fst + trail_part)
                )
        graph_round = pynini.union(*round_pats)
        graph_trail = pynini.union(*trail_pats) if trail_pats else None
        return graph_round, graph_trail

    def _build_magnitude_pattern(
        self,
        one_label, plural_suffix, magnitude_zeros,
        trailing_pair,
        connector_e, insert_space,
        graph_digit, graph_tens, graph_hundreds,
    ):
        """Restrict length; round + optional non-zero trailing."""
        restrict = (NEMO_DIGIT - "0") + pynini.closure(
            NEMO_DIGIT, magnitude_zeros, magnitude_zeros + 2
        )
        graph_round, graph_trail = self._magnitude_graph(
            one_label, plural_suffix, magnitude_zeros,
            graph_digit, graph_tens, graph_hundreds,
            connector_e, insert_space, trailing_pair,
        )
        if graph_trail is None:
            return pynutil.add_weight(restrict @ graph_round, -1.0)
        non_zero_trail = pynini.union(
            *[
                NEMO_DIGIT**n + (NEMO_DIGIT**magnitude_zeros - pynini.accep("0" * magnitude_zeros))
                for n in (1, 2, 3)
            ]
        )
        return pynutil.add_weight(restrict @ (graph_round | (non_zero_trail @ graph_trail)), -1.0)
