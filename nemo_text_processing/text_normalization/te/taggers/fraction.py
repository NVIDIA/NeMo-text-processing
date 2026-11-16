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

from nemo_text_processing.text_normalization.te.graph_utils import (
    NEMO_ALL_DIGIT,
    NEMO_CHAR,
    NEMO_DIGIT,
    NEMO_SPACE,
    GraphFst,
)
from nemo_text_processing.text_normalization.te.utils import get_abs_path


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying Telugu fractions.

    "౨౩ ౪/౬" ->
    fraction { integer_part: "ఇరవై మూడు" numerator: "నాలుగు" denominator: "ఆరు" }
    """

    def __init__(self, cardinal, deterministic: bool = True):
        super().__init__(name="fraction", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.final_graph
        slash = pynini.union("/", NEMO_SPACE + "/" + NEMO_SPACE)
        morph_pre = pynutil.insert("morphosyntactic_features: \"")
        morph_suf = pynutil.insert("\"" + NEMO_SPACE)

        te_digit = pynini.difference(NEMO_ALL_DIGIT, NEMO_DIGIT).optimize()
        has_ascii = pynini.closure(NEMO_CHAR) + NEMO_DIGIT + pynini.closure(NEMO_CHAR)
        has_te = pynini.closure(NEMO_CHAR) + te_digit + pynini.closure(NEMO_CHAR)

        def no_te(fst):
            return pynini.difference(fst, has_te).optimize()

        def no_ascii(fst):
            return pynini.difference(fst, has_ascii).optimize()

        optional_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\"") + pynutil.insert(NEMO_SPACE),
            0,
            1,
        )

        def signed(graph):
            return optional_negative + graph

        num_den = (
            pynutil.insert("numerator: \"")
            + cardinal_graph
            + pynini.cross(slash, "\"")
            + pynutil.insert(NEMO_SPACE)
            + pynutil.insert("denominator: \"")
            + cardinal_graph
            + pynutil.insert("\"")
        )

        common = pynini.string_file(get_abs_path("data/fraction/common.tsv"))
        int_in = pynini.project(cardinal_graph, "input").optimize()
        common_in = pynini.project(common, "input").optimize()
        tail_in = pynini.project(cardinal_graph + slash + cardinal_graph, "input").optimize()
        same_script = pynini.union(no_te(tail_in), no_ascii(tail_in)).optimize()
        rational_tail = pynini.difference(same_script, common_in).optimize()
        cross_tail = pynini.difference(pynini.difference(tail_in, common_in), same_script).optimize()

        def mixed(int_script, com_script):
            return (
                pynutil.insert("integer_part: \"")
                + pynini.compose(int_script, cardinal_graph)
                + pynini.cross(NEMO_SPACE, "\" ")
                + morph_pre
                + pynini.compose(com_script, common)
                + morph_suf
            )

        morph = morph_pre + common + morph_suf
        optional_int = pynini.closure(
            pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"") + pynini.accep(NEMO_SPACE),
            0,
            1,
        )

        self.fst = self.add_tokens(
            signed(morph | mixed(no_ascii(int_in), no_ascii(common_in)) | mixed(no_te(int_in), no_te(common_in)))
            | signed(optional_int + pynini.compose(rational_tail, num_den))
            | signed(pynini.compose(cross_tail, pynutil.insert('morphosyntactic_features: "slash" ') + num_den))
        ).optimize()
