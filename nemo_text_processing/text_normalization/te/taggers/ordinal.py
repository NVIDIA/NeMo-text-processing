# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
# See the License for the specific language governing permissions and limitations under the License.

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.te.graph_utils import NEMO_ALL_DIGIT, NEMO_SIGMA, GraphFst
from nemo_text_processing.text_normalization.te.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.te.utils import get_abs_path, load_labels


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying Telugu ordinals, e.g.
        21వ -> ordinal { integer: "ఇరవై ఒకటవ" }

    Args:
        cardinal: CardinalFst used to verbalize the numeric part
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, cardinal: CardinalFst, deterministic: bool = True):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        written_suffix = load_labels(get_abs_path("data/ordinal/suffix.tsv"))[0][0].strip()
        delete_suffix = pynutil.delete(written_suffix)

        digit_map = {row[0]: row[1] for row in load_labels(get_abs_path("data/numbers/digit.tsv"))}
        one_verbal = digit_map["1"]
        digits_as_one = pynini.union(
            *[inp for inp, verbal in digit_map.items() if verbal == one_verbal]
        ).optimize()

        tie_stems = {row[1] for row in load_labels(get_abs_path("data/numbers/ties.tsv"))}
        extra_stems = {row[0].strip() for row in load_labels(get_abs_path("data/ordinal/word_append_stems.tsv"))}
        word_append = pynini.union(
            *(
                pynini.cross(stem, stem + written_suffix)
                for stem in tie_stems | extra_stems
            )
        ).optimize()

        char_rewrite = pynini.string_file(get_abs_path("data/ordinal/char_rewrites.tsv"))
        cardinal_to_ordinal = (NEMO_SIGMA + (word_append | char_rewrite)).optimize()

        sep, two, three = pynutil.delete(","), NEMO_ALL_DIGIT**2, NEMO_ALL_DIGIT**3
        grouped_numbers = pynini.union(
            pynini.closure(NEMO_ALL_DIGIT, 1, 2) + pynini.closure(sep + two) + sep + three,
            pynini.closure(NEMO_ALL_DIGIT, 1, 3) + pynini.closure(sep + three, 1),
        )
        plain_numbers = pynini.difference(pynini.closure(NEMO_ALL_DIGIT, 1), digits_as_one)
        number_input = pynini.union(plain_numbers, grouped_numbers).optimize()

        ordinal_graph = (
            number_input @ cardinal.final_graph @ cardinal_to_ordinal
        ) + delete_suffix
        exceptions = pynini.string_file(get_abs_path("data/ordinal/exceptions.tsv"))
        graph = pynini.union(exceptions, ordinal_graph).optimize()
        self.graph = graph

        final_graph = pynutil.insert('integer: "') + graph + pynutil.insert('"')
        self.fst = self.add_tokens(final_graph).optimize()
