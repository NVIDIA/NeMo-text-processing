# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.hi.graph_utils import (
    GraphFst,
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_HI_DIGIT,
    NEMO_SPACE,
    insert_space,
)
from nemo_text_processing.text_normalization.hi.utils import get_abs_path

address_context = pynini.string_file(get_abs_path("data/address/address_context.tsv"))

def get_context(keywords: 'pynini.FstLike'):
    return (keywords + pynini.closure(pynini.accep(NEMO_SPACE), 0, 1)).optimize()


class AddressFst(GraphFst):
    """
    Finite state transducer for tagging address, e.g.
    """

    def __init__(self):
        super().__init__(name="address", kind="classify")
        single_digit_verbalizer = (
            pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
            | pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        )

        single_digit = NEMO_DIGIT | NEMO_HI_DIGIT

        digit_verbalizer = pynini.compose(single_digit, single_digit_verbalizer)

        char_verbalizer = digit_verbalizer | NEMO_ALPHA
        verbalizer = pynini.closure(char_verbalizer + insert_space, 1) + char_verbalizer

        address_lexicon = pynini.string_file(get_abs_path("data/address/lexicon.tsv"))
        address_suffix = pynini.string_file(get_abs_path("data/address/suffix.tsv"))

        lexicon_graph = pynini.closure(NEMO_ALPHA, 1) + address_suffix
        place_name_graph = pynini.union(address_lexicon, pynutil.add_weight(lexicon_graph, 0.1))

        context_before = get_context(address_context)

        number_part = (
            pynutil.insert('number_part: "')
            + pynini.closure(context_before, 0, 1)
            + verbalizer
            + pynutil.insert('"')
        )

        number_with_place_name = (
            pynutil.insert('number_part: "')
            + verbalizer
            + insert_space
            + place_name_graph
            + pynutil.insert('"')
        )

        place_name_with_number = (
            pynutil.insert('number_part: "')
            + place_name_graph
            + insert_space
            + verbalizer
            + pynutil.insert('"')
        )

        hyphen_graph = pynini.cross("-", " ")

        slash_graph = (
            pynutil.insert('number_part: "')
            + digit_verbalizer
            + insert_space
            + pynini.cross("/", "बटा")
            + insert_space
            + digit_verbalizer
            + pynutil.insert('"')
        )

        final_graph = number_part | hyphen_graph | slash_graph | number_with_place_name | place_name_with_number

        final_graph = pynutil.add_weight(final_graph, -0.1)
        self.fst = self.add_tokens(final_graph)