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

from nemo_text_processing.text_normalization.hi.graph_utils import GraphFst, NEMO_DIGIT, NEMO_HI_DIGIT, insert_space
from nemo_text_processing.text_normalization.hi.utils import get_abs_path

class AddressFst(GraphFst):
    """
    Finite state transducer for tagging address, e.g.
    """

    def __init__(self):
        super().__init__(name="address", kind="classify")
        # Implement address tagging logic here
        single_digit_verbalizer = (
            pynini.string_file(get_abs_path("data/telephone/number.tsv"))
            | pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
            | pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        )

        single_digit = NEMO_DIGIT | NEMO_HI_DIGIT

        digit_verbalizer = pynini.compose(single_digit, single_digit_verbalizer)

        verbalizer = pynini.closure(digit_verbalizer + insert_space, 1) + digit_verbalizer

        number_part = pynutil.insert('number_part: "') + verbalizer + pynutil.insert('"')

        hyphen_graph = pynini.cross("-", " ")

        slash_graph = pynini.cross("/", " बटा ")

        final_graph = number_part | hyphen_graph | slash_graph

        self.fst = self.add_tokens(final_graph)
