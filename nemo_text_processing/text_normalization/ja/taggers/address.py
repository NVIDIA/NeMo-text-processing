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

from nemo_text_processing.text_normalization.ja.graph_utils import NEMO_DIGIT, NEMO_NOT_SPACE, GraphFst
from nemo_text_processing.text_normalization.ja.utils import get_abs_path


class AddressFst(GraphFst):
    """
    Finite state transducer for classifying Japanese address-like expressions.

    Examples:
        東京都千代田区丸の内1-1-1 -> name: "東京都千代田区丸の内一の一の一"
        503号室 -> name: "五〇三号室"
        〒100-0001 -> name: "郵便番号一ゼロゼロのゼロゼロゼロ一"
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="address", kind="classify", deterministic=deterministic)

        digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        zero_maru = pynini.string_file(get_abs_path("data/numbers/zero_maru.tsv"))

        address_number = cardinal.just_cardinals
        digit_for_room = digit | zero_maru
        digit_for_postal = digit | zero
        separator = pynini.string_file(get_abs_path("data/address/separator.tsv"))
        separator_input = pynini.project(separator, "input")
        postal = pynini.string_file(get_abs_path("data/address/postal.tsv"))
        room_suffix = pynini.string_file(get_abs_path("data/address/room_suffix.tsv"))

        hyphen_to_no = (
            pynini.closure(pynutil.delete(" "), 0, 1) + separator + pynini.closure(pynutil.delete(" "), 0, 1)
        )

        address_chain = address_number + hyphen_to_no + address_number + hyphen_to_no + address_number

        address_prefix_char = pynini.difference(
            NEMO_NOT_SPACE,
            NEMO_DIGIT | separator_input,
        )
        address_with_prefix = pynini.closure(address_prefix_char, 1) + address_chain

        postal_code = postal + digit_for_postal**3 + hyphen_to_no + digit_for_postal**4

        room = (NEMO_DIGIT**3 @ (digit_for_room**3)) + room_suffix

        graph = address_with_prefix | postal_code | room
        self.fst = (pynutil.insert('name: "') + graph + pynutil.insert('"')).optimize()
