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
        zero_maru = pynini.cross("0", "〇")

        address_number = cardinal.just_cardinals
        digit_for_room = digit | zero_maru
        digit_for_postal = digit | zero

        hyphen_to_no = (
            pynini.closure(pynutil.delete(" "), 0, 1)
            + (pynutil.delete("-") | pynutil.delete("－") | pynutil.delete("ー"))
            + pynini.closure(pynutil.delete(" "), 0, 1)
            + pynutil.insert("の")
        )

        address_chain = address_number + hyphen_to_no + address_number + hyphen_to_no + address_number

        address_prefix_char = pynini.difference(
            NEMO_NOT_SPACE,
            NEMO_DIGIT | pynini.union("-", "－", "ー"),
        )
        address_with_prefix = pynini.closure(address_prefix_char, 1) + address_chain

        postal_code = (
            pynutil.delete("〒")
            + pynutil.insert("郵便番号")
            + digit_for_postal**3
            + hyphen_to_no
            + digit_for_postal**4
        )

        room = (NEMO_DIGIT**3 @ (digit_for_room**3)) + pynini.accep("号室")

        graph = address_with_prefix | postal_code | room
        self.fst = (pynutil.insert('name: "') + graph + pynutil.insert('"')).optimize()
