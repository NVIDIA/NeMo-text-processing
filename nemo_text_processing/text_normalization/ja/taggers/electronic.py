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

from nemo_text_processing.text_normalization.ja.graph_utils import (
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_NOT_SPACE,
    NEMO_SIGMA,
    GraphFst,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.ja.utils import get_abs_path


class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying Japanese electronic expressions.

    Examples:
        abc@abc.com -> name: "abc アット abc ドット com"
        https://www.nvidia.com -> name: "https コロン スラッシュ スラッシュ www ドット nvidia ドット com"
        1234-5678-9012-3456 -> name: "一二三四 五六七八 九〇一二 三四五六"
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="electronic", kind="classify", deterministic=deterministic)

        digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        digit_zero_maru = digit | pynini.cross("0", "〇")
        digit_zero_user = digit | pynini.cross("0", "ゼロ")

        special_digit_run = pynutil.add_weight(
            pynini.string_file(get_abs_path("data/electronic/special_digit_runs.tsv")),
            -0.1,
        )
        symbol = pynini.string_file(get_abs_path("data/electronic/symbol.tsv"))
        at = symbol @ pynini.cross("アット", " アット ")
        dot = symbol @ pynini.cross("ドット", " ドット ")
        hyphen = symbol @ pynini.cross("ハイフン", " ハイフン ")
        slash = symbol @ pynini.cross("スラッシュ", " スラッシュ ")
        colon = symbol @ pynini.cross("コロン", " コロン ")

        raw_label = pynini.closure(NEMO_ALPHA | NEMO_DIGIT, 1)
        alpha_label = pynini.closure(NEMO_ALPHA, 1)
        digit_label = pynini.closure(NEMO_DIGIT, 1)
        raw_label_with_hyphen = raw_label + pynini.closure(pynutil.delete("-") + pynutil.insert(" ハイフン ") + raw_label)

        insert_alpha_digit_space = pynini.cdrewrite(pynutil.insert(" "), NEMO_ALPHA, NEMO_DIGIT, NEMO_SIGMA)
        insert_digit_alpha_space = pynini.cdrewrite(pynutil.insert(" "), NEMO_DIGIT, NEMO_ALPHA, NEMO_SIGMA)
        alnum_spacing = insert_alpha_digit_space @ insert_digit_alpha_space
        username_reader = pynini.closure(NEMO_ALPHA | special_digit_run | digit_zero_user | pynini.accep(" "), 1)

        raw_alnum = pynini.closure(NEMO_ALPHA | NEMO_DIGIT, 1)
        raw_mixed_alnum = (
            pynini.closure(NEMO_ALPHA | NEMO_DIGIT)
            + NEMO_ALPHA
            + pynini.closure(NEMO_ALPHA | NEMO_DIGIT)
            + NEMO_DIGIT
            + pynini.closure(NEMO_ALPHA | NEMO_DIGIT)
        ) | (
            pynini.closure(NEMO_ALPHA | NEMO_DIGIT)
            + NEMO_DIGIT
            + pynini.closure(NEMO_ALPHA | NEMO_DIGIT)
            + NEMO_ALPHA
            + pynini.closure(NEMO_ALPHA | NEMO_DIGIT)
        )
        username_alnum = (raw_mixed_alnum @ alnum_spacing @ username_reader).optimize()
        username_segment = pynutil.add_weight(username_alnum, -0.1) | alpha_label | digit_label
        username = username_segment + pynini.closure((dot | hyphen) + username_segment)

        domain = raw_label + pynini.closure(dot + raw_label) + dot + alpha_label
        email = username + at + domain

        protocol = (pynini.accep("https") | pynini.accep("http")) + colon + slash + slash
        path = slash + raw_label_with_hyphen + pynini.closure(slash + raw_label_with_hyphen)
        url = protocol + domain + pynini.closure(path, 0, 1)

        four_digits = NEMO_DIGIT**4 @ (digit_zero_maru**4)
        card_separator = (pynutil.delete("-") | pynutil.delete(" ")) + insert_space
        credit_card = four_digits + card_separator + four_digits + card_separator + four_digits + card_separator + four_digits

        card_cue = pynini.string_file(get_abs_path("data/electronic/card_cues.tsv"))
        card_with_cue = card_cue + credit_card
        card_tail_with_cue = card_cue + four_digits
        card_tail = pynini.accep("カード下") + (NEMO_DIGIT @ cardinal.just_cardinals) + pynini.accep("桁") + (
            NEMO_DIGIT**4 @ (digit_zero_maru**4)
        )

        extension = pynini.string_file(get_abs_path("data/electronic/file_extensions.tsv"))
        filename_stem = pynini.closure(
            pynini.difference(NEMO_NOT_SPACE, pynini.union(".", "/", "@")),
            1,
        )
        filename = filename_stem + insert_space + extension
        email_with_context = (
            pynini.accep("email")
            + delete_space
            + insert_space
            + email
            + delete_space
            + insert_space
            + pynini.accep("です")
        )

        graph = (
            pynutil.add_weight(credit_card, -0.1)
            | pynutil.add_weight(card_with_cue, -0.1)
            | pynutil.add_weight(card_tail_with_cue, -0.1)
            | card_tail
            | email_with_context
            | email
            | url
            | domain
            | filename
        )

        self.fst = (pynutil.insert('name: "') + graph.optimize() + pynutil.insert('"')).optimize()
