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
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_preserve_order,
    delete_space,
)


class TelephoneFst(GraphFst):
    """
    Finite state transducer for verbalizing Japanese telephone numbers.

    Example:
        telephone { number_part: "ゼロ九ゼロ 一二三四 五六七八" preserve_order: true }
        -> ゼロ九ゼロ、 一二三四、 五六七八
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="telephone", kind="verbalize", deterministic=deterministic)

        country_code = (
            pynutil.delete('country_code: "')
            + pynutil.insert("プラス")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        group_separator = pynutil.insert("、 ")
        number_group = pynini.closure((NEMO_NOT_QUOTE - " ") | pynini.cross(" ", "、 "), 1)
        number_part = pynutil.delete('number_part: "') + number_group + pynutil.delete('"')

        extension = (
            pynutil.delete('extension: "')
            + pynutil.insert("内線 ")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        optional_extension = pynini.closure(delete_space + group_separator + extension, 0, 1)

        graph = (
            ((country_code + delete_space + group_separator + number_part) | number_part)
            + optional_extension
            + delete_preserve_order
        )

        self.fst = self.delete_tokens(graph).optimize()
