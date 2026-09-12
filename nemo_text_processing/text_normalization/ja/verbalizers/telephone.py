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

from nemo_text_processing.text_normalization.ja.graph_utils import (
    NEMO_NARROW_NON_BREAK_SPACE,
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_preserve_order,
    delete_space,
)
from nemo_text_processing.text_normalization.ja.utils import get_abs_path


class TelephoneFst(GraphFst):
    """
    Finite state transducer for verbalizing Japanese telephone numbers.

    Example:
        telephone { number_part: "ゼロ九ゼロ 一二三四 五六七八" preserve_order: true }
        -> ゼロ九ゼロ、 一二三四、 五六七八
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="telephone", kind="verbalize", deterministic=deterministic)

        country_code_prefix = pynini.string_file(get_abs_path("data/telephone/country_code_prefix.tsv"))
        group_separator = pynini.string_file(get_abs_path("data/telephone/group_separator.tsv"))
        extension_cue = pynini.project(
            pynini.string_file(get_abs_path("data/telephone/extension.tsv")),
            "output",
        )
        country_code = (
            pynutil.delete('country_code: "')
            + pynutil.insert(country_code_prefix)
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        protected_space = pynutil.insert(NEMO_NARROW_NON_BREAK_SPACE)
        spoken_group_separator = pynutil.insert(group_separator) + protected_space
        number_group = pynini.closure(
            (NEMO_NOT_QUOTE - " ") | (pynutil.delete(" ") + spoken_group_separator),
            1,
        )
        number_part = pynutil.delete('number_part: "') + number_group + pynutil.delete('"')

        extension = (
            pynutil.delete('extension: "')
            + pynutil.insert(extension_cue)
            + protected_space
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        optional_extension = pynini.closure(delete_space + spoken_group_separator + extension, 0, 1)

        graph = (
            ((country_code + delete_space + spoken_group_separator + number_part) | number_part)
            + optional_extension
            + delete_preserve_order
        )

        self.fst = self.delete_tokens(graph).optimize()
