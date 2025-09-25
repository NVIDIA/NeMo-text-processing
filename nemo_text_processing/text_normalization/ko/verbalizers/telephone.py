# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-
import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.ko.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space, insert_space


class TelephoneFst(GraphFst):
    """
    telephone { [country_code: "...",] number_part: "..." [extension: "..."] }
      -> [country_code + ' '] + number_part [+ ', 내선 ' + extension]
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="telephone", kind="verbalize", deterministic=deterministic)

        NOT_QUOTE = NEMO_NOT_QUOTE

        # country_code (있으면 끝에 공백 하나)
        country = (
            pynini.closure(delete_space, 0, 1)
            + pynutil.delete('country_code: "')
            + pynini.closure(NOT_QUOTE, 1)
            + pynutil.delete('"')
            + insert_space
        )

        # number_part (필수)
        number = (
            pynini.closure(delete_space, 0, 1)
            + pynutil.delete('number_part: "')
            + pynini.closure(NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        # extension (있을 때만 앞에 ", 내선 " 삽입)
        ext_field = (
            pynini.closure(delete_space, 0, 1)
            + pynutil.delete('extension: "')
            + pynini.closure(NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        extension_opt = pynini.closure(pynutil.insert(", 내선 ") + ext_field, 0, 1)

        # telephone { ... } 래퍼 삭제
        graph = (
            pynutil.delete("telephone")
            + pynini.closure(delete_space, 0, 1)
            + pynutil.delete("{")
            + pynini.closure(country, 0, 1)
            + number
            + extension_opt
            + pynini.closure(delete_space, 0, 1)
            + pynutil.delete("}")
        )

        self.fst = graph.optimize()
