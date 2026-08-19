# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.vi.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space, insert_space


class DateFst(GraphFst):
    """
    Finite state transducer for verbalizing Vietnamese dates, e.g.
        date { day: "mười lăm" month: "một" year: "hai nghìn hai mươi tư" }
        -> ngày mười lăm tháng một năm hai nghìn hai mươi tư

        date { month: "tư" year: "hai nghìn hai mươi tư" }
        -> tháng tư năm hai nghìn hai mươi tư

        date { year: "hai mươi" era: "sau công nguyên" }
        -> năm hai mươi sau công nguyên

        date { ordinal: "mười" era: "trước công nguyên" }
        -> năm thứ mười trước công nguyên
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="date", kind="verbalize", deterministic=deterministic)

        quoted_content = pynini.closure(NEMO_NOT_QUOTE)

        day_expr = pynutil.delete("day: \"") + quoted_content + pynutil.delete("\"")
        day_with_prefix = pynutil.insert("ngày ") + day_expr

        month_expr = pynutil.delete("month: \"") + quoted_content + pynutil.delete("\"")
        month_with_prefix = pynutil.insert("tháng ") + month_expr

        year_expr = pynutil.delete("year: \"") + quoted_content + pynutil.delete("\"")
        year_with_prefix = pynutil.insert("năm ") + year_expr

        era_expr = pynutil.delete("era: \"") + quoted_content + pynutil.delete("\"")

        ordinal_expr = pynutil.delete("ordinal: \"") + quoted_content + pynutil.delete("\"")
        ordinal_with_prefix = pynutil.insert("năm thứ ") + ordinal_expr

        date_graph = pynini.union(
            day_with_prefix
            + delete_space
            + insert_space
            + month_with_prefix
            + delete_space
            + insert_space
            + year_with_prefix,
            month_with_prefix + delete_space + insert_space + year_with_prefix,
            day_with_prefix + delete_space + insert_space + month_with_prefix,
            year_with_prefix,
            year_with_prefix + delete_space + insert_space + era_expr,
            ordinal_with_prefix + delete_space + insert_space + era_expr,
        )

        self.fst = self.delete_tokens(date_graph).optimize()
