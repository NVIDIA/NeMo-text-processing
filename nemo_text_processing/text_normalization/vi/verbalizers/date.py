# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space


class DateFst(GraphFst):
    """
    Finite state transducer for verbalizing Vietnamese dates, e.g.
        date { day: "mười lăm" month: "một" year: "hai nghìn hai mươi tư" }
        -> ngày mười lăm tháng một năm hai nghìn hai mươi tư

        date { month: "tư" year: "hai nghìn hai mươi tư" }
        -> tháng tư năm hai nghìn hai mươi tư
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="date", kind="verbalize", deterministic=deterministic)

        quoted_content = pynini.closure(NEMO_NOT_QUOTE)
        day = pynutil.delete("day:") + delete_space + pynutil.delete("\"") + quoted_content + pynutil.delete("\"")
        month = pynutil.delete("month:") + delete_space + pynutil.delete("\"") + quoted_content + pynutil.delete("\"")
        year = pynutil.delete("year:") + delete_space + pynutil.delete("\"") + quoted_content + pynutil.delete("\"")

        insert_day = pynutil.insert("ngày ")
        insert_month = pynutil.insert("tháng ")
        insert_year = pynutil.insert("năm ")
        insert_space = pynutil.insert(" ")

        date_graph = pynini.union(
            insert_day
            + day
            + delete_space
            + insert_space
            + insert_month
            + month
            + delete_space
            + insert_space
            + insert_year
            + year,
            insert_month + month + delete_space + insert_space + insert_year + year,
            insert_day + day + delete_space + insert_space + insert_month + month,
            insert_year + year,
        )

        self.fst = self.delete_tokens(date_graph).optimize()
