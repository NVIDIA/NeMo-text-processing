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

from nemo_text_processing.inverse_text_normalization.ta.graph_utils import GraphFst
from nemo_text_processing.inverse_text_normalization.ta.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, delete_space, insert_space
from nemo_text_processing.text_normalization.ta.graph_utils import NEMO_TA_LETTER
from nemo_text_processing.text_normalization.ta.utils import get_abs_path as tn_abs_path


class DateFst(GraphFst):
    """
    Finite state transducer for classifying spoken dates, e.g.
        பதினைந்து ஜூன் இரண்டாயிரத்து இருபத்துநான்கு -> date { day: "15" month: "ஜூன்" year: "2024" preserve_order: true }
        இரண்டாயிரத்து இருபத்துநான்கு ஜூன் பதினைந்து -> date { year: "2024" month: "ஜூன்" day: "15" preserve_order: true }

    The month names are the spoken side of the TN months table, so the two directions share
    one list.

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: CardinalFst):
        super().__init__(name="date", kind="classify")

        month_names = pynini.project(pynini.string_file(tn_abs_path("data/date/months.tsv")), "output").optimize()

        # Days are 1-31 and years four digits, so இரண்டாயிரத்து இருபத்துநான்கு is never a day.
        valid_day = pynini.union(*[str(n) for n in range(1, 32)]).optimize()
        four_digits = NEMO_DIGIT**4 + pynini.closure(NEMO_TA_LETTER)
        day = pynutil.insert("day: \"") + (cardinal.words_to_digits @ valid_day) + pynutil.insert("\"")
        month = pynutil.insert("month: \"") + month_names + pynutil.insert("\"")
        # A case suffix on the year travels into the written form (... 2024ல்).
        year_words = pynini.union(cardinal.words_to_digits, pynutil.add_weight(cardinal.words_to_digits_suffixed, 0.1))
        year = pynutil.insert("year: \"") + (year_words @ four_digits) + pynutil.insert("\"")
        sep = delete_space + insert_space

        graph_dmy = day + sep + month + pynini.closure(sep + year, 0, 1)
        graph_my = month + sep + year
        graph_ymd = year + sep + month + sep + day
        graph_mdy = month + sep + day + pynini.closure(sep + year, 0, 1)

        graph = (graph_dmy | graph_my | graph_ymd | graph_mdy) + pynutil.insert(" preserve_order: true")
        self.fst = self.add_tokens(graph).optimize()
