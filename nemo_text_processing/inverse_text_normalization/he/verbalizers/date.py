# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.he.graph_utils import (
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_space,
    delete_zero_or_one_space,
    insert_space,
)


class DateFst(GraphFst):
    """
    Finite state transducer for verbalizing date,
        e.g. { day_prefix: "ה" day: "1" month_prefix: "ב" month: "6" year: "2012" } -> ה-1.6.2012
    """

    def __init__(self):
        super().__init__(name="date", kind="verbalize")

        day_prefix = (
                pynutil.delete("day_prefix:")
                + delete_space
                + pynutil.delete("\"")
                + pynini.closure(NEMO_NOT_QUOTE, 1)
                + pynutil.insert('-')
                + pynutil.delete("\"")
        )

        day = (
            pynutil.delete("day:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1, 2)
            + pynutil.insert('.')
            + pynutil.delete("\"")
            + delete_space
        )

        month_prefix_ignore = (
                pynutil.delete("month_prefix:")
                + delete_space
                + pynutil.delete("\"")
                + pynutil.delete(NEMO_NOT_QUOTE, 1)
                + pynutil.delete("\"")
                + delete_space
        )

        month_prefix_keep = (
                pynutil.delete("month_prefix:")
                + delete_space
                + pynutil.delete("\"")
                + pynini.closure(NEMO_NOT_QUOTE, 1)
                + pynutil.delete("\"")
                + delete_space
        )

        month = (
            pynutil.delete("month:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        year_only_prefix = (
                pynutil.delete("year_only_prefix:")
                + delete_space
                + pynutil.delete("\"")
                + pynini.closure(NEMO_NOT_QUOTE, 3)
                + pynutil.delete("\"")
                + delete_space
        )

        year = (
            pynutil.delete("year:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        #######################
        # DATE FORMATS GRAPHS #
        #######################

        # day and month only
        graph_dm = (
            pynini.closure(day_prefix + delete_zero_or_one_space, 0, 1) +
            day +
            pynini.closure(month_prefix_ignore + delete_zero_or_one_space, 0, 1) +
            month +
            delete_zero_or_one_space
        )

        # day month and year
        graph_dmy = (
            graph_dm +
            delete_space +
            pynutil.insert('.') +
            pynini.closure(delete_zero_or_one_space + year, 0, 1)
        )

        # only month and year
        graph_my = (
            pynini.closure(month_prefix_keep + delete_zero_or_one_space, 0, 1) +
            month +
            pynutil.insert(' ') +
            pynini.closure(delete_zero_or_one_space + year, 0, 1)
        )

        # only year
        graph_y_only = year_only_prefix + insert_space + year

        final_graph = (graph_dm | graph_dmy | graph_my | graph_y_only) + delete_space

        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()
