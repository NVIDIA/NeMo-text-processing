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

from nemo_text_processing.text_normalization.vi.graph_utils import (
    NEMO_DIGIT,
    NEMO_SPACE,
    GraphFst,
    convert_space,
    insert_space,
)
from nemo_text_processing.text_normalization.vi.utils import get_abs_path


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time in Vietnamese.

    Supports various formats including:
    - Digital formats: "8:30", "14:45", "5:20:35"
    - Vietnamese formats: "14 giờ 30 phút", "2 giờ 15 phút 10 giây"
    - Abbreviated formats: "9h", "9g", "14h30", "14g30", "3p20s"
    - With time zones: "8:23 gmt", "15h cst"

    Args:
        cardinal: CardinalFst for number conversion
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="time", kind="classify", deterministic=deterministic)

        time_zone = pynini.string_file(get_abs_path("data/time/time_zones.tsv"))
        delete_leading_zero = (pynutil.delete("0").ques | (NEMO_DIGIT - "0")) + NEMO_DIGIT
        cardinal_graph = cardinal.graph

        hours = pynini.union(*[str(x) for x in range(0, 25)])
        minutes_seconds = pynini.union(*[str(x) for x in range(0, 60)])

        def label(name, graph):
            return pynutil.insert(f'{name}: "') + graph + pynutil.insert('"')

        hour = label('hours', delete_leading_zero @ hours @ cardinal_graph)
        minute = label('minutes', delete_leading_zero @ minutes_seconds @ cardinal_graph)
        second = label('seconds', delete_leading_zero @ minutes_seconds @ cardinal_graph)
        zone = label('zone', convert_space(time_zone))

        h_suffix = pynini.union(pynutil.delete("h"), pynutil.delete("g"))
        h_word = pynutil.delete(" giờ")
        m_word = pynutil.delete(" phút")
        s_word = pynutil.delete(" giây")

        opt_zone_space = pynini.closure(pynini.accep(NEMO_SPACE) + zone, 0, 1)
        opt_zone = pynini.closure(zone, 0, 1)
        preserve = pynutil.insert(" preserve_order: true")

        # Define sub-patterns for better readability
        # Digital formats
        pattern_hour_minute = hour + pynutil.delete(":") + insert_space + minute + opt_zone_space

        pattern_hour_minute_second = (
            hour
            + pynutil.delete(":")
            + insert_space
            + minute
            + pynutil.delete(":")
            + insert_space
            + second
            + opt_zone_space
            + preserve
        )

        # Abbreviated formats
        pattern_hour_suffix = hour + h_suffix + opt_zone_space
        pattern_hour_suffix_minute = hour + h_suffix + minute + opt_zone
        pattern_minute_p = minute + pynutil.delete("p")
        pattern_second_s = second + pynutil.delete("s")
        pattern_minute_p_second_s = minute + pynutil.delete("p") + insert_space + second + pynutil.delete("s")

        # Vietnamese word formats
        pattern_hour_word = hour + h_word + opt_zone_space

        pattern_hour_word_minute = hour + h_word + pynutil.delete(NEMO_SPACE) + minute + m_word + opt_zone_space

        pattern_hour_word_minute_second = (
            hour
            + h_word
            + pynutil.delete(NEMO_SPACE)
            + minute
            + m_word
            + pynutil.delete(NEMO_SPACE)
            + second
            + s_word
            + opt_zone_space
            + preserve
        )

        pattern_minute_word = minute + m_word
        pattern_minute_word_second = minute + m_word + pynutil.delete(NEMO_SPACE) + second + s_word
        pattern_second_word = second + s_word

        # Time zone specific patterns
        pattern_hour_suffix_space_zone = hour + h_suffix + pynini.accep(NEMO_SPACE) + zone
        pattern_hour_suffix_zone = hour + h_suffix + zone

        patterns = [
            pattern_hour_minute,
            pattern_hour_minute_second,
            pattern_hour_suffix,
            pattern_hour_suffix_minute,
            pattern_minute_p,
            pattern_second_s,
            pattern_minute_p_second_s,
            pattern_hour_word,
            pattern_hour_word_minute,
            pattern_hour_word_minute_second,
            pattern_minute_word,
            pattern_minute_word_second,
            pattern_second_word,
            pattern_hour_suffix_space_zone,
            pattern_hour_suffix_zone,
        ]

        final_graph = pynini.union(*patterns).optimize()

        self.fst = self.add_tokens(final_graph).optimize()
