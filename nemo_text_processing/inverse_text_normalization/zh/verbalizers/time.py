# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.zh.graph_utils import NEMO_DIGIT, GraphFst, delete_space


class TimeFst(GraphFst):
    """
    Finite state transcucer for verbalizing time, e.g.,
    time { hours: "12" minutes: "30" } -> 12:30
    time { hours: "01" minutes: "30" } -> 01:30
    time { hours: "1" minutes: "30" seconds: "05" } -> 01:30:05
    time { hours: "1" minutes: "1刻" } -> 1点1刻
    time { hours: "一点" } -> 1点
    time { hours: "一小时" } -> 1小时
    time { hours: "一个钟头" } -> 1个钟头
    time { minutes: "一分" } -> 1分
    time { minutes: "一分钟" } -> 1分钟
    time { seconds: "一秒" } -> 1秒
    time { seconds: "一秒钟" } -> 1秒钟
    time { hours: "五点" minutes: "一刻" } -> 5点1刻
    """

    def __init__(self):
        super().__init__(name="time", kind="verbalize")
        # add_leading_zero = (NEMO_DIGIT + NEMO_DIGIT) | (pynutil.insert("0") + NEMO_DIGIT)
        token_hour = (
            pynutil.delete("hours:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_DIGIT, 1, 2)
            + pynutil.delete('"')
        )
        token_minute = (
            pynutil.delete("minutes:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_DIGIT, 1, 2)
            + pynutil.delete('"')
        )
        token_second = (
            pynutil.delete("seconds:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_DIGIT, 1, 2)
            + pynutil.delete('"')
        )

        add_colon = pynutil.insert(":")
        graph_regular_time = (token_hour + delete_space + add_colon + token_minute) | (
            token_hour + delete_space + add_colon + token_minute + delete_space + add_colon + token_second
        )

        hours = (
            pynini.accep("点")
            | pynini.accep("小时")
            | pynini.accep("时")
            | pynini.accep("个钟头")
            | pynini.accep("个点")
            | pynini.accep("个小时")
        )
        hour_mandarin = (
            pynutil.delete("hours:")
            + delete_space
            + pynutil.delete('"')
            + (pynini.closure(NEMO_DIGIT) + pynini.closure(hours, 1))
            + pynutil.delete('"')
        )
        minutes = pynini.accep("分") | pynini.accep("分钟") | pynini.accep("半")
        minute_mandarin = (
            pynutil.delete("minutes:")
            + delete_space
            + pynutil.delete('"')
            + (((pynini.closure(NEMO_DIGIT) + pynini.closure(minutes, 1))) | pynini.closure(minutes, 1))
            + pynutil.delete('"')
        )
        seconds = pynini.accep("秒") | pynini.accep("秒钟")
        second_mandarin = (
            pynutil.delete("seconds:")
            + delete_space
            + pynutil.delete('"')
            + (pynini.closure(NEMO_DIGIT) + pynini.closure(seconds, 1))
            + pynutil.delete('"')
        )
        quarters = pynini.accep("刻") | pynini.accep("刻钟")
        quarter_mandarin = (
            pynutil.delete("minutes:")
            + delete_space
            + pynutil.delete('"')
            + (pynini.closure(NEMO_DIGIT) + pynini.closure(quarters, 1))
            + pynutil.delete('"')
        )

        graph_mandarin_time = (
            hour_mandarin
            | minute_mandarin
            | second_mandarin
            | quarter_mandarin
            | (hour_mandarin + delete_space + quarter_mandarin)
            | (hour_mandarin + delete_space + minute_mandarin)
            | (hour_mandarin + delete_space + minute_mandarin + delete_space + second_mandarin)
            | (minute_mandarin + delete_space + second_mandarin)
        )

        final_graph = graph_regular_time | graph_mandarin_time
        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()
