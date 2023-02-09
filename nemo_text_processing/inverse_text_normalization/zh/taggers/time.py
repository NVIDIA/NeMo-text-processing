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
from nemo_text_processing.inverse_text_normalization.zh.graph_utils import GraphFst, delete_space
from nemo_text_processing.inverse_text_normalization.zh.utils import get_abs_path
from pynini.lib import pynutil


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time
    e.g., 五d点 -> time { hours: "5" minutes: "00" }
    e.g., 正午 -> time { hours: "12" minutes: "00" }
    e.g., 两点一刻 -> time { hours: "2" minutes: "15" }
    e.g., 上午九点 -> time { hours: "2"  minutes: "00" affix: "a.m." }
    e.g., 五点差五分 -> time { hours: "4" minutes: "55"}
    """

    def __init__(self):
        super().__init__(name="time", kind="classify")

        # data imported
        hours = pynini.string_file(get_abs_path("data/time/time_hours-nano.tsv"))  # hours from 1 to 24
        minutes = pynini.string_file(get_abs_path("data/time/time_minutes-nano.tsv"))  # minutes from 1 to 60
        hours_to = pynini.string_file(get_abs_path("data/time/hours_to-nano.tsv"))  # used for back counting, see below
        minutes_to = pynini.string_file(
            get_abs_path("data/time/minutes_to-nano.tsv")
        )  # used for back counting, see below

        # graph for one quarter (e.g., 10:15)
        graph_quarter = pynini.accep("一刻") | pynini.accep("壹刻") | pynini.accep("幺刻")
        graph_quarter = pynini.cross(graph_quarter, "15")

        # grammar for two quarters or half (e.g., 10:30)
        graph_half = pynini.accep("半").ques
        graph_half = pynini.cross(graph_half, "30")
        graph_half_alt = pynini.accep("二刻") | pynini.accep("貳刻") | pynini.accep("两刻") | pynini.accep("兩刻")
        graph_half_alt = pynini.cross(graph_half_alt, "30")
        graph_half = graph_half | graph_half_alt

        # grammar for three quarters (e.g., 10:45)
        graph_three_quarter = pynini.accep("三刻", "45") | pynini.accep("叁刻", "45")
        graph_three_quarter = pynini.cross(graph_three_quarter, "45")

        # combining grammars quarter, two quater, and three quarter
        graph_fractions = graph_quarter | graph_half | graph_three_quarter

        # graph for "Noon-12PM"
        graph_noon = pynini.cross("中午", "12") | pynini.cross("正午", "12") | pynini.cross("午间", "12")
        graph_midnight = pynini.cross("午夜", "0") | pynini.cross("半夜", "0") | pynini.cross("子夜", "0")

        # graph for hour
        graph_delete_hours = (
            pynutil.delete("点") | pynutil.delete("點") | pynutil.delete("时") | pynutil.delete("時")
        )  # "点": Mandarin for "hour | o'clock" (e.g.,十点=ten o' clock)
        graph_hours = hours + graph_delete_hours

        # graph for minutes
        graph_minutes = pynutil.delete('分')
        graph_minutes = minutes + graph_minutes

        # add tokenization for hours position component
        graph_hours_component = pynini.union(graph_hours, graph_noon, graph_midnight)  # what to put at hours-position
        graph_hours_component = pynutil.insert("hours: \"") + graph_hours_component + pynutil.insert("\"")

        # add tokenization for minutes position component
        graph_minutes_component = pynini.union(graph_minutes, graph_fractions) | pynutil.insert(
            "00"
        )  # what to put at minutes-position
        graph_minutes_component = pynutil.insert(" minutes: \"") + graph_minutes_component + pynutil.insert("\"")
        graph_minutes_component = delete_space + graph_minutes_component

        # combine two above to process digit + "hours" + digit " minutes/null" (e.g., 十点五十分/十点五十-> hours: "10" minutes: "50")
        graph_time_standard = graph_hours_component + graph_minutes_component

        # combined hours and minutes but with prefix
        graph_time_standard_affix = (
            (
                (pynutil.delete("上午") | pynutil.delete("早上"))
                + graph_time_standard
                + pynutil.insert(" affix: \"")
                + pynutil.insert("a.m.")
                + pynutil.insert("\"")
            )
        ) | (
            (
                (pynutil.delete("下午") | pynutil.delete("晚上"))
                + graph_time_standard
                + pynutil.insert(" affix: \"")
                + pynutil.insert("p.m.")
                + pynutil.insert("\"")
            )
        )

        # combined hours and minutes (上午十點五十-> hours: "10" minutes: "50" affix: "a.m.")
        graph_time_standard = graph_time_standard | graph_time_standard_affix

        # grammar for back-counting
        # converting hours back
        graph_hours_to_component = graph_hours | graph_noon | graph_midnight  # | graph_hours_count
        graph_hours_to_component @= hours_to  # hours_to is the string_file data
        graph_hours_to_component = pynutil.insert("hours: \"") + graph_hours_to_component + pynutil.insert("\"")

        # converting minutes back
        graph_minutes_to_component = minutes | graph_half | graph_quarter | graph_three_quarter | graph_half_alt
        graph_minutes_to_component @= minutes_to  # minutes_to is the string_file data
        graph_minutes_to_component = pynutil.insert(" minutes: \"") + graph_minutes_to_component + pynutil.insert("\"")

        graph_delete_back_counting = pynutil.delete("差") | pynutil.delete("还有") | pynutil.delete("還有")
        graph_delete_minutes = pynutil.delete("分") | pynutil.delete("分钟") | pynutil.delete("分鐘")

        # adding a.m. and p.m.
        graph_time_to = (
            graph_hours_to_component + graph_delete_back_counting + graph_minutes_to_component + graph_delete_minutes
        )
        graph_time_to_affix = (
            (
                (pynutil.delete("上午") | pynutil.delete("早上"))
                + graph_time_to
                + pynutil.insert(" affix: \"")
                + pynutil.insert("a.m.")
                + pynutil.insert("\"")
            )
        ) | (
            (
                (pynutil.delete("下午") | pynutil.delete("晚上"))
                + graph_time_to
                + pynutil.insert(" prefix: \"")
                + pynutil.insert("p.m.")
                + pynutil.insert("\"")
            )
        )
        graph_time_to = graph_time_to | graph_time_to_affix

        # final grammar
        final_graph = graph_time_standard | graph_time_to
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
