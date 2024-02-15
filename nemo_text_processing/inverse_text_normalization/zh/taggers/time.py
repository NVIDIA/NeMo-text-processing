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

from nemo_text_processing.inverse_text_normalization.zh.graph_utils import GraphFst
from nemo_text_processing.inverse_text_normalization.zh.utils import get_abs_path


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time
    e.g., 五点十分 -> time { hours: "05" minutes: "10" }
    e.g., 五时十五分 -> time { hours: "05" minutes: "15" }
    e.g., 十五点十分 -> time { hours: "15" minutes: "10" }
    e.g., 十五点十分二十秒 -> time { hours: "15" minutes: "10" seconds: "20 }
    e.g., 两点一刻 -> time { hours: "2" minutes: "1刻" }
    e.g., 五点 -> time { hours: "5点" }
    e.g., 五小时 -> time { hours: "5小时" }
    e.g., 五分 -> time { minutess: "5分" }
    e.g., 五分钟 -> time { seconds: "5分钟" }
    e.g., 五秒 -> time { seconds: "5秒" }
    e.g., 五秒钟 -> time { seconds: "5秒钟" }
    """

    def __init__(self):
        super().__init__(name="time", kind="classify")

        hours = pynini.string_file(get_abs_path("data/time/time_hours.tsv"))
        minutes = pynini.string_file(get_abs_path("data/time/time_minutes.tsv"))
        seconds = pynini.string_file(get_abs_path("data/time/time_seconds.tsv"))
        quarters = pynini.string_file(get_abs_path("data/time/time_quarters.tsv"))
        for_mandarin = pynini.string_file(get_abs_path("data/time/time_mandarin.tsv"))

        graph_delete_hours = pynutil.delete("点") | pynutil.delete("點") | pynutil.delete("时") | pynutil.delete("時")
        graph_hours = hours + graph_delete_hours
        graph_hours_component = pynutil.insert('hours: "') + graph_hours + pynutil.insert('"')

        graph_minutes = pynutil.delete("分")
        graph_minutes = minutes + graph_minutes
        graph_minutes_component = pynutil.insert('minutes: "') + graph_minutes + pynutil.insert('"')

        graph_seconds = pynutil.delete("秒")
        graph_seconds = seconds + graph_seconds
        graph_seconds_component = pynutil.insert('seconds: "') + graph_seconds + pynutil.insert('"')

        graph_time_standard = (graph_hours_component + pynutil.insert(" ") + graph_minutes_component) | (
            graph_hours_component
            + pynutil.insert(" ")
            + graph_minutes_component
            + pynutil.insert(" ")
            + graph_seconds_component
        )

        quarter_mandarin = (
            quarters + pynini.accep("刻") | pynini.cross("刻鈡", "刻钟") | pynini.accep("刻钟") | pynini.accep("半")
        )
        hour_mandarin = (
            pynini.accep("点")
            | pynini.accep("时")
            | pynini.cross("點", "点")
            | pynini.cross("時", "时")
            | pynini.accep("小时")
            | pynini.cross("小時", "小时")
            | pynini.cross("個點", "个点")
            | pynini.accep("个点")
            | pynini.accep("个钟头")
            | pynini.cross("個鐘頭", "个钟头")
            | pynini.accep("个小时")
            | pynini.cross("個小時", "个小时")
        )
        minute_mandarin = pynini.accep("分") | pynini.cross("分鐘", "分钟") | pynini.accep("分钟")
        second_mandarin = pynini.accep("秒") | pynini.cross("秒鐘", "秒钟") | pynini.accep("秒钟")

        hours_only = for_mandarin + hour_mandarin
        minutes_only = for_mandarin + minute_mandarin
        seconds_only = for_mandarin + second_mandarin

        graph_mandarin_hour = pynutil.insert('hours: "') + hours_only + pynutil.insert('"')
        graph_mandarin_minute = pynutil.insert('minutes: "') + minutes_only + pynutil.insert('"')
        graph_mandarin_second = pynutil.insert('seconds: "') + seconds_only + pynutil.insert('"')
        graph_mandarin_quarter = pynutil.insert('minutes: "') + quarter_mandarin + pynutil.insert('"')
        graph_mandarins = (
            graph_mandarin_hour
            | graph_mandarin_minute
            | graph_mandarin_second
            | graph_mandarin_quarter
            | (graph_mandarin_hour + pynutil.insert(" ") + graph_mandarin_quarter)
        )

        final_graph = graph_time_standard | graph_mandarins
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
