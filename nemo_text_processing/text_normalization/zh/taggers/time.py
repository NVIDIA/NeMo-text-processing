# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.zh.graph_utils import GraphFst
from nemo_text_processing.text_normalization.zh.utils import get_abs_path


class TimeFst(GraphFst):
    """
    Finite state transducer for clasifying time, e.g.
        01:02 -> tokens { time { hours: "一" minutes: "二" } }
        1:02:03 -> tokens { time { hours: "一" minutes: "二" second: "三" } }
        1点5分19秒 -> tokens { time { hours: "一" minutes: "五" second: "秒" } }
        1点1刻 -> tokens { time { hours: "一" minutes: "一刻" } }
    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="time", kind="classify", deterministic=deterministic)

        # mappings imported
        hour = pynini.string_file(get_abs_path("data/time/hour.tsv"))
        minute = pynini.string_file(get_abs_path("data/time/minute.tsv"))
        second = pynini.string_file(get_abs_path("data/time/second.tsv"))
        alphabet_am = pynini.string_file(get_abs_path("data/time/AM.tsv"))
        alphabet_pm = pynini.string_file(get_abs_path("data/time/PM.tsv"))

        # gramamr for time, separated by colons 05:03:13
        symbol = pynutil.delete(":") | pynutil.delete("：")
        hour_component = pynutil.insert("hours: \"") + hour + pynutil.insert('点') + pynutil.insert("\"")
        minute_component = pynutil.insert("minutes: \"") + minute + pynutil.insert('分') + pynutil.insert("\"")
        second_component = pynutil.insert("seconds: \"") + second + pynutil.insert('秒') + pynutil.insert("\"")
        # combining 3 components
        hour_minute_second = (
            hour_component
            + symbol
            + pynutil.insert(' ')
            + minute_component
            + symbol
            + pynutil.insert(' ')
            + second_component
        )
        hour_minute = hour_component + symbol + pynutil.insert(' ') + minute_component
        graph_colon = hour_minute_second | hour_minute

        # gramamr for time as clock, with morphems, 点， 分， 秒
        hour_clock = pynini.accep("点") | pynini.cross("點", "点")
        minute_clock = pynini.accep("分") | pynini.accep('刻')
        second_clock = pynini.accep('秒')
        # grammar for time, as period of time 小时，分钟，秒
        hour_duration = (
            pynini.accep('个点')
            | pynini.cross("個點", "個点")
            | pynini.accep('小时')
            | pynini.cross('小時', '小时')
            | pynini.accep('个小时')
            | pynini.cross('個小時', '个小时')
            | pynini.accep('个钟头')
            | pynini.cross('個鐘頭', '个钟头')
        )
        minute_duration = pynini.accep("分钟") | pynini.accep('刻') | pynini.accep('刻钟')
        second_duration = pynini.accep("秒钟") | pynini.cross('秒鐘', '秒钟') | pynini.accep('秒')
        # combining two above
        hour_component = pynutil.insert("hours: \"") + hour + (hour_clock | hour_duration) + pynutil.insert("\"")
        minute_component = (
            pynutil.insert("minutes: \"") + minute + (minute_clock | minute_duration) + pynutil.insert("\"")
        )
        second_component = (
            pynutil.insert("seconds: \"") + second + (second_clock | second_duration) + pynutil.insert("\"")
        )
        hour_minute = hour_component + pynutil.insert(' ') + minute_component
        hour_second = hour_component + pynutil.insert(' ') + second_component
        minute_second = minute_component + pynutil.insert(' ') + second_component
        clock_all = hour_component + pynutil.insert(' ') + minute_component + pynutil.insert(' ') + second_component
        graph_clock_period = (
            hour_component
            | minute_component
            | second_component
            | hour_minute
            | hour_second
            | minute_second
            | clock_all
        )

        # gramamr for time, back count; 五点差n分n秒
        backcount = pynutil.insert("morphosyntactic_features: \"") + pynini.accep('差') + pynutil.insert("\"")
        graph_hour = (
            (
                pynini.closure(backcount)
                + pynutil.insert(' ')
                + hour_component
                + pynutil.insert(' ')
                + pynini.closure(backcount)
                + pynutil.insert(' ')
                + minute_component
            )
            | (
                pynini.closure(backcount)
                + pynutil.insert(' ')
                + hour_component
                + pynutil.insert(' ')
                + pynini.closure(backcount)
                + pynutil.insert(' ')
                + second_component
            )
            | (
                pynini.closure(backcount)
                + pynutil.insert(' ')
                + hour_component
                + pynutil.insert(' ')
                + pynini.closure(backcount)
                + pynutil.insert(' ')
                + minute_component
                + pynutil.insert(' ')
                + second_component
            )
        )
        graph_minute = minute_component + pynutil.insert(' ') + backcount + pynutil.insert(' ') + second_component
        graph_backcount = graph_hour | graph_minute

        # grammar for time, with am, pr, or Mandarin words as prefiex/suffix 早上5点 05：04：04am
        suffix_am = pynini.closure(alphabet_am, 0, 1)
        suffix_pm = pynini.closure(alphabet_pm, 0, 1)

        am_component = pynutil.insert("suffix: \"") + suffix_am + pynutil.insert("\"")
        pm_component = pynutil.insert("suffix: \"") + suffix_pm + pynutil.insert("\"")

        graph_suffix = (
            (graph_clock_period | graph_colon | graph_backcount) + pynutil.insert(' ') + (am_component | pm_component)
        )

        graph = graph_colon | graph_clock_period | graph_backcount | graph_suffix

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
