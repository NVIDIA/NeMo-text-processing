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
from nemo_text_processing.text_normalization.zh.graph_utils import GraphFst
from nemo_text_processing.text_normalization.zh.utils import get_abs_path
from pynini.lib import pynutil


class TimeFst(GraphFst):
    """
    Finite state transducer for clasifying time, e.g.
        01:02 -> tokens { time { hours: "一" minutes: "二" } }
        1:02am -> tokens { time { hours: "一" minutes: "二" suffix: "am" } }
        1:02:03 -> tokens { time { hours: "一" minutes: "二" second: "三" } }
        1点5分19秒 -> tokens { time { hours: "一" minutes: "五" second: "秒" } }
        早上1点5分19秒 -> tokens { time { prefix: "am" hours: "一" minutes: "五" second: "秒" } }
        1点1刻 -> tokens { time { hours: "一" minutes: "十五" } }
        
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="time", kind="classify")

        # mappings imported
        hour = pynini.string_file(get_abs_path("data/time/hour.tsv"))
        minute = pynini.string_file(get_abs_path("data/time/minute.tsv"))
        second = pynini.string_file(get_abs_path("data/time/second.tsv"))
        word_am = pynini.string_file(get_abs_path("data/time/word_am.tsv"))
        word_pm = pynini.string_file(get_abs_path("data/time/word_pm.tsv"))
        alphabet_am = pynini.string_file(get_abs_path("data/time/suffix_am.tsv"))
        alphabet_pm = pynini.string_file(get_abs_path("data/time/suffix_pm.tsv"))

        # gramamr for time, separated by colons 05:03:13
        symbol = pynutil.delete(":") | pynutil.delete("：")
        hour_component = pynutil.insert("hour: \"") + hour + pynutil.insert('点') + pynutil.insert("\"")
        minute_component = pynutil.insert("minute: \"") + minute + pynutil.insert('分') + pynutil.insert("\"")
        second_component = pynutil.insert("second: \"") + second + pynutil.insert('秒') + pynutil.insert("\"")
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
        quarter = pynini.union('一', '二', '三', '四')
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
        hour_component = pynutil.insert("hour: \"") + hour + (hour_clock | hour_duration) + pynutil.insert("\"")
        minute_component = (
            pynutil.insert("minute: \"") + minute + (minute_clock | minute_duration) + pynutil.insert("\"")
        )
        second_component = (
            pynutil.insert("second: \"") + second + (second_clock | second_duration) + pynutil.insert("\"")
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
        backcount = pynutil.insert("verb: \"") + pynini.accep('差') + pynutil.insert("\"")
        graph_hour = (
            (hour_component + pynutil.insert(' ') + backcount + pynutil.insert(' ') + minute_component)
            | (hour_component + pynutil.insert(' ') + backcount + pynutil.insert(' ') + second_component)
            | (
                hour_component
                + pynutil.insert(' ')
                + backcount
                + pynutil.insert(' ')
                + minute_component
                + pynutil.insert(' ')
                + second_component
            )
        )
        graph_minute = minute_component + pynutil.insert(' ') + backcount + pynutil.insert(' ') + second_component
        graph_backcount = graph_hour | graph_minute

        # grammar for time, with am, pr, or Mandarin words as prefiex/suffix 早上5点 05：04：04am
        prefix_am = pynini.closure(word_am, 0, 1)
        prefix_pm = pynini.closure(word_pm, 0, 1)
        suffix_am = pynini.closure(alphabet_am, 0, 1)
        suffix_pm = pynini.closure(alphabet_pm, 0, 1)

        am_component = (pynutil.insert("affix: \"") + prefix_am + pynutil.insert("\"")) | (
            pynutil.insert("affix: \"") + suffix_am + pynutil.insert("\"")
        )
        pm_component = (pynutil.insert("affix: \"") + prefix_pm + pynutil.insert("\"")) | (
            pynutil.insert("affix: \"") + suffix_pm + pynutil.insert("\"")
        )

        graph_affix = (
            (am_component | pm_component) + pynutil.insert(' ') + (graph_colon | graph_clock_period | graph_backcount)
        ) | (
            (graph_clock_period | graph_colon | graph_backcount) + pynutil.insert(' ') + (am_component | pm_component)
        )

        graph = graph_colon | graph_clock_period | graph_backcount | graph_affix

        # range
        ranges = (
            pynini.accep("从")
            | pynini.accep("-")
            | pynini.accep("~")
            | pynini.accep("——")
            | pynini.accep("—")
            | pynini.accep("到")
            | pynini.accep("至")
        )
        graph_range = (
            pynini.closure((pynutil.insert("range: \"") + ranges + pynutil.insert("\"") + pynutil.insert(" ")), 0, 1)
            + graph
            + pynutil.insert(" ")
            + pynutil.insert("range: \"")
            + ranges
            + pynutil.insert("\"")
            + pynutil.insert(" ")
            + graph
        )

        final_graph = self.add_tokens(graph | graph_range)
        self.fst = final_graph.optimize()
