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
from nemo_text_processing.text_normalization.zh.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space
from nemo_text_processing.text_normalization.zh.utils import get_abs_path
from pynini.lib import pynutil


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time e.g.
        tokens { time { hour: "五点" } } -> 五点
        tokens { time { minute: "三分" }' } -> 三分
        tokens { time { hour: "五点" minute: "三分" } } -> 五点三分
        tokens { time { affix: "am" hour: "五点" verb: "差" minute: "三分" }' } -> 早上五点差三分
        tokens { time { affix: "am" hour: "一点" minute: "三分" } } -> 深夜一点三分
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="time", kind="verbalize", deterministic=deterministic)

        # data imported to process am/pm into mandarin
        morning = pynini.string_file(get_abs_path("data/time/morning.tsv"))
        bfnoon = pynini.string_file(get_abs_path("data/time/before_noon.tsv"))
        noon_am = pynini.string_file(get_abs_path("data/time/noon_am.tsv"))
        noon_pm = pynini.string_file(get_abs_path("data/time/noon_pm.tsv"))
        afnoon = pynini.string_file(get_abs_path("data/time/after_noon.tsv"))
        night = pynini.string_file(get_abs_path("data/time/night.tsv"))
        mid_night = pynini.string_file(get_abs_path("data/time/mid_night.tsv"))
        late_night = pynini.string_file(get_abs_path("data/time/late_night.tsv"))
        early_morning = pynini.string_file(get_abs_path("data/time/early_morning.tsv"))

        # fundamental components
        hour_component = pynutil.delete("hour: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        minute_component = pynutil.delete("minute: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        second_component = pynutil.delete("second: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        graph_regular = (
            hour_component
            | minute_component
            | second_component
            | (hour_component + delete_space + minute_component + delete_space + second_component)
            | (hour_component + delete_space + minute_component)
            | (hour_component + delete_space + second_component)
            | (minute_component + delete_space + second_component)
        )

        # back count 三点差五分
        delete_verb = pynutil.delete("verb: \"") + pynini.accep("差") + pynutil.delete("\"")
        graph_back = (
            (hour_component + delete_space + delete_verb + delete_space + minute_component)
            | (hour_component + delete_space + delete_verb + delete_space + second_component)
            | (
                hour_component
                + delete_space
                + delete_verb
                + delete_space
                + minute_component
                + delete_space
                + second_component
            )
        )

        # with words 早上/晚上/etc.
        word_component = pynutil.delete("affix: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")

        # with am/pm
        graph_am = pynini.accep('am') | pynini.accep('AM') | pynini.accep('a.m.') | pynini.accep('A.M.')
        graph_pm = pynini.accep('pm') | pynini.accep('PM') | pynini.accep('p.m.') | pynini.accep('P.M.')

        delete_morning = pynutil.delete("affix: \"") + pynini.cross(graph_am, '早上') + pynutil.delete("\"")
        graph_morning = (
            delete_morning
            + delete_space
            + pynutil.delete("hour: \"")
            + morning
            + pynini.closure('点')
            + pynutil.delete("\"")
        )

        delete_bfnoon = pynutil.delete("affix: \"") + pynini.cross(graph_am, "上午") + pynutil.delete("\"")
        graph_bfnoon = (
            delete_bfnoon
            + delete_space
            + pynutil.delete("hour: \"")
            + bfnoon
            + pynini.closure('点')
            + pynutil.delete("\"")
        )

        delete_noonam = pynutil.delete("affix: \"") + pynini.cross(graph_am, "中午") + pynutil.delete("\"")
        graph_noon_am = (
            delete_noonam
            + delete_space
            + pynutil.delete("hour: \"")
            + noon_am
            + pynini.closure('点')
            + pynutil.delete("\"")
        )

        delete_noonpm = pynutil.delete("affix: \"") + pynini.cross(graph_pm, "中午") + pynutil.delete("\"")
        graph_noon_pm = (
            delete_noonpm
            + delete_space
            + pynutil.delete("hour: \"")
            + noon_pm
            + pynini.closure('点')
            + pynutil.delete("\"")
        )

        delete_afnoon = pynutil.delete("affix: \"") + pynini.cross(graph_pm, "下午") + pynutil.delete("\"")
        graph_afnoon = (
            delete_afnoon
            + delete_space
            + pynutil.delete("hour: \"")
            + afnoon
            + pynini.closure('点')
            + pynutil.delete("\"")
        )

        delete_night = pynutil.delete("affix: \"") + pynini.cross(graph_pm, "晚上") + pynutil.delete("\"")
        graph_night = (
            delete_night
            + delete_space
            + pynutil.delete("hour: \"")
            + night
            + pynini.closure('点')
            + pynutil.delete("\"")
        )

        delete_mnight = pynutil.delete("affix: \"") + pynini.cross(graph_am, "晚上") + pynutil.delete("\"")
        graph_mnight = (
            delete_mnight
            + delete_space
            + pynutil.delete("hour: \"")
            + mid_night
            + pynini.closure('点')
            + pynutil.delete("\"")
        )

        delete_lnight = pynutil.delete("affix: \"") + pynini.cross(graph_am, "深夜") + pynutil.delete("\"")
        graph_lnight = (
            delete_lnight
            + delete_space
            + pynutil.delete("hour: \"")
            + late_night
            + pynini.closure('点')
            + pynutil.delete("\"")
        )

        delete_emorning = pynutil.delete("affix: \"") + pynini.cross(graph_am, "凌晨") + pynutil.delete("\"")
        graph_emorning = (
            delete_emorning
            + delete_space
            + pynutil.delete("hour: \"")
            + early_morning
            + pynini.closure(NEMO_NOT_QUOTE)
            + pynutil.delete("\"")
        )

        graph_to_morpheme = (
            graph_morning
            | graph_bfnoon
            | graph_noon_am
            | graph_noon_pm
            | graph_afnoon
            | graph_lnight
            | graph_emorning
            | graph_mnight
            | graph_night
        )

        graph_to_mandarin = (
            graph_to_morpheme
            | (graph_to_morpheme + delete_space + minute_component + delete_space + second_component)
            | (graph_to_morpheme + delete_space + minute_component)
            | (graph_to_morpheme + delete_space + second_component)
            | (graph_to_morpheme + delete_space + minute_component + delete_space + second_component)
        )
        graph_mandarin_words = pynini.closure(word_component, 0, 1) + delete_space + (graph_regular | graph_back)
        graph_back_count = (
            (graph_to_morpheme + delete_space + delete_verb + delete_space + minute_component)
            | (
                graph_to_morpheme
                + delete_space
                + delete_verb
                + delete_space
                + minute_component
                + delete_space
                + second_component
            )
            | (graph_to_morpheme + delete_space + delete_verb + delete_space + second_component)
        )

        graph = graph_to_mandarin | graph_back_count | graph_mandarin_words

        # range
        symbols = pynini.accep("-") | pynini.accep("~") | pynini.accep("——") | pynini.accep("—")
        ranges = pynini.accep("从") | pynini.cross((symbols), "到") | pynini.accep("到") | pynini.accep("至")
        range_component = pynutil.delete("range: \"") + ranges + pynutil.delete("\"")
        graph_range = range_component + delete_space + graph + delete_space + range_component + graph
        graph_range2 = graph + delete_space + range_component + graph
        graph_range_final = graph_range | graph_range2

        final_graph = graph | graph_range_final

        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()
