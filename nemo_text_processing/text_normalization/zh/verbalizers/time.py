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

from nemo_text_processing.text_normalization.zh.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space
from nemo_text_processing.text_normalization.zh.utils import get_abs_path


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
        alphabet_am = pynini.string_file(get_abs_path("data/time/AM.tsv"))
        alphabet_pm = pynini.string_file(get_abs_path("data/time/PM.tsv"))

        # fundamental components
        hour_component = pynutil.delete("hours: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        minute_component = pynutil.delete("minutes: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        second_component = pynutil.delete("seconds: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
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
        delete_verb = pynutil.delete("morphosyntactic_features: \"") + pynini.accep("差") + pynutil.delete("\"")
        graph_back_count = (
            (
                pynini.closure(delete_verb + pynutil.insert(' '))
                + hour_component
                + delete_space
                + pynini.closure(delete_verb)
                + delete_space
                + minute_component
            )
            | (
                pynini.closure(delete_verb + pynutil.insert(' '))
                + hour_component
                + delete_space
                + pynini.closure(delete_verb)
                + delete_space
                + second_component
            )
            | (
                pynini.closure(delete_verb + pynutil.insert(' '))
                + hour_component
                + delete_space
                + pynini.closure(delete_verb)
                + delete_space
                + minute_component
                + delete_space
                + second_component
            )
        )

        graph = graph_regular | graph_back_count

        delete_suffix = pynutil.delete("suffix: \"") + pynini.closure(alphabet_am | alphabet_pm) + pynutil.delete("\"")
        graph = graph | (graph + delete_space + delete_suffix)

        final_graph = graph

        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()
