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

from nemo_text_processing.inverse_text_normalization.zh.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    GraphFst,
    delete_space,
)


class CardinalFst(GraphFst):
    def __init__(self):
        super().__init__(name="cardinal", kind="verbalize")

        # group numbers by three
        exactly_three_digits = NEMO_DIGIT ** 3
        at_most_three_digits = pynini.closure(NEMO_DIGIT, 1, 3)

        suffix = pynini.union(
            "千",
            "仟",
            "万",
            "十万",
            "百万",
            "千万",
            "亿",
            "十亿",
            "百亿",
            "千亿",
            "萬",
            "十萬",
            "百萬",
            "千萬",
            "億",
            "十億",
            "百億",
            "千億",
            "拾萬",
            "佰萬",
            "仟萬",
            "拾億",
            "佰億",
            "仟億",
            "拾万",
            "佰万",
            "仟万",
            "仟亿",
            "佰亿",
            "仟亿",
        )

        # inserting a "," between every 3 numbers
        group_by_threes = (
            at_most_three_digits + (pynutil.insert(",") + exactly_three_digits).closure()
        ) + pynini.closure(suffix)

        # remove the negative attribute and leaves the sign if occurs
        optional_sign = pynini.closure(
            pynutil.delete("negative: ")
            + delete_space
            + pynutil.delete('"')
            + pynini.accep("-")
            + pynutil.delete('"')
            + delete_space
        )

        # remove integer aspect
        graph = (
            pynutil.delete("integer:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_DIGIT, 0, 1)
            + pynini.closure(NEMO_SIGMA)
            + pynutil.delete('"')
        )
        graph = graph @ group_by_threes

        graph = optional_sign + graph

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
