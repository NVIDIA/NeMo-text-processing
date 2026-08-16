# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2023, Jim O'Regan for Språkbanken Tal
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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, NEMO_WHITE_SPACE, GraphFst
from nemo_text_processing.text_normalization.se.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.se.utils import get_abs_path


class TelephoneFst(GraphFst):
    """Classifies telephone numbers as groups of nominative cardinals."""

    def __init__(self, cardinal: CardinalFst, deterministic: bool = True):
        super().__init__(name="telephone", kind="classify", deterministic=deterministic)

        nominative_cardinal = cardinal if deterministic else CardinalFst(deterministic=True)
        nominative = nominative_cardinal.graphs["nom_sg"]
        non_zero = NEMO_DIGIT - "0"
        zero = pynini.cross("0", "nolla")
        spoken_space = pynutil.insert(" ")

        ordinary_group = pynini.closure(NEMO_DIGIT, 2, 4) @ nominative
        one_leading_zero = zero + spoken_space + ((non_zero + pynini.closure(NEMO_DIGIT, 0, 2)) @ nominative)
        two_leading_zeroes = (
            zero
            + spoken_space
            + zero
            + spoken_space
            + ((non_zero + pynini.closure(NEMO_DIGIT, 0, 1)) @ nominative)
        )
        all_zeroes = zero + pynini.closure(pynutil.insert(" ") + zero, 1, 3)
        group = pynini.union(ordinary_group, one_leading_zero, two_leading_zeroes, all_zeroes).optimize()

        group_separator = pynutil.delete(pynini.closure(NEMO_WHITE_SPACE, 1)) + pynutil.insert(" ")
        number = group + pynini.closure(group_separator + group, 1)

        prefix = pynini.string_file(get_abs_path("data/telephone/telephone_abbr.tsv"))
        prefixed_number = prefix + group_separator + number
        number |= prefixed_number

        number_part = pynutil.insert('number_part: "') + number + pynutil.insert('"')
        self.fst = self.add_tokens(number_part).optimize()
