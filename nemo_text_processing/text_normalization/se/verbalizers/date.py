# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    GraphFst,
    delete_extra_space,
    delete_space,
)
from pynini.lib import pynutil


class DateFst(GraphFst):
    """
    Finite state transducer for verbalizing date, e.g.
        date { month: "february" day: "five" year: "twenty twelve" preserve_order: true } -> february fifth twenty twelve
        date { month: "skábmamánnu" day: "gávccát" year: "duhátovccičuođivihttalogiguhtta" preserve_order: true } -> "skábmamánu gávccát beaivi duhátovccičuođivihttalogiguhtta"
    Args:
        ordinal: OrdinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="date", kind="verbalize", deterministic=deterministic)
        month_nom_to_gen_map = pynini.cdrewrite(pynini.cross("mánu", "mánnu"), "", "[EOS]", NEMO_SIGMA)

        month = pynini.closure(NEMO_NOT_QUOTE, 1)
        day = (
            pynutil.delete("day:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        month = (
            pynutil.delete("month:")
            + delete_space
            + pynutil.delete("\"")
            + (month @ month_nom_to_gen_map)
            + pynutil.delete("\"")
        )

        year = (
            pynutil.delete("year:")
            + delete_space
            + pynutil.delete("\"")
            + pynutil.insert("beaivi ")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + delete_space
            + pynutil.delete("\"")
        )

        graph_mdy = month + delete_extra_space + day + pynini.closure(delete_extra_space + year, 0, 1)

        optional_preserve_order = pynini.closure(
            pynutil.delete("preserve_order:") + delete_space + pynutil.delete("true") + delete_space
            | pynutil.delete("field_order:")
            + delete_space
            + pynutil.delete("\"")
            + NEMO_NOT_QUOTE
            + pynutil.delete("\"")
            + delete_space
        )

        final_graph = (graph_mdy | year) + delete_space + optional_preserve_order
        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()
