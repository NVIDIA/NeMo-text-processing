# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.hi.graph_utils import (
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_zero_or_one_space,
    insert_space,
)
from nemo_text_processing.text_normalization.hi.utils import get_abs_path


class RomanFst(GraphFst):
    """
    Finite state transducer for verbalizing Roman numerals in Hindi.
        roman { preserve_order: true key_cardinal: "भास्कर" integer: "II" } -> भास्कर दो
        roman { preserve_order: true key_cardinal: "कक्षा" integer: "XII" } -> कक्षा बारह
        roman { preserve_order: true integer: "XII" default_ordinal: "बारहवीं" key_cardinal: "कक्षा" } -> बारहवीं कक्षा
        roman { preserve_order: true integer: "IV" default_ordinal: "चौथी" key_cardinal: "कक्षा" } -> चौथी कक्षा

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="roman", kind="verbalize", deterministic=deterministic)

        key_cardinal = (
            pynutil.delete('key_cardinal: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')
        ).optimize()

        integer = (pynutil.delete('integer: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')).optimize()

        default_ordinal = (
            pynutil.delete('default_ordinal: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')
        ).optimize()

        ignore_integer = (
            pynutil.delete('integer: "') + pynutil.delete(pynini.closure(NEMO_NOT_QUOTE, 1)) + pynutil.delete('"')
        ).optimize()

        drop_preserve_order = pynini.closure(
            delete_zero_or_one_space
            + pynutil.delete("preserve_order:")
            + delete_zero_or_one_space
            + pynutil.delete("true")
            + delete_zero_or_one_space,
            0,
            1,
        ).optimize()

        key_first = (
            drop_preserve_order
            + key_cardinal
            + delete_zero_or_one_space
            + insert_space
            + integer
            + drop_preserve_order
        ).optimize()

        numeral_first = (
            drop_preserve_order
            + integer
            + delete_zero_or_one_space
            + insert_space
            + key_cardinal
            + drop_preserve_order
        ).optimize()

        glued_ordinal = (
            drop_preserve_order
            + ignore_integer
            + delete_zero_or_one_space
            + default_ordinal
            + pynini.closure(delete_zero_or_one_space + insert_space + key_cardinal, 0, 1)
            + drop_preserve_order
        ).optimize()

        graph = pynini.union(key_first, numeral_first, glued_ordinal).optimize()

        self.fst = self.delete_tokens(graph).optimize()
