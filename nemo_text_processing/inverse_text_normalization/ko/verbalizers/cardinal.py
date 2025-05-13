# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.ko.graph_utils import (
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_space,
)


class CardinalFst(GraphFst):
    """
    Finite state transducer for verbalizing cardinal
        e.g. cardinal { negative: "-" integer: "23" } -> -23
    """

    def __init__(self):
        super().__init__(name="cardinal", kind="verbalize")
        negative_sign = (
            pynutil.delete("negative:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.accep("-") 
            + pynutil.delete("\"")
        )

        optional_sign_output = pynini.closure(negative_sign + delete_space, 0, 1)

        digits_from_tag = pynini.closure(NEMO_NOT_QUOTE, 1) 
        integer_cardinal = (
            pynutil.delete("integer:")
            + delete_space
            + pynutil.delete("\"")
            + digits_from_tag
            + pynutil.delete("\"")
        )

        graph = integer_cardinal
        final_graph = optional_sign_output + graph
        self.fst = self.delete_tokens(final_graph).optimize()