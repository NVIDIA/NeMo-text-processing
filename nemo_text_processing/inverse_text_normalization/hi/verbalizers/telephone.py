# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2025 and onwards Google, Inc.
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
    delete_extra_space,
    delete_space,
)


class TelephoneFst(GraphFst):
    """
    Finite state transducer for verbalizing telephone, e.g.
        telephone { number_part: "123-123-5678" }
        -> 123-123-5678
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="telephone", kind="verbalize")

        number_part = pynutil.delete("number_part: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        optional_country_code = pynini.closure(
            pynutil.delete("country_code: \"")
            + pynutil.insert("+")
            + delete_space
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
            + pynini.accep(" "),
            0,
            1,
        )
        optional_city_code = pynini.closure(
            pynutil.delete("extension: \"")
            + pynutil.insert("०")
            + delete_space
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + delete_space
            + pynutil.insert("-")
            + pynutil.delete("\" ")
        )

        delete_tokens = self.delete_tokens(optional_country_code + number_part)
        delete_tokens |= self.delete_tokens(optional_city_code + number_part)
        self.fst = delete_tokens.optimize()