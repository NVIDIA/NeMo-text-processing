# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space


class TelephoneFst(GraphFst):
    """
    Finite state transducer for verbalizing telephone
        e.g. tokens { telephone { number_part: "08-789 52 25" } } -> 08-789 52 25
        e.g. tokens { telephone { country_code: "telefon" number_part: "112" } } -> telefon112
    """

    def __init__(self, project_input: bool = False):
        super().__init__(name="telephone", kind="verbalize", project_input=project_input)
        
        number_part = (
            pynutil.delete("number_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        
        country_code = (
            pynutil.delete("country_code:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        extension = (
            pynutil.delete("extension:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        
        # Optional fields
        optional_country_code = pynini.closure(country_code + delete_space, 0, 1)
        optional_extension = pynini.closure(delete_space + extension, 0, 1)
        
        # Main pattern: [country_code] number_part [extension]
        graph = optional_country_code + number_part + optional_extension

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()