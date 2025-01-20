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

from nemo_text_processing.inverse_text_normalization.hi.utils import apply_fst
from nemo_text_processing.text_normalization.hi.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space


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
            pynutil.delete("city_code: \"")
            + pynutil.insert("०")
            + delete_space
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
            + pynini.accep(" "),
            0,
            1,
        )
        delete_tokens = self.delete_tokens(optional_country_code + number_part)
        delete_tokens |= self.delete_tokens(optional_city_code + number_part)
        self.fst = delete_tokens.optimize()


# from nemo_text_processing.inverse_text_normalization.hi.taggers.cardinal import CardinalFst
# cardinal = CardinalFst()
# telephone = TelephoneFst(cardinal)
# input_text = 'telephone { country_code: "९१" number_part: "९८७६५४३२१०"  }'
# input_text = 'telephone { country_code: "९१" number_part: "९४१११२३४१२"  }'
# input_text = 'telephone { country_code: "९१" number_part: "९४२२२२२२२"  }'
# input_text = 'telephone{ country_code: "९१" number_part: "११२३४५६७८९" }'
# input_text = 'telephone{ country_code: "९१" number_part: "९८७६५४३२११" }'
# input_text = 'telephone{ country_code: "९१" number_part: "९४५६७८९०१२" }'
# input_text = 'telephone{ country_code: "९१" number_part: "९५६७८९०१२३" }'
# input_text = 'telephone { city_code: "७९" number_part: "१९८७६५४"  }'
# input_text = 'telephone { city_code: "४०" number_part: "२७८१८३९"  }'
# input_text = 'telephone { city_code: "११" number_part: "२९४१११२"  }'
# input_text = 'telephone { city_code: "८०" number_part: "२९४१११२"  }'
# output = apply_fst(input_text, telephone.fst)
# print(output)
