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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_SPACE, GraphFst, convert_space


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone numbers, e.g.
        noll åtta sjuhundraåttionio femtiotvå tjugofem -> tokens { name: "08-789 52 25" }

    Args:
        tn_cardinal_tagger: TN Cardinal Tagger
    """

    def __init__(self, tn_cardinal_tagger: GraphFst, tn_telephone_tagger: GraphFst):
        super().__init__(name="telephone", kind="classify")
        # country_plus_area_code = pynini.invert(tn_telephone_tagger.country_plus_area_code).optimize()
        area_codes = pynini.invert(tn_telephone_tagger.area_codes).optimize()
        # lead = (country_plus_area_code | area_codes) + pynini.cross(" ", "-")
        lead = area_codes + pynini.cross(" ", "-")

        two_digits = pynini.invert(tn_cardinal_tagger.two_digits_read).optimize()
        three_digits = pynini.invert(tn_cardinal_tagger.three_digits_read).optimize()

        base_number_part = pynini.union(
            three_digits + NEMO_SPACE + three_digits + NEMO_SPACE + two_digits,
            three_digits + NEMO_SPACE + two_digits + NEMO_SPACE + two_digits,
            two_digits + NEMO_SPACE + two_digits + NEMO_SPACE + two_digits,
            three_digits + NEMO_SPACE + two_digits,
        )

        graph = convert_space(lead + base_number_part)
        final_graph = pynutil.insert("name: \"") + graph + pynutil.insert("\"")

        self.fst = final_graph.optimize()
