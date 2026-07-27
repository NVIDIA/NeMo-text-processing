# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.ar.graph_utils import NEMO_SIGMA, GraphFst


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying spoken Arabic time by inverting the
    text-normalization time verbalizer, e.g.
        الثالثة وخمس عشرة دقيقة -> time { hours: "3" minutes: "15" }
        الخامسة وعشر دقائق وثانيتان -> time { hours: "5" minutes: "10" seconds: "2" }
        التاسعة صباحًا -> time { hours: "9" suffix: "صباحًا" }

    Args:
        tn_time_verbalizer: TN time verbalizer, whose .graph maps the tagged fields
            (hours/minutes/seconds/suffix/zone) to their spoken Arabic form.
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, tn_time_verbalizer: GraphFst, deterministic: bool = True):
        super().__init__(name="time", kind="classify", deterministic=deterministic)
        # allow the spoken form to be matched with flexible spacing
        optional_delete_space = pynini.closure(NEMO_SIGMA | pynutil.delete(" ", weight=0.0001))
        graph = (tn_time_verbalizer.graph @ optional_delete_space).invert().optimize()
        self.fst = self.add_tokens(graph).optimize()
