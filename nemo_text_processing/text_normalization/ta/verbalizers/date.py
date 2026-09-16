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

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_NOT_QUOTE,
    NEMO_SPACE,
    GraphFst,
    delete_preserve_order,
    delete_space,
)


def _field(name: str) -> 'pynini.FstLike':
    """
    Consumes ``name: "value"``, emitting the value.
    """
    return pynutil.delete(f"{name}: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")


class DateFst(GraphFst):
    """
    Finite state transducer for verbalizing dates, e.g.
        date { day: "பதினைந்து" month: "ஜூன்" year: "இரண்டாயிரத்து இருபத்துநான்கு" } -> பதினைந்து ஜூன் இரண்டாயிரத்து இருபத்துநான்கு
        date { year: "இரண்டாயிரத்து இருபத்துநான்கு" month: "ஜூன்" day: "பதினைந்து" } -> இரண்டாயிரத்து இருபத்துநான்கு ஜூன் பதினைந்து
        date { era: "கிறிஸ்து பிறகு" year: "இரண்டாயிரத்து இருபத்துநான்கு" } -> கிறிஸ்து பிறகு இரண்டாயிரத்து இருபத்துநான்கு

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="date", kind="verbalize", deterministic=deterministic)

        day, month, year, era = _field("day"), _field("month"), _field("year"), _field("era")
        graph = (
            day + NEMO_SPACE + month
            | month + NEMO_SPACE + day
            | day + NEMO_SPACE + month + NEMO_SPACE + year
            | month + NEMO_SPACE + day + NEMO_SPACE + year
            | year + NEMO_SPACE + month + NEMO_SPACE + day
            | era
            | era + NEMO_SPACE + year
        )
        self.graph = graph + delete_space + delete_preserve_order
        self.fst = self.delete_tokens(self.graph).optimize()
