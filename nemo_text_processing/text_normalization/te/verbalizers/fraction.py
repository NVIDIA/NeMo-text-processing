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

from nemo_text_processing.text_normalization.te.graph_utils import NEMO_NOT_QUOTE, GraphFst
from nemo_text_processing.text_normalization.te.utils import get_abs_path, load_labels


class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing Telugu fractions.

    e.g.
    fraction { integer_part: "ఇరవై మూడు" numerator: "నాలుగు" denominator: "ఆరు" }
    -> ఇరవై మూడు మరియు నాలుగు బై ఆరు
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="fraction", kind="verbalize", deterministic=deterministic)

        minus = dict(load_labels(get_abs_path("data/decimal/symbols.tsv")))["minus"]
        words = load_labels(get_abs_path("data/fraction/symbols.tsv"))
        bai, mariyu = words[0][0].strip(), words[1][0].strip()

        optional_sign = pynini.closure(pynini.cross("negative: \"true\" ", minus + " "), 0, 1)

        def field(name, end=""):
            return pynutil.delete(f'{name}: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete(f'"{end}')

        bai_frac = field("numerator", " ") + pynutil.insert(f" {bai} ") + field("denominator")
        cross_frac = (
            pynutil.delete('morphosyntactic_features: "slash" ')
            + field("numerator", " ")
            + pynutil.insert("/")
            + field("denominator")
        )
        body = cross_frac | bai_frac | field("morphosyntactic_features")
        with_int = field("integer_part") + pynutil.delete(" ") + pynutil.insert(f" {mariyu} ")

        self.fst = self.delete_tokens(optional_sign + (body | with_int + body)).optimize()
