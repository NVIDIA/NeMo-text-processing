# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
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

from nemo_text_processing.text_normalization.pt.graph_utils import (
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
)
from nemo_text_processing.text_normalization.pt.utils import get_abs_path


class OrdinalFst(GraphFst):
    """
    Finite state transducer for verbalizing Portuguese ordinals, e.g.
        ordinal { integer: "primeiro" morphosyntactic_features: "gender_masc" } -> primeiro
        ordinal { integer: "primeira" morphosyntactic_features: "gender_fem" } -> primeira (feminine rewrite applied)

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="ordinal", kind="verbalize", deterministic=deterministic)
        integer = (
            pynutil.delete('integer: "')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        fem_rewrite = pynini.string_file(
            get_abs_path("data/ordinals/feminine.tsv")
        )
        feminine_rewrite = pynini.cdrewrite(
            fem_rewrite,
            "",
            pynini.union(NEMO_SPACE, pynini.accep("[EOS]")),
            NEMO_SIGMA,
        )

        graph_masc = (
            integer
            + pynutil.delete(' morphosyntactic_features: "gender_masc"')
        )
        graph_fem = (
            (integer @ feminine_rewrite)
            + pynutil.delete(' morphosyntactic_features: "gender_fem"')
        )
        self.fst = self.delete_tokens(
            pynini.union(graph_masc, graph_fem)
        ).optimize()
