# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2026 and onwards Google, Inc.
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

from nemo_text_processing.inverse_text_normalization.te.graph_utils import (
    GraphFst,
    delete_extra_space,
    delete_space,
)
from nemo_text_processing.inverse_text_normalization.te.verbalizers.verbalize import VerbalizeFst


class VerbalizeFinalFst(GraphFst):
    """
    Verbalizes full Telugu ITN token output.

    Example:
        tokens { cardinal { integer: "౨౩" } } -> ౨౩
        tokens { name: "పరుగులు" } -> పరుగులు
    """

    def __init__(self):
        super().__init__(name="verbalize_final", kind="verbalize")

        verbalize = VerbalizeFst().fst

        graph = (
            pynutil.delete("tokens")
            + delete_space
            + pynutil.delete("{")
            + delete_space
            + verbalize
            + delete_space
            + pynutil.delete("}")
        )

        graph = delete_space + graph + pynini.closure(delete_extra_space + graph) + delete_space

        self.fst = graph.optimize()