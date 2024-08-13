# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.he.graph_utils import NEMO_CHAR, NEMO_SIGMA, GraphFst, \
    delete_space, NEMO_ALPHA


class WhiteListFst(GraphFst):
    """
    Finite state transducer for verbalizing whitelist
        e.g. tokens { name: "mrs." } -> mrs.
    """

    def __init__(self):
        super().__init__(name="whitelist", kind="verbalize")
        # Keep the prefix if exists and add a dash
        optional_prefix = pynini.closure(
            pynutil.delete("prefix:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_ALPHA, 1)
            + pynutil.delete("\"")
            + delete_space,
            0,
            1,
        )
        graph = (
            pynutil.delete("name:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_CHAR - " ", 1)
            + pynutil.delete("\"")
        )
        graph = graph @ pynini.cdrewrite(pynini.cross(u"\u00A0", " "), "", "", NEMO_SIGMA)
        final_graph = optional_prefix + graph
        self.fst = final_graph.optimize()


if __name__ == "__main__":

    from nemo_text_processing.inverse_text_normalization.he.graph_utils import apply_fst

    g = WhiteListFst().fst

    # To test this FST, remove comment out and change the input text
    # apply_fst('טקסט לבדיקה כאן', g)
