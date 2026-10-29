# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.ta.graph_utils import NEMO_SIGMA, GraphFst
from nemo_text_processing.text_normalization.ta.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.ta.utils import get_abs_path


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying Tamil ordinals, e.g.
        1ஆவது  -> ordinal { integer: "முதலாவது" }
        ௨௭ஆவது -> ordinal { integer: "இருபத்தேழாவது" }
        2nd    -> ordinal { integer: "இரண்டாவது" }

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: CardinalFst, deterministic: bool = True):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        exceptions = pynini.string_file(get_abs_path("data/ordinal/exceptions.tsv"))
        affixes = pynini.string_file(get_abs_path("data/ordinal/affixes.tsv"))

        written_suffix = pynini.intersect(
            pynini.project(affixes, "input"), pynini.project(affixes, "output")
        ).optimize()

        cardinal_ending_to_ordinal = pynini.cdrewrite(affixes, "", "[EOS]", NEMO_SIGMA).optimize()

        ta_ordinal_graph = (cardinal.final_graph @ cardinal_ending_to_ordinal) + pynutil.delete(written_suffix)

        exception_inputs = pynini.project(exceptions, "input").optimize()
        ordinal_input = pynini.project(ta_ordinal_graph, "input").optimize()
        ordinal_inputs = pynini.difference(ordinal_input, exception_inputs).optimize()

        ordinal_graph = (ordinal_inputs @ ta_ordinal_graph).optimize()

        graph = pynini.union(exceptions, ordinal_graph).optimize()

        self.graph = graph
        final_graph = pynutil.insert('integer: "') + graph + pynutil.insert('"')
        self.fst = self.add_tokens(final_graph).optimize()
