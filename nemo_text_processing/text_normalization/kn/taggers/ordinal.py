# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
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

from nemo_text_processing.text_normalization.kn.graph_utils import (GraphFst,NEMO_SIGMA)
from nemo_text_processing.text_normalization.kn.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.kn.utils import get_abs_path


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying Kannada ordinals, e.g.
        ೧೦ನೇ -> ordinal { integer: "ಹತ್ತನೆಯ" }
        12ನೇ -> ordinal { integer: "ಹನ್ನೆರಡನೆಯ" } # English/arabic digits also supported 

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """
    def __init__(self,cardinal: CardinalFst,deterministic: bool = True):
        super().__init__( name="ordinal",kind="classify", deterministic=deterministic)

        exceptions = pynini.string_file(get_abs_path("data/ordinals/exceptions.tsv"))
        endings = pynini.string_file(get_abs_path("data/ordinals/ending.tsv"))
        kn_suffixes = pynini.string_file(get_abs_path("data/ordinals/kn_suffixes.tsv"))
        en_suffixes = pynini.string_file(get_abs_path("data/ordinals/en_suffixes.tsv"))

        drop_cardinal_ending = pynini.cdrewrite(pynutil.delete(endings),"","[EOS]", NEMO_SIGMA).optimize()

        kn_ordinal_graph = (cardinal.final_graph @ drop_cardinal_ending + kn_suffixes)
        en_ordinal_graph = (cardinal.final_graph @ drop_cardinal_ending + en_suffixes)
        ordinal_g = pynini.union(kn_ordinal_graph,en_ordinal_graph).optimize()

        exception_inputs = pynini.project(exceptions,"input").optimize()
        ordinal_input = pynini.project(ordinal_g,"input").optimize()
        ordinal_inputs = pynini.difference(ordinal_input,exception_inputs).optimize()

        ordinal_graph = (ordinal_inputs @ ordinal_g).optimize()

        graph = pynini.union(exceptions,ordinal_graph).optimize()

        final_graph = (pynutil.insert('integer: "') + graph + pynutil.insert('"'))
        self.fst = self.add_tokens(final_graph).optimize()