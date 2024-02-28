# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.ar.graph_utils import GraphFst, convert_space, delete_extra_space
from nemo_text_processing.text_normalization.ar.taggers.measure import unit_singular


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure. Allows for plural form for unit.
        e.g. "عشرون  في المائة" -> measure { cardinal { integer: "20" } units: "%" }

    Args:
        itn_cardinal_tagger: ITN Cardinal tagger
        itn_decimal_tagger: ITN Decimal tagger
        itn_fraction_tagger: ITN Fraction tagger
    """

    def __init__(
        self,
        itn_cardinal_tagger: GraphFst,
        itn_decimal_tagger: GraphFst,
        itn_fraction_tagger: GraphFst,
        deterministic: bool = True,
    ):
        super().__init__(name="measure", kind="classify", deterministic=deterministic)

        cardinal_graph = itn_cardinal_tagger.graph

        graph_unit_singular = pynini.invert(unit_singular)
        unit = convert_space(graph_unit_singular)

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("سالب", "\"true\"") + delete_extra_space, 0, 1
        )

        unit = pynutil.insert("units: \"") + (unit) + pynutil.insert("\"")

        subgraph_decimal = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + itn_decimal_tagger.final_graph_wo_negative
            + pynutil.insert(" }")
            + delete_extra_space
            + unit
        )

        subgraph_fraction = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + pynutil.insert("integer_part: \"")
            + itn_fraction_tagger.graph
            + pynutil.insert("\" }")
            + delete_extra_space
            + unit
        )

        subgraph_cardinal = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + cardinal_graph
            + pynutil.insert("\"")
            + pynutil.insert(" }")
            + delete_extra_space
            + unit
        )
        final_graph = subgraph_cardinal | subgraph_decimal | subgraph_fraction
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
