# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.fr.utils import get_abs_path


class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing fraction
            e.g. tokens { fraction { integer: "treinta y tres" numerator: "cuatro" denominator: "quinto" } } ->
        treinta y tres y cuatro quintos
    Args:
            deterministic: if True will provide a single transduction option,
                    for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, ordinal: GraphFst, deterministic: bool = True):
        super().__init__(name="fraction", kind="verbalize", deterministic=deterministic)

        numerator = pynutil.delete("numerator: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")

        denominator = ordinal.graph
        irregular_denominators = pynini.string_file(get_abs_path("data/fractions/irregular_denominators.tsv"))
        denominator = pynini.union(pynutil.add_weight(denominator @ irregular_denominators, -0.01), denominator)
        denominator = (pynini.cross("denominator:", "integer:") + pynini.closure(NEMO_SIGMA)) @ denominator

        irregular_half = pynini.closure(NEMO_SIGMA) + pynini.cross("et un demi", "et demi")

        optional_integer = pynutil.delete("integer_part: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        optional_integer = pynini.closure(
            optional_integer + pynini.accep(NEMO_SPACE) + pynutil.insert("et") + insert_space, 0, 1
        )

        optional_sign = pynini.closure(pynini.cross("negative: \"true\" ", "moins") + insert_space, 0, 1)

        final_graph = optional_sign + optional_integer + numerator + pynini.accep(NEMO_SPACE) + denominator
        final_graph = pynini.union(pynutil.add_weight(final_graph @ irregular_half, -0.01), final_graph)
        self.fst = self.delete_tokens(final_graph).optimize()
