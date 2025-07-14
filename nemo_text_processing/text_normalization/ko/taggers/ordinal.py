# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.ko.graph_utils import GraphFst
from nemo_text_processing.text_normalization.ko.utils import get_abs_path


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying Korean ordinal expressions, e.g.
    1번째 -> ordinal { integer: "첫번째" }

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        # Load base .tsv files
        graph_digit = pynini.string_file(get_abs_path("data/ordinal/digit.tsv"))
        graph_zero = pynini.string_file(get_abs_path("data/number/zero.tsv"))
        graph_exceptions = pynini.string_file(get_abs_path("data/ordinal/exceptions.tsv"))
        graph_tens = pynini.string_file(get_abs_path("data/ordinal/tens.tsv"))
        graph_tens_prefix = pynini.string_file(get_abs_path("data/ordinal/tens_prefix.tsv"))

        graph_11_to_39 = (graph_tens_prefix + graph_digit).optimize()

        # Combine all ordinal forms from 1 to 39
        graph_ordinal_1to39 = (
            graph_exceptions | graph_digit | graph_zero | graph_tens | graph_11_to_39
        ).optimize() + pynini.accep("번째")

        # Accept tens digit 4–9
        tens_digit_4_to_9_accep = pynini.union(*[pynini.accep(str(i)) for i in range(4, 10)])
        # Accept any single digit
        any_single_digit_accep = pynini.union(*[pynini.accep(str(i)) for i in range(0, 10)])
        # Combine two digits
        from_40_to_99_inputs = tens_digit_4_to_9_accep + any_single_digit_accep

        # Match numbers with 3 or more digits
        input_100_plus = pynini.closure(any_single_digit_accep, 3)

        # Combine both ranges (40–99 and 100+): total range = 40 and above
        filter_inputs_from_40 = (from_40_to_99_inputs | input_100_plus).optimize()

        # Only allow cardinal numbers that are 40 or more
        graph_cardinal_from40_filtered = pynini.compose(filter_inputs_from_40, cardinal.graph)

        # Add "번째" to the filtered cardinal graph.
        graph_ordinal_from40 = graph_cardinal_from40_filtered + pynini.accep("번째")

        graph_ordinal = (graph_ordinal_1to39 | graph_ordinal_from40).optimize()  # Handles 1-39  # Handles 40+

        final_graph = pynutil.insert('integer: "') + graph_ordinal + pynutil.insert('"')
        self.fst = self.add_tokens(final_graph).optimize()
