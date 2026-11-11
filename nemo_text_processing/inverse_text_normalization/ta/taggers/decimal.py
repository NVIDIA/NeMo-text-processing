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

from nemo_text_processing.inverse_text_normalization.ta.graph_utils import GraphFst, delete_extra_space, delete_space
from nemo_text_processing.inverse_text_normalization.ta.utils import get_abs_path


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal numbers in Tamil, e.g.
        நூற்று ஒன்று புள்ளி ஆறு -> decimal { integer_part: "௧௦௧" fractional_part: "௬" }
        இருநூற்று ஆறு புள்ளி இரண்டு ஒன்பது -> decimal { integer_part: "௨௦௬" fractional_part: "௨௯" }
        எதிர்மறை ஒன்று புள்ளி ஐந்து -> decimal { negative: "true" integer_part: "௧" fractional_part: "௫" }

    The integer part is parsed with the cardinal grammar; the digits after "புள்ளி"
    are read one by one (zero variants come from zero.tsv).

    Args:
        cardinal: CardinalFst, used for the integer part
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="decimal", kind="classify")

        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert()
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert()

        # Integer part: reuse the full cardinal graph if the cardinal class exposes it,
        # otherwise assemble it from the sub-graphs CardinalFst always exposes.
        cardinal_graph = getattr(cardinal, "graph_no_exception", None)
        if cardinal_graph is None:
            cardinal_graph = (
                graph_zero
                | graph_digit
                | cardinal.graph_two_digit
                | cardinal.graph_exact_hundreds
                | cardinal.graph_hundred_with_remainder
                | cardinal.graph_exact_thousands
                | cardinal.graph_thousand_with_remainder
                | cardinal.graph_lakh
                | cardinal.graph_lakh_with_remainder
                | cardinal.graph_crore
                | cardinal.graph_crore_with_remainder
            ).optimize()

        # Digits after the decimal point, spoken one at a time: "இரண்டு ஒன்பது" -> "௨௯"
        graph_single_digit = (graph_zero | graph_digit).optimize()
        graph_fraction_digits = (pynini.closure(graph_single_digit + delete_space) + graph_single_digit).optimize()

        # =====================================================
        # Decimal point marker
        # =====================================================
        point = pynutil.delete("à®ªà¯à®³à¯à®³à®¿")

        optional_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("எதிர்மறை", '"true"') + delete_extra_space,
            0,
            1,
        )

        integer_part = pynutil.insert('integer_part: "') + cardinal_graph + pynutil.insert('"')
        fractional_part = pynutil.insert('fractional_part: "') + graph_fraction_digits + pynutil.insert('"')

        # =====================================================
        # Final decimal graph
        # Integer part is mandatory (all blind tests have it)
        # =====================================================
        self.final_graph_wo_negative = (
            integer_part + delete_space + point + delete_extra_space + fractional_part
        ).optimize()

        final_graph = optional_negative + self.final_graph_wo_negative
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
