# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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

from nemo_text_processing.inverse_text_normalization.hy.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_LOWER_CASED,
    NEMO_SIGMA,
    GraphFst,
    convert_space,
    delete_extra_space,
)


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money
        e.g. քսան հազար դրամ -> tokens { money { integer_part: "20000" currency: "֏" } }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        input_case: accepting either "lower_cased" or "cased" input.
        (input_case is not necessary everything is made for lower_cased input)
        TODO add cased input support
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="money", kind="classify")
        # quantity, integer_part, fractional_part, currency

        cardinal_graph = cardinal.graph_no_exception
        graph_decimal_final = decimal.final_graph_wo_negative
        unit = pynini.string_file(get_abs_path("data/currency.tsv"))
        unit_singular = pynini.invert(unit)

        graph_unit_singular = pynutil.insert("currency: \"") + convert_space(unit_singular) + pynutil.insert("\"")

        graph_integer = (
            pynutil.insert("integer_part: \"")
            + (NEMO_SIGMA @ cardinal_graph)
            + pynutil.insert("\"")
            + delete_extra_space
            + graph_unit_singular
        )
        graph_decimal = graph_decimal_final + delete_extra_space + graph_unit_singular
        final_graph = graph_integer | graph_decimal

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
