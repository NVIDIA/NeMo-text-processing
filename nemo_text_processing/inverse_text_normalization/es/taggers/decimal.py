# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from nemo_text_processing.inverse_text_normalization.es.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_CASED,
    INPUT_LOWER_CASED,
    MIN_NEG_WEIGHT,
    NEMO_DIGIT,
    NEMO_SIGMA,
    TO_LOWER,
    GraphFst,
    capitalized_input_graph,
    delete_extra_space,
    delete_space,
)
from nemo_text_processing.text_normalization.es.graph_utils import ES_MINUS


def get_quantity(
    decimal: 'pynini.FstLike', cardinal_up_to_million: 'pynini.FstLike', input_case: str = INPUT_LOWER_CASED
) -> 'pynini.FstLike':
    """
    Returns FST that transforms either a cardinal or decimal followed by a quantity into a numeral,
    e.g. one million -> integer_part: "1" quantity: "million"
    e.g. one point five million -> integer_part: "1" fractional_part: "5" quantity: "million"

    Args:
        decimal: decimal FST
        cardinal_up_to_million: cardinal FST
        input_case: accepting either "lower_cased" or "cased" input.
    """
    numbers = cardinal_up_to_million @ (
        pynutil.delete(pynini.closure("0")) + pynini.difference(NEMO_DIGIT, "0") + pynini.closure(NEMO_DIGIT)
    )

    suffix_labels = [
        "millón",
        "millones",
        "millardo",
        "millardos",
        "billón",
        "billones",
        "trillón",
        "trillones",
        "cuatrillón",
        "cuatrillones",
    ]
    suffix = pynini.union(*suffix_labels)

    if input_case == INPUT_CASED:
        suffix |= pynini.union(*[x[0].upper() + x[1:] for x in suffix_labels]).optimize()

    res = (
        pynutil.insert("integer_part: \"")
        + numbers
        + pynutil.insert("\"")
        + delete_extra_space
        + pynutil.insert("quantity: \"")
        + suffix
        + pynutil.insert("\"")
    )
    res |= decimal + delete_extra_space + pynutil.insert("quantity: \"") + suffix + pynutil.insert("\"")
    return res


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal
        Decimal point is either "." or ",", determined by whether "punto" or "coma" is spoken.
            e.g. menos uno coma dos seis -> decimal { negative: "true" integer_part: "1" morphosyntactic_features: "," fractional_part: "26" }
            e.g. menos uno punto dos seis -> decimal { negative: "true" integer_part: "1" morphosyntactic_features: "." fractional_part: "26" }

        This decimal rule assumes that decimals can be pronounced as:
        (a cardinal) + ('coma' or 'punto') plus (any sequence of cardinals <1000, including 'zero')

        Also writes large numbers in shortened form, e.g.
            e.g. uno coma dos seis millón -> decimal { negative: "false" integer_part: "1" morphosyntactic_features: "," fractional_part: "26" quantity: "millón" }
            e.g. dos millones -> decimal { negative: "false" integer_part: "2" quantity: "millones" }
            e.g. mil ochocientos veinticuatro millones -> decimal { negative: "false" integer_part: "1824" quantity: "millones" }
    Args:
        cardinal: CardinalFst
        input_case: accepting either "lower_cased" or "cased" input.

    """

    def __init__(self, cardinal: GraphFst, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="decimal", kind="classify")

        # number after decimal point can be any series of cardinals <1000, including 'zero'
        graph_decimal = cardinal.numbers_up_to_thousand
        graph_decimal = pynini.closure(graph_decimal + delete_space) + graph_decimal
        self.graph = graph_decimal.optimize()

        # decimal point can be denoted by 'coma' or 'punto'
        decimal_point = pynini.cross("coma", "morphosyntactic_features: \",\"")
        decimal_point |= pynini.cross("punto", "morphosyntactic_features: \".\"")

        if input_case == INPUT_CASED:
            decimal_point |= pynini.cross("Coma", "morphosyntactic_features: \",\"")
            decimal_point |= pynini.cross("Punto", "morphosyntactic_features: \".\"")

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross(ES_MINUS, "\"true\"") + delete_extra_space, 0, 1
        )

        graph_fractional = pynutil.insert("fractional_part: \"") + graph_decimal + pynutil.insert("\"")

        cardinal_graph = cardinal.graph_no_exception | pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        graph_integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        final_graph_wo_sign = (
            pynini.closure(graph_integer + delete_extra_space, 0, 1)
            + decimal_point
            + delete_extra_space
            + graph_fractional
        )
        final_graph = optional_graph_negative + final_graph_wo_sign

        self.final_graph_wo_negative = (
            final_graph_wo_sign
            | get_quantity(final_graph_wo_sign, cardinal.numbers_up_to_million, input_case=input_case).optimize()
        )

        # accept semiotic spans that start with a capital letter
        self.final_graph_wo_negative |= pynutil.add_weight(
            pynini.compose(TO_LOWER + NEMO_SIGMA, self.final_graph_wo_negative), MIN_NEG_WEIGHT
        ).optimize()

        quantity_graph = get_quantity(final_graph_wo_sign, cardinal.numbers_up_to_million, input_case=input_case)
        final_graph |= optional_graph_negative + quantity_graph

        if input_case == INPUT_CASED:
            final_graph |= capitalized_input_graph(final_graph)

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
