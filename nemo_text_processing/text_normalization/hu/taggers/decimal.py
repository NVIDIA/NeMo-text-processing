# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright (c) 2023, Jim O'Regan.
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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, NEMO_SIGMA, GraphFst, insert_space
from nemo_text_processing.text_normalization.hu.utils import get_abs_path, load_labels, naive_inflector

quantities = load_labels(get_abs_path("data/number/quantities.tsv"))


def inflect_quantities():
    output = []
    for quantity in quantities:
        if len(quantity) == 2:
            output.append((quantity[0], quantity[1]))
            output += naive_inflector(quantity[0], quantity[1], True)
        else:
            output.append((quantity[0], quantity[0]))
            tmp = naive_inflector(".", quantity[0], True)
            real = [t[1] for t in tmp]
            output += [(t, t) for t in real]
            if "lli" in quantity[0]:
                output.append((quantity[0].replace("lli", "li"), quantity[0]))
                orth = [(x.replace("lli", "li"), x) for x in real]
                output += orth
    return output


def get_quantity(decimal: 'pynini.FstLike', cardinal_up_to_hundred: 'pynini.FstLike') -> 'pynini.FstLike':
    """
    Returns FST that transforms either a cardinal or decimal followed by a quantity into a numeral,
    e.g. 1 millió -> integer_part: "egy" quantity: "millió"
    e.g. 1,4 million -> integer_part: "egy" fractional_part: "négy" quantity: "millió"

    Args:
        decimal: decimal FST
        cardinal_up_to_hundred: cardinal FST
    """
    numbers = cardinal_up_to_hundred
    quant_fst = pynini.string_map(inflect_quantities())

    res = (
        pynutil.insert("integer_part: \"")
        + numbers
        + pynutil.insert("\"")
        + pynini.accep(" ")
        + pynutil.insert("quantity: \"")
        + quant_fst
        + pynutil.insert("\"")
    )
    res |= decimal + pynini.accep(" ") + pynutil.insert("quantity: \"") + quant_fst + pynutil.insert("\"")
    return res


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal, e.g.
        -11,4006 milliárd -> decimal { negative: "true" integer_part: "tizenegy"  fractional_part: "négyezer-hat tízezred" quantity: "milliárd" preserve_order: true }
        1 milliárd -> decimal { integer_part: "egy" quantity: "milliárd" preserve_order: true }
    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)
        cardinal_graph = cardinal.graph
        digit_no_zero = NEMO_DIGIT - "0"
        digit_or_del_zero = pynutil.delete("0") | digit_no_zero
        final_zero = pynini.closure(pynutil.delete("0"))

        # In Hungarian, the fraction is read as a whole number
        # with a word for the decimal place added
        # see: https://helyesiras.mta.hu/helyesiras/default/numerals
        decimal_number = digit_no_zero @ cardinal_graph + final_zero + pynutil.insert(" tized")
        decimal_number |= (digit_or_del_zero + NEMO_DIGIT) @ cardinal_graph + final_zero + pynutil.insert(" század")
        order = 2
        for decimal_name in [
            "ezred",
            "milliomod",
            "milliárdod",
            "billiomod",
            "billiárdod",
            "trilliomod",
            "trilliárdod",
        ]:
            for modifier in ["", "tíz", "száz"]:
                decimal_number |= (
                    (NEMO_DIGIT ** order + (NEMO_DIGIT - "0"))
                    @ pynini.cdrewrite(pynini.cross("0", ""), "[BOS]", "", NEMO_SIGMA)
                    @ cardinal_graph
                    + final_zero
                    + pynutil.insert(f" {modifier}{decimal_name}")
                )
                order += 1
        if not deterministic:
            alts = pynini.string_map([("billiomod", "ezer milliárdod"), ("billiárdod", "millió milliárdod")])
            decimal_alts = decimal_number @ pynini.cdrewrite(alts, "", "[EOS]", NEMO_SIGMA)
            decimal_number |= decimal_alts

        self.graph = decimal_number

        point = pynutil.delete(",")
        optional_graph_negative = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        self.graph_fractional = pynutil.insert("fractional_part: \"") + self.graph + pynutil.insert("\"")
        self.graph_integer = pynutil.insert("integer_part: \"") + cardinal.graph + pynutil.insert("\"")
        final_graph_wo_sign = self.graph_integer + point + insert_space + self.graph_fractional

        self.final_graph_wo_negative = final_graph_wo_sign | get_quantity(
            final_graph_wo_sign, cardinal.graph_hundreds_component_at_least_one_non_zero_digit
        )
        final_graph = optional_graph_negative + self.final_graph_wo_negative
        final_graph += pynutil.insert(" preserve_order: true")

        final_graph = self.add_tokens(final_graph)

        self.fst = final_graph.optimize()
