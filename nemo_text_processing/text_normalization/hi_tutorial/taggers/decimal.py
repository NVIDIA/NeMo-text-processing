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
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    NEMO_NOT_QUOTE,
    TO_UPPER,
    get_abs_path,
    GraphFst,
    delete_space,
    insert_space,
)

from pynini.lib import pynutil

#delete_space = pynutil.delete(" ")
zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
teens_and_ties = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv"))
hundred = pynini.string_file(get_abs_path("data/numbers/hundred.tsv"))
thousands = pynini.string_file(get_abs_path("data/numbers/thousands.tsv"))

def get_quantity(decimal: 'pynini.FstLike', cardinal_up_to_hundred: 'pynini.FstLike') -> 'pynini.FstLike':
    """
    Returns FST that transforms either a cardinal or decimal followed by a quantity into a numeral,
    e.g. १ लाख -> integer_part: "एक" quantity: "लाख"
    e.g. १.५ लाख -> integer_part: "एक" fractional_part: "पाँच" quantity: "लाख"

    Args: 
        decimal: decimal FST
        cardinal_up_to_hundred: cardinal FST
    """

    numbers = cardinal_up_to_hundred

    res = (
        pynutil.insert("integer_part: \"")
        + numbers
        + pynutil.insert("\"")
        + pynini.accep(" ")
        + pynutil.insert("quantity: \"")
        + thousands
        + pynutil.insert("\"")
    )
    res |= decimal + pynini.accep(" ") + pynutil.insert("quantity: \"") + quantities + pynutil.insert("\"")
    return res

class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal, e.g. 
        -१२.५००६ अरब -> decimal {negative: "true" integer_part: "बारह"  fractional_part: "पाँच शून्य शून्य छह" quantity: "अरब"}
        १ अरब -> decimal {integer_part: "एक" quantity: "अरब"}

    cardinal: CardinalFst
    """
    def __init__(self, cardinal: GraphFst, deterministic: bool):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)   



      









