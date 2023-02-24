# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
from nemo_text_processing.text_normalization.zh.graph_utils import delete_space, GraphFst, NEMO_NOT_QUOTE
from nemo_text_processing.text_normalization.zh.utils import get_abs_path 
from pynini.lib import pynutil


class MoneyFst(GraphFst):
    """
    Finite state transducer for verbalizing fraction e.g.
        tokens { money { integer: "二" currency: "$"} } -> 二美元
        tokens { money { integer: "三" major_unit: "块"} } -> 三块
        tokens { money { currency: "$" integer: "二" } } -> 二美元
    """

    def __init__(self, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="verbalize", deterministic=deterministic)
        
        # mandarin currency data imported
        currencies = pynini.string_file (get_abs_path("data/money/currency_mandarin.tsv"))
        
        # components to combine to make graphs
        number_component = pynutil.delete("integer: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        currency_component = pynutil.delete("currency: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        mandarin_currency_component = pynutil.delete("mandarin_currency: \"") + pynini.closure(currencies) + pynutil.delete("\"")
        decimal_component  = decimal.decimal_component
        unit_only_component= (pynutil.delete("major_unit: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")) | (pynutil.delete("minor_unit: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")) | (pynutil.delete("minor_alt_unit: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\""))
        
        # graphs 
        graph_regular_money = number_component + delete_space + (currency_component | mandarin_currency_component)
        graph_unit_money = pynini.closure((number_component + delete_space + unit_only_component + delete_space)) + pynini.closure(mandarin_currency_component)
        graph_decimal_money = decimal_component + delete_space + ((currency_component | mandarin_currency_component)) 
        
        graph = graph_unit_money | graph_regular_money | graph_decimal_money
        
        # ranges
        range_component = (pynutil.delete("range: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\""))
        graph_range = pynutil.delete("integer_pre: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"") + delete_space + range_component + delete_space + pynutil.delete("integer_suf: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"") + delete_space + (mandarin_currency_component | currency_component | unit_only_component)

        final_graph = graph | pynutil.add_weight(graph_range, -1.0)
        

        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()