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


from nemo_text_processing.text_normalization.zh.utils import get_abs_path
from nemo_text_processing.text_normalization.zh.graph_utils import GraphFst, NEMO_NOT_QUOTE

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class MoneyFst(GraphFst):
    def __init__(self, cardinal: GraphFst, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="classify", deterministic=deterministic)
     

        super().__init__(name="money", kind="classify")
            
        cardinal = cardinal.just_cardinals
        decimal = decimal.decimal

        currency = pynini.string_file (get_abs_path("data/money/currency_major.tsv"))
        currency_mandarin = pynini.string_file (get_abs_path("data/money/currency_mandarin.tsv"))
        
        # regular money gramamr with currency symbols $1000
        currency_component = pynutil.insert("currency: \"") + currency + pynutil.insert("\"") 
        number_component = pynutil.insert("integer: \"") + cardinal + pynutil.insert("\"")
        graph_regular_money = currency_component + pynutil.insert(" ") + number_component 
        
        # 块 元 毛 with optional symbols
        unit_major = pynini.accep("块") | pynini.accep("元") | pynini.closure(pynini.cross("塊", "块"),1)
        unit_minor = pynini.accep("角") | pynini.accep("毛") 
        unit_minor_alt = pynini.accep("分") 
        currency_mandarin_component = pynutil.insert("mandarin_currency: \"") + currency_mandarin + pynutil.insert("\"")
        unit_components = (pynutil.insert("major_unit: \"") + unit_major + pynutil.insert("\"")) | (pynutil.insert("minor_unit: \"") + unit_minor + pynutil.insert("\"")) | (pynutil.insert("minor_alt_unit: \"") + unit_minor_alt + pynutil.insert("\""))

        graph_unit_only = (number_component + pynutil.insert(" ") + unit_components + pynini.closure(pynutil.insert(" ") + currency_mandarin_component,0,1)) | (number_component + pynutil.insert(" ") + unit_components + pynini.closure(pynutil.insert(" ") + currency_mandarin_component,0,1)) | (number_component + pynutil.insert(" ") + unit_components  + pynini.closure(pynutil.insert(" ") + currency_mandarin_component,0,1))
        graph_units = pynini.closure(graph_unit_only, 1,3)
        graph_unit_money = graph_unit_only | graph_units
     
        # only currency part as mandarins
        graph_mandarin_money = number_component + pynutil.insert(" ") + currency_mandarin_component
        
        # larger money as decimals
        graph_decimal_money = (decimal + pynutil.insert(" ") + currency_mandarin_component) | (currency_component + pynutil.insert(" ") + decimal)
        
        graph = graph_regular_money | graph_unit_money | graph_mandarin_money | graph_decimal_money
        
        # range part
        morpheme = pynini.accep("到") | pynini.accep("至") | pynini.closure(pynini.cross("-", "到")) | pynini.closure(pynini.cross("——", "到")) | pynini.closure(pynini.cross("~", "到"))
        range_component = (pynutil.insert("range: \"") + morpheme + pynutil.insert("\"")) 
        graph_range = (pynutil.insert("integer_pre: \"") + cardinal + pynutil.insert("\"") + pynutil.insert(" ") + range_component + pynutil.insert(" ") + pynutil.insert("integer_suf: \"") + cardinal + pynutil.insert("\"") + pynutil.insert(" ") + (currency_mandarin_component | unit_components)) | (currency_component + pynutil.insert(" ") + pynutil.insert("integer_pre: \"") + cardinal + pynutil.insert("\"") + pynutil.insert(" ") + range_component + pynutil.insert(" ") + pynutil.insert("integer_suf: \"") + cardinal + pynutil.insert("\"")) 
        
        final_graph = graph | graph_range

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

