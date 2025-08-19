# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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


import os
import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.ger.taggers.cardinal import (
    CardinalFst,
)
from nemo_text_processing.inverse_text_normalization.ger.taggers.ordinal import (
    OrdinalFst,
)
from nemo_text_processing.inverse_text_normalization.ger.taggers.decimal import (
    DecimalFst,
)
from nemo_text_processing.inverse_text_normalization.ger.taggers.fraction import (
    FractionFst,
)
from nemo_text_processing.inverse_text_normalization.ger.taggers.date import (
    DateFst,
)
from nemo_text_processing.inverse_text_normalization.ger.taggers.time import (
    TimeFst,
)
from nemo_text_processing.inverse_text_normalization.ger.taggers.money import (
    MoneyFst,
)
from nemo_text_processing.inverse_text_normalization.ger.taggers.measure import (
    MeasureFst,
)
from nemo_text_processing.inverse_text_normalization.ger.taggers.telephone import (
    TelephoneFst,
)
from nemo_text_processing.inverse_text_normalization.ger.taggers.punctuation import (
    PunctuationFst,
)
from nemo_text_processing.inverse_text_normalization.ger.taggers.word import WordFst
from nemo_text_processing.inverse_text_normalization.ger.graph_utils import (
    INPUT_LOWER_CASED,
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
)


class ClassifyFst(GraphFst):
    def __init__(
        self,
        cache_dir: str = None,
        overwrite_cache: bool = False,
        input_case: str = INPUT_LOWER_CASED,
    ):
        super().__init__(name="tokenize_and_classify", kind="classify")

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(cache_dir, f"ger_itn_{input_case}.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
        else:
            cardinal = CardinalFst()
            cardinal_graph = cardinal.fst

            ordinal = OrdinalFst(cardinal)
            ordinal_graph = ordinal.fst

            decimal = DecimalFst(cardinal)
            decimal_graph = decimal.fst

            fraction = FractionFst(cardinal)
            fraction_graph = fraction.fst

            date = DateFst(cardinal, ordinal)
            date_graph = date.fst

            time = TimeFst(cardinal)
            time_graph = time.fst

            money = MoneyFst(cardinal)
            money_graph = money.fst

            measure = MeasureFst(cardinal, decimal, fraction)
            measure_graph = measure.fst

            telephone = TelephoneFst(cardinal)
            telephone_graph = telephone.fst

            word_graph = WordFst().fst
            punct_graph = PunctuationFst().fst

            classify = (
                pynutil.add_weight(cardinal_graph, 1.0)
                | pynutil.add_weight(ordinal_graph, 1.1)
                | pynutil.add_weight(decimal_graph, 1.1)
                | pynutil.add_weight(fraction_graph, 1.1)
                | pynutil.add_weight(date_graph, 1.11)
                | pynutil.add_weight(time_graph, 1.12)
                | pynutil.add_weight(money_graph, 1.1)
                | pynutil.add_weight(measure_graph, 1.1)
                | pynutil.add_weight(telephone_graph, 1.1)
                | pynutil.add_weight(word_graph, 100)
            )

            punct = (
                pynutil.insert("tokens { ")
                + pynutil.add_weight(punct_graph, weight=1.1)
                + pynutil.insert(" }")
            )

            token = pynutil.insert("tokens { ") + classify + pynutil.insert(" }")

            token_plus_punct = (
                pynini.closure(punct + pynutil.insert(" "))
                + token
                + pynini.closure(pynutil.insert(" ") + punct)
            )

            graph = token_plus_punct + pynini.closure(
                delete_extra_space + token_plus_punct
            )
            graph = delete_space + graph + delete_space

            self.fst = graph.optimize()

            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})
