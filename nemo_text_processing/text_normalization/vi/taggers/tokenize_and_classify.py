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

from nemo_text_processing.text_normalization.vi.graph_utils import (
    NEMO_SPACE,
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
)
from nemo_text_processing.text_normalization.vi.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.vi.taggers.date import DateFst
from nemo_text_processing.text_normalization.vi.taggers.decimal import DecimalFst
from nemo_text_processing.text_normalization.vi.taggers.fraction import FractionFst
from nemo_text_processing.text_normalization.vi.taggers.measure import MeasureFst
from nemo_text_processing.text_normalization.vi.taggers.money import MoneyFst
from nemo_text_processing.text_normalization.vi.taggers.ordinal import OrdinalFst
from nemo_text_processing.text_normalization.vi.taggers.punctuation import PunctuationFst
from nemo_text_processing.text_normalization.vi.taggers.range import RangeFst
from nemo_text_processing.text_normalization.vi.taggers.roman import RomanFst
from nemo_text_processing.text_normalization.vi.taggers.time import TimeFst
from nemo_text_processing.text_normalization.vi.taggers.whitelist import WhiteListFst
from nemo_text_processing.text_normalization.vi.taggers.word import WordFst
from nemo_text_processing.text_normalization.vi.verbalizers.cardinal import CardinalFst as VCardinalFst
from nemo_text_processing.text_normalization.vi.verbalizers.date import DateFst as VDateFst
from nemo_text_processing.text_normalization.vi.verbalizers.decimal import DecimalFst as VDecimalFst
from nemo_text_processing.text_normalization.vi.verbalizers.fraction import FractionFst as VFractionFst
from nemo_text_processing.text_normalization.vi.verbalizers.measure import MeasureFst as VMeasureFst
from nemo_text_processing.text_normalization.vi.verbalizers.money import MoneyFst as VMoneyFst
from nemo_text_processing.text_normalization.vi.verbalizers.time import TimeFst as VTimeFst
from nemo_text_processing.utils.logging import logger


class ClassifyFst(GraphFst):
    def __init__(
        self,
        input_case: str,
        deterministic: bool = True,
        cache_dir: str = None,
        overwrite_cache: bool = False,
        whitelist: str = None,
    ):
        super().__init__(name="tokenize_and_classify", kind="classify", deterministic=deterministic)

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(
                cache_dir,
                f"vi_tn_{deterministic}_deterministic_{input_case}_tokenize.far",
            )
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            logger.info(f"ClassifyFst.fst was restored from {far_file}.")
        else:
            logger.info(f"Creating Vietnamese ClassifyFst grammars.")

            cardinal = CardinalFst(deterministic=deterministic)
            cardinal_graph = cardinal.fst

            punctuation = PunctuationFst(deterministic=deterministic)
            punct_graph = punctuation.fst

            whitelist = WhiteListFst(input_case=input_case, deterministic=deterministic)
            whitelist_graph = whitelist.fst

            word_graph = WordFst(deterministic=deterministic).fst

            ordinal = OrdinalFst(cardinal=cardinal, deterministic=deterministic)
            ordinal_graph = ordinal.fst

            decimal = DecimalFst(cardinal=cardinal, deterministic=deterministic)
            decimal_graph = decimal.fst

            fraction = FractionFst(cardinal=cardinal, deterministic=deterministic)
            fraction_graph = fraction.fst

            date = DateFst(cardinal=cardinal, deterministic=deterministic)
            date_graph = date.fst

            roman = RomanFst(cardinal=cardinal, deterministic=deterministic)
            roman_graph = roman.fst

            time_fst = TimeFst(cardinal=cardinal, deterministic=deterministic)
            time_graph = time_fst.fst

            money = MoneyFst(cardinal=cardinal, decimal=decimal, deterministic=deterministic)
            money_graph = money.fst

            measure = MeasureFst(cardinal=cardinal, decimal=decimal, fraction=fraction, deterministic=deterministic)
            measure_graph = measure.fst

            # Create composed verbalizers for range processing
            v_cardinal = VCardinalFst(deterministic=deterministic)
            v_date = VDateFst(deterministic=deterministic)
            date_final = pynini.compose(date_graph, v_date.fst)

            v_decimal = VDecimalFst(v_cardinal, deterministic=deterministic)
            decimal_final = pynini.compose(decimal_graph, v_decimal.fst)

            v_time = VTimeFst(deterministic=deterministic)
            time_final = pynini.compose(time_graph, v_time.fst)

            v_money = VMoneyFst(deterministic=deterministic)
            money_final = pynini.compose(money_graph, v_money.fst)

            v_fraction = VFractionFst(deterministic=deterministic)
            v_measure = VMeasureFst(
                decimal=v_decimal, cardinal=v_cardinal, fraction=v_fraction, deterministic=deterministic
            )
            measure_final = pynini.compose(measure_graph, v_measure.fst)

            # Create range graph
            range_fst = RangeFst(
                time=time_final,
                date=date_final,
                decimal=decimal_final,
                money=money_final,
                measure=measure_final,
                deterministic=deterministic,
            )
            range_graph = range_fst.fst

            classify = (
                pynutil.add_weight(whitelist_graph, 1.01)
                | pynutil.add_weight(money_graph, 1.1)
                | pynutil.add_weight(range_graph, 1.1)
                | pynutil.add_weight(decimal_graph, 1.1)
                | pynutil.add_weight(date_graph, 1.1)
                | pynutil.add_weight(cardinal_graph, 1.1)
                | pynutil.add_weight(ordinal_graph, 1.1)
                | pynutil.add_weight(fraction_graph, 1.1)
                | pynutil.add_weight(time_graph, 1.1)
                | pynutil.add_weight(measure_graph, 1.1)
                | pynutil.add_weight(word_graph, 100)
                | pynutil.add_weight(roman_graph, 101)
            )
            punct = (
                pynutil.insert("tokens { ") + pynutil.add_weight(punct_graph, 2.1) + pynutil.insert(" }")
            )  # Lower priority than semantic classes
            token = pynutil.insert("tokens { ") + classify + pynutil.insert(" }")
            token_plus_punct = (
                pynini.closure(punct + pynutil.insert(NEMO_SPACE))
                + token
                + pynini.closure(pynutil.insert(NEMO_SPACE) + punct)
            )

            graph = token_plus_punct + pynini.closure((delete_extra_space).ques + token_plus_punct)
            graph = delete_space + graph + delete_space

            self.fst = graph.optimize()

            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})
