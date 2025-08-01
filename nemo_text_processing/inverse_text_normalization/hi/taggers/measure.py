# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2024 and onwards Google, Inc.
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

from nemo_text_processing.inverse_text_normalization.hi.graph_utils import (
    GraphFst,
    convert_space,
    delete_extra_space,
    delete_space,
    insert_space,
)
from nemo_text_processing.inverse_text_normalization.hi.utils import get_abs_path, apply_fst


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure
        e.g. ऋण बारह किलोग्राम -> measure { decimal { negative: "true"  integer_part: "१२"  fractional_part: "५०"} units: "kg" }
        e.g. ऋण बारह किलोग्राम -> measure { cardinal { negative: "true"  integer_part: "१२"} units: "kg" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        measure: MeasureFst
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst):
        super().__init__(name="measure", kind="classify")

        cardinal_graph = cardinal.graph_no_exception
        decimal_graph = decimal.final_graph_wo_negative

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("ऋण", "\"true\"") + delete_extra_space, 0, 1,
        )

        measurements_graph = pynini.string_file(get_abs_path("data/measure/measurements.tsv")).invert()
        paune_graph = pynini.string_file(get_abs_path("data/numbers/paune.tsv")).invert()

        self.measurements = pynutil.insert("units: \"") + measurements_graph + pynutil.insert("\" ")
        graph_integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        graph_integer_paune = pynutil.insert("integer_part: \"") + paune_graph + pynutil.insert("\"")
        
        graph_saade_single_digit = pynutil.add_weight(pynutil.delete("साढ़े") + delete_space + graph_integer + delete_space + pynutil.insert(" fractional_part: \"५\""), 0.1)
        graph_sava_single_digit = pynutil.add_weight(pynutil.delete("सवा") + delete_space + graph_integer + delete_space + pynutil.insert(" fractional_part: \"२५\""), 0.1)
        graph_paune_single_digit = pynutil.add_weight(pynutil.delete("पौने") + delete_space + graph_integer_paune + delete_space + pynutil.insert(" fractional_part: \"७५\""), 1)
        graph_dedh_single_digit = pynutil.add_weight(pynini.union(pynutil.delete("डेढ़") | pynutil.delete("डेढ़")) + delete_space + pynutil.insert("integer_part: \"१\"") + delete_space + pynutil.insert(" fractional_part: \"५\""), 0.1)
        graph_dhaai_single_digit = pynutil.add_weight(pynutil.delete("ढाई") + delete_space + pynutil.insert("integer_part: \"२\"") + delete_space + pynutil.insert(" fractional_part: \"५\""), 1)
        
        graph_exceptions = graph_saade_single_digit | graph_sava_single_digit | graph_paune_single_digit | graph_dedh_single_digit | graph_dhaai_single_digit
        

        graph_measurements = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + decimal_graph
            + pynutil.insert(" }")
            + delete_extra_space
            + self.measurements
        )
        graph_measurements |= (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + cardinal_graph
            + pynutil.insert("\"")
            + pynutil.insert(" }")
            + delete_extra_space
            + self.measurements
        )
        graph_quarterly_measurements = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + graph_exceptions
            + pynutil.insert(" }")
            + delete_extra_space
            + self.measurements
        )
        graph_exception_bai = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + cardinal_graph
            + delete_space
            + pynini.cross("बाई", "x")
            + delete_space
            + cardinal_graph            
            + pynutil.insert("\"")
            + pynutil.insert(" }")
            + pynini.closure(delete_extra_space
            + self.measurements)
        )
        
        graph = graph_measurements | graph_quarterly_measurements | graph_exception_bai
        self.graph = graph.optimize()

        final_graph = self.add_tokens(graph)
        self.fst = final_graph
