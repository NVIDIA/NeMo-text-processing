# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.hi.graph_utils import GraphFst, delete_space, insert_space
from nemo_text_processing.text_normalization.hi.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.hi.taggers.decimal import DecimalFst
from nemo_text_processing.text_normalization.hi.utils import apply_fst, get_abs_path


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure, suppletive aware, e.g. 
        -१२kg -> measure { negative: "true" cardinal { integer: "बारह" } units: "किलोग्राम" }
        -१२.२kg -> measure { decimal { negative: "true"  integer_part: "बारह"  fractional_part: "दो"} units: "किलोग्राम" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst):
        super().__init__(name="measure", kind="classify")

        cardinal_graph = cardinal.final_graph
        decimal_graph = decimal.final_graph_wo_negative
        unit_graph = pynini.string_file(get_abs_path("data/measure/unit.tsv"))

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\"") + insert_space, 0, 1,
        )

        # Define the unit handling
        unit = pynutil.insert("units: \"") + unit_graph + pynutil.insert("\" ")

        # Handling symbols like x, X, *, -
        symbol_graph = pynini.string_map(
            [
                ("x", "बाई"),
                ("X", "बाई"),
                ("*", "बाई"),
                # ("-", "से")
            ]
        )

        # Units requiring special fractional handling
        dynamic_units = pynini.string_map(
            [
                ("हफ़्ता", "हफ़्ता"),
                ("month", "महीना"),
                ("months", "महीने"),
                ("yr", "साल"),
                ("doz", "दर्जन"),
                ("सप्ताह", "सप्ताह"),
                ("min", "मिनट"),
                ("s", "सेकंड"),
                ("d", "दिन"),
            ]
        )

        # Fractional exceptions with dynamic units
        fractional_exceptions_with_units = (
            pynutil.insert("measure { decimal { ")
            + optional_graph_negative
            + pynutil.insert("integer_part: \"")
            + decimal.fractional_exceptions  # Apply fractional exceptions
            + pynutil.insert("\" } units: \"")
            + dynamic_units
            + pynutil.insert("\" }")
        )

        print(decimal.fractional_exceptions)
        print(apply_fst(decimal.fractional_exceptions, "१.५"))

        graph_measurements = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + decimal_graph
            + pynutil.insert(" }")
            + delete_space
            + unit
        )

        graph_measurements |= (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + cardinal_graph
            + pynutil.insert("\"")
            + pynutil.insert(" }")
            + delete_space
            + unit
        )

        # Handling cardinal clubbed with symbol as single token
        graph_measurements |= (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + cardinal_graph
            + pynutil.insert("\"")
            + pynutil.insert(" }")
            + pynutil.insert(" units: \"")
            + symbol_graph
            + pynutil.insert("\" ")
            + pynutil.insert("} }")
            + insert_space
            + pynutil.insert("tokens { cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + cardinal_graph
            + pynutil.insert("\"")
        )

        graph = fractional_exceptions_with_units | graph_measurements
        self.graph = graph.optimize()

        final_graph = self.add_tokens(graph)
        self.fst = final_graph
