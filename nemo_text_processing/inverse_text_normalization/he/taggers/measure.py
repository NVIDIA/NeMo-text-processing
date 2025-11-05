# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.he.graph_utils import GraphFst
from nemo_text_processing.inverse_text_normalization.he.taggers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.he.taggers.decimal import DecimalFst
from nemo_text_processing.inverse_text_normalization.he.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_SPACE,
    delete_extra_space,
    delete_space,
    delete_zero_or_one_space,
    insert_space,
)


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure in Hebrew
        e.g. מש עשרה אחוז -> measure { cardinal { integer: "15" } units: "%" }
        e.g. מינוס חמש עשרה אחוז -> measure { cardinal { negative: "-"  integer: "15" } units: "%" }
        e.g. שלוש מיליגרם -> measure { cardinal { integer: "3" } units: "מ״ג" }
        e.g. אלף אחוז -> measure { cardinal { integer: "1000" } units: "%" }
        e.g. אחוז אחד -> measure { units: "%" cardinal { integer: "1" } }
        e.g. סנטימטר אחד -> measure { units: "ס״מ" cardinal { integer: "1" } }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
    """

    def __init__(self, cardinal: CardinalFst, decimal: DecimalFst):
        super().__init__(name="measure", kind="classify")

        # optional negative sign
        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("מינוס", '"-"') + NEMO_SPACE,
            0,
            1,
        )

        prefix_graph = pynini.string_file(get_abs_path("data/prefix.tsv"))
        optional_prefix_graph = pynini.closure(
            pynutil.insert('morphosyntactic_features: "') + prefix_graph + pynutil.insert('"') + insert_space,
            0,
            1,
        )

        # cardinal numbers
        cardinal_graph = cardinal.graph_no_exception

        # Let singular apply to values > 1 as they could be part of an adjective phrase (e.g. 14 foot tall building)
        subgraph_decimal = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + decimal.final_graph_wo_sign
            + pynutil.insert(" }")
            + delete_extra_space
        )

        subgraph_cardinal = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert('integer: "')
            + cardinal_graph
            + pynutil.insert('"')
            + pynutil.insert(" }")
            + delete_extra_space
        )

        # convert units
        joined_units = pynini.string_file(get_abs_path("data/measurements.tsv"))
        joined_units = pynini.invert(joined_units)
        joined_units = pynutil.insert('units: "') + joined_units + pynutil.insert('"')

        spaced_units = pynini.string_file(get_abs_path("data/spaced_measurements.tsv"))
        spaced_units = pynini.invert(spaced_units)
        spaced_units = pynutil.insert('units: "\[SPACE\]') + spaced_units + pynutil.insert('"')  # noqa: W605

        # in joint units the unit is concatenated to the number, in spaced unit separate the unit with a space
        units_graph = joined_units | spaced_units

        # one graph is needed since it changed the order of the words.
        # We say "ten percent" for 10% but "percent one" for 1%
        one = pynini.string_map([("אחד", "1")])
        one_graph = (
            insert_space
            + pynutil.insert("cardinal { ")
            + pynutil.insert('integer: "')
            + one
            + pynutil.insert('"')
            + pynutil.insert(" }")
        )

        number_graph = subgraph_decimal | subgraph_cardinal
        number_unit_graph = (number_graph + units_graph) | (units_graph + delete_space + one_graph)

        final_graph = optional_prefix_graph + number_unit_graph + delete_zero_or_one_space
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
