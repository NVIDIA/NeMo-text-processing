
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.te.graph_utils import (
    NEMO_CHAR,
    GraphFst,
)
from nemo_text_processing.inverse_text_normalization.te.utils import get_abs_path


class OrdinalFst(GraphFst):
    def __init__(self, cardinal: GraphFst):
        super().__init__(name="ordinal", kind="classify")

        cardinal_graph = cardinal.graph_no_exception

        graph_digit = pynini.string_file(
            get_abs_path("data/ordinals/digit.tsv")
        )

        graph_teens = pynini.string_file(
            get_abs_path("data/ordinals/teens_and_ties.tsv")
        )

        graph_hundred = pynini.string_file(
            get_abs_path("data/ordinals/hundred_digit.tsv")
        )

        ordinal_tail = pynini.union(
            graph_digit,
            graph_teens,
            graph_hundred,
        ).optimize()

        graph = pynini.compose(
            pynini.closure(NEMO_CHAR) + ordinal_tail,
            cardinal_graph,
        ).optimize()

        final_graph = (
            pynutil.insert('integer: "')
            + graph
            + pynutil.insert('"')
            + pynutil.insert(' morphosyntactic_features: "వ"')
        )

        self.fst = self.add_tokens(final_graph).optimize()

