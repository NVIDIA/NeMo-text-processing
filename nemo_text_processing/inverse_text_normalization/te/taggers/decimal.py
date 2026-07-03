# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2026 and onwards Google, Inc.
#
# Licensed under the Apache License, Version 2.0

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.te.graph_utils import (
    GraphFst,
    delete_extra_space,
    delete_space,
)
from nemo_text_processing.inverse_text_normalization.te.utils import get_abs_path


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying Telugu decimals.

    Examples:
        ఒకటి దశాంశం రెండు మూడు
        -> decimal { integer_part: "౧" fractional_part: "౨౩" }

        దశాంశం ఐదు
        -> decimal { fractional_part: "౫" }
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="decimal", kind="classify")

        cardinal_graph = cardinal.graph_no_exception

        graph_digit = pynini.string_file(
            get_abs_path("data/numbers/digit.tsv")
        ).invert()

        graph_zero = pynini.string_file(
            get_abs_path("data/numbers/zero.tsv")
        ).invert()

        graph_decimal_digit = graph_digit | graph_zero

        graph_decimal_digits = (
            pynini.closure(graph_decimal_digit + delete_space)
            + graph_decimal_digit
        )

        point = pynini.union(
            pynutil.delete("దశాంశం"),
            pynutil.delete("పాయింట్"),
        )

        graph_integer = (
            pynutil.insert('integer_part: "')
            + cardinal_graph
            + pynutil.insert('"')
        )

        graph_fractional = (
            pynutil.insert(' fractional_part: "')
            + graph_decimal_digits
            + pynutil.insert('"')
        )

        final_graph_wo_negative = (
            pynini.closure(graph_integer + delete_extra_space, 0, 1)
            + point
            + delete_extra_space
            + graph_fractional
        )

        self.graph = final_graph_wo_negative
        self.final_graph_wo_negative = final_graph_wo_negative

        final_graph = self.add_tokens(final_graph_wo_negative)
        self.fst = final_graph.optimize()