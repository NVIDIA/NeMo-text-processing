# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright 2026 and onwards Google, Inc.
#
# Licensed under the Apache License, Version 2.0

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.te.graph_utils import (
    GraphFst,
    NEMO_NOT_QUOTE,
    delete_space,
)


class DecimalFst(GraphFst):
    """
    Verbalizes Telugu decimals.

    Examples:
        decimal { integer_part: "౧" fractional_part: "౨౩" }
            -> ౧.౨౩

        decimal { fractional_part: "౫" }
            -> .౫
    """

    def __init__(self):
        super().__init__(name="decimal", kind="verbalize")

        integer = (
            pynutil.delete("integer_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        optional_integer = pynini.closure(
            integer + delete_space,
            0,
            1,
        )

        fractional = (
            pynutil.insert(".")
            + pynutil.delete("fractional_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        graph = optional_integer + fractional

        delete_tokens = self.delete_tokens(graph)

        self.fst = delete_tokens.optimize()