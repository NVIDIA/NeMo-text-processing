# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2026 and onwards Google, Inc.
#
# Licensed under the Apache License, Version 2.0

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.te.graph_utils import (
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_space,
)


class FractionFst(GraphFst):
    def __init__(self):
        super().__init__(name="fraction", kind="verbalize")

        integer_part = (
            pynutil.delete('integer_part: "')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        optional_integer_part = pynini.closure(
            integer_part + delete_space + pynutil.insert(" "),
            0,
            1,
        )

        numerator = (
            pynutil.delete('numerator: "')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        denominator = (
            pynutil.delete('denominator: "')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        graph = (
            optional_integer_part
            + numerator
            + delete_space
            + pynutil.insert("/")
            + denominator
        )

        self.fst = self.delete_tokens(graph).optimize()