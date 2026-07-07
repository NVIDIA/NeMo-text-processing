# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2026 and onwards Google, Inc.
#
# Licensed under the Apache License, Version 2.0

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.te.graph_utils import (
    NEMO_NOT_SPACE,
    GraphFst,
)


class WordFst(GraphFst):
    """
    Finite state transducer for classifying plain Telugu tokens.
    """

    def __init__(self):
        super().__init__(name="word", kind="classify")
        word = (
            pynutil.insert('name: "')
            + pynini.closure(NEMO_NOT_SPACE, 1)
            + pynutil.insert('"')
        )
        self.fst = word.optimize()
