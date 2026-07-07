# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2026 and onwards Google, Inc.
#
# Licensed under the Apache License, Version 2.0

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.te.graph_utils import (
    NEMO_CHAR,
    NEMO_SIGMA,
    GraphFst,
    delete_space,
)


class WordFst(GraphFst):
    """
    Finite state transducer for verbalizing plain Telugu tokens.
    """

    def __init__(self):
        super().__init__(name="word", kind="verbalize")
        chars = pynini.closure(NEMO_CHAR - " ", 1)
        char = (
            pynutil.delete("name:")
            + delete_space
            + pynutil.delete('"')
            + chars
            + pynutil.delete('"')
        )
        graph = char @ pynini.cdrewrite(
            pynini.cross(u"\u00a0", " "), "", "", NEMO_SIGMA
        )

        self.fst = graph.optimize()
