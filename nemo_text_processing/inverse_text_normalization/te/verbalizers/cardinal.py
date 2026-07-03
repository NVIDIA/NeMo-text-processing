# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.te.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space


class CardinalFst(GraphFst):
    """
    Finite state transducer for verbalizing cardinal
        e.g.
        cardinal { integer: "౨౩" } -> ౨౩
        cardinal { integer: "౨౩" negative: "-" } -> -౨౩
    """

    def __init__(self):
        super().__init__(name="cardinal", kind="verbalize")

        optional_sign = pynini.closure(
            pynutil.delete("negative:")
            + delete_space
            + pynutil.delete("\"")
            + NEMO_NOT_QUOTE
            + pynutil.delete("\"")
            + delete_space,
            0,
            1,
        )

        graph = (
            pynutil.delete("integer:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        self.numbers = graph
        graph = optional_sign + graph

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()