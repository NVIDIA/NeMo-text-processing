# Copyright (c) 2026 NVIDIA CORPORATION.  All rights reserved.
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
from nemo_text_processing.inverse_text_normalization.te.verbalizers.verbalize import VerbalizeFst


class VerbalizeFinalFst(GraphFst):
    """
    Verbalizes full Telugu ITN token output.

    Example:
        tokens { cardinal { integer: "౨౩" } } -> ౨౩
        tokens { name: "పరుగులు" } -> పరుగులు
    """

    def __init__(self):
        super().__init__(name="verbalize_final", kind="verbalize")

        verbalize = VerbalizeFst().fst

        graph = (
            pynutil.delete("tokens")
            + delete_space
            + pynutil.delete("{")
            + delete_space
            + verbalize
            + delete_space
            + pynutil.delete("}")
        )

        graph = delete_space + graph + pynini.closure(delete_extra_space + graph) + delete_space

        self.fst = graph.optimize()