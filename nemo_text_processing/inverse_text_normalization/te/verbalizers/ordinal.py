
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.te.graph_utils import (
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_space,
)


class OrdinalFst(GraphFst):
    def __init__(self):
        super().__init__(name="ordinal", kind="verbalize")

        integer = (
            pynutil.delete("integer:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        morph = (
            delete_space
            + pynutil.delete("morphosyntactic_features:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        graph = integer + morph

        self.fst = self.delete_tokens(graph).optimize()

