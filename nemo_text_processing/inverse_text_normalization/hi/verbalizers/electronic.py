# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.hi.graph_utils import (
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_space,
)


class ElectronicFst(GraphFst):
    """
    ITN verbalizer for electronic expressions.
    All fields pass content through unchanged.
    """

    def __init__(self):
        super().__init__(name="electronic", kind="verbalize")

        def field_graph(field_name: str) -> pynini.Fst:
            return (
                pynutil.delete(f"{field_name}:")
                + delete_space
                + pynutil.delete("\"")
                + pynini.closure(NEMO_NOT_QUOTE, 1)
                + pynutil.delete("\"")
            )

        ip_graph       = field_graph("ip")
        domain_graph   = field_graph("domain")
        username_graph = field_graph("username")
        path_graph     = field_graph("path")

        email_graph = (
            username_graph
            + pynutil.insert("@")
            + delete_space
            + domain_graph
        )

        # email before domain (both use domain: field)
        graph = ip_graph | email_graph | path_graph | domain_graph

        self.fst = self.delete_tokens(graph).optimize()