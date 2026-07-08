# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2024 and onwards Google, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.kn.graph_utils import (
    NEMO_CHAR,
    NEMO_KN_DIGIT,
    NEMO_SIGMA,
    GraphFst,
)
from nemo_text_processing.inverse_text_normalization.kn.utils import get_abs_path


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinal

        e.g. ಹದಿಮೂರನೇ -> ordinal { integer: "13" }

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="ordinal", kind="classify")

        cardinal_graph = cardinal.graph_no_exception

        graph_digit = pynini.string_file(get_abs_path("data/ordinals/digit.tsv"))
        graph_teens = pynini.string_file(get_abs_path("data/ordinals/teens_and_ties.tsv"))
        graph_digit_hundred = pynini.string_file(get_abs_path("data/ordinals/hundred_digit.tsv"))

        graph = pynini.closure(NEMO_CHAR) + pynini.union(
            graph_digit,
            graph_teens,
            graph_digit_hundred,
            pynini.cross("ನೇ", "ನೇ"),
            pynini.cross("ನೆಯ", "ನೆಯ"),
            pynini.cross("ನೆಯದು", "ನೆಯದು"),
        )

        graph_fem_digit = pynini.string_file(get_abs_path("data/ordinals/digit_fem.tsv"))
        graph_fem_teens = pynini.string_file(get_abs_path("data/ordinals/teens_and_ties_fem.tsv"))
        graph_digit_hundred_fem = pynini.string_file(get_abs_path("data/ordinals/hundred_digit_fem.tsv"))

        graph_fem = pynini.closure(NEMO_CHAR) + pynini.union(
            graph_fem_digit,
            graph_fem_teens,
            graph_digit_hundred_fem,
            pynini.cross("ನೆಯ", "ನೆಯ"),
            pynini.cross("ನೆಯದು", "ನೆಯದು"),
        )

        graph = pynini.compose(
            graph | graph_fem,
            (
                cardinal_graph
                + pynini.union(
                    pynini.cross("ನೇ", "ನೇ"),
                    pynini.cross("ನೆಯ", "ನೆಯ"),
                    pynini.cross("ನೆಯದು", "ನೆಯದು"),
                )
            ),
        ).optimize()

        morph_features_graph = pynini.string_file(get_abs_path("data/ordinals/morph_features.tsv"))

        morpho_graph = pynutil.insert("\" morphosyntactic_features: \"") + morph_features_graph + pynutil.insert("\"")

        rule = pynini.cdrewrite(
            morpho_graph,
            pynini.closure(NEMO_KN_DIGIT),
            pynini.union("[EOS]", " "),
            NEMO_SIGMA,
        )

        final_graph = pynutil.insert("integer: \"") + graph @ rule

        self.final_graph = self.add_tokens(final_graph)
        self.fst = self.final_graph.optimize()
