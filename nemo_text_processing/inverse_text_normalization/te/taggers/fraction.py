# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2026 and onwards Google, Inc.
#
# Licensed under the Apache License, Version 2.0

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.te.graph_utils import (
    GraphFst,
    NEMO_SIGMA,
)
class FractionFst(GraphFst):
    def __init__(self, cardinal: GraphFst):
        super().__init__(name="fraction", kind="classify")

        graph_cardinal = cardinal.graph_no_exception
        

        integer = (
            pynutil.insert('integer_part: "')
            + graph_cardinal
            + pynutil.insert('" ')
        )

        numerator = (
            pynutil.insert('numerator: "')
            + graph_cardinal
            + pynutil.insert('"')
        )

        denominator = (
            pynutil.insert(' denominator: "')
            + pynutil.add_weight(graph_cardinal, -10.0)
            + pynutil.insert('"')
        )

        delete_bai = pynutil.delete(" బై ")
        delete_mariyu = pynutil.delete(" మరియు ")

        graph_fraction = numerator + delete_bai + denominator

        graph_half = (
            pynutil.delete("అర")
            + pynutil.insert('numerator: "౧" denominator: "౨"')
        )

        graph_quarter = (
            pynutil.delete("పావు")
            + pynutil.insert('numerator: "౧" denominator: "౪"')
        )

        graph_three_quarter = (
            pynutil.delete("ముప్పావు")
            + pynutil.insert('numerator: "౩" denominator: "౪"')
        )

        graph_lexical_fraction = (
            graph_half
            | graph_quarter
            | graph_three_quarter
        ).optimize()

        graph_mixed_fraction = (
            integer
            + delete_mariyu
            + (graph_fraction | graph_lexical_fraction)
        )

        graph_one_half = (
            pynutil.delete("ఒకటిన్నర")
            + pynutil.insert('integer_part: "౧" numerator: "౧" denominator: "౨"')
        )

        graph_two_half = (
            pynutil.delete("రెండున్నర")
            + pynutil.insert('integer_part: "౨" numerator: "౧" denominator: "౨"')
        )

        graph_three_half = (
            pynutil.delete("మూడున్నర")
            + pynutil.insert('integer_part: "౩" numerator: "౧" denominator: "౨"')
        )

        graph_eight_half = (
            pynutil.delete("ఎనిమిదిన్నర")
            + pynutil.insert('integer_part: "౮" numerator: "౧" denominator: "౨"')
        )

        graph_spoken_mixed = (
            graph_one_half
            | graph_two_half
            | graph_three_half
            | graph_eight_half
        ).optimize()

        graph = (
            graph_fraction
            | graph_mixed_fraction
            | graph_lexical_fraction
            | graph_spoken_mixed
        ).optimize()

        self.graph = graph
        self.final_graph_wo_negative = graph
        self.fst = self.add_tokens(graph).optimize()