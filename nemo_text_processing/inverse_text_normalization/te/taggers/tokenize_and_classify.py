# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2026 and onwards Google, Inc.
#
# Licensed under the Apache License, Version 2.0

import logging
import os

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.te.graph_utils import (
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
)
from nemo_text_processing.inverse_text_normalization.te.taggers.cardinal import CardinalFst


class ClassifyFst(GraphFst):
    """
    Final classification grammar for Telugu ITN cardinal processing.
    """

    def __init__(
        self,
        cache_dir: str = None,
        overwrite_cache: bool = False,
        whitelist: str = None,
        input_case: str = None,
    ):
        super().__init__(name="tokenize_and_classify", kind="classify")

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(cache_dir, "te_itn.far")

        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            logging.info(f"ClassifyFst.fst was restored from {far_file}.")
        else:
            logging.info("Creating Telugu ClassifyFst grammar.")

            cardinal = CardinalFst()
            cardinal_graph = cardinal.fst

            classify = pynutil.add_weight(cardinal_graph, 1.1)

            token = (
                pynutil.insert("tokens { ")
                + classify
                + pynutil.insert(" }")
            )

            graph = token + pynini.closure(
                pynutil.add_weight(delete_extra_space + token, 1000.0)
            )
            graph = delete_space + graph + delete_space

            self.fst = graph.optimize()

            if far_file:
                generator_main(
                    far_file,
                    {"tokenize_and_classify": self.fst},
                )
                logging.info(
                    f"ClassifyFst grammars are saved to {far_file}."
                )