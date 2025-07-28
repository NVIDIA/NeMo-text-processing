# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.es.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_CASED,
    INPUT_LOWER_CASED,
    GraphFst,
    convert_space,
)
from nemo_text_processing.text_normalization.en.utils import load_labels


class WhiteListFst(GraphFst):
    """
    Finite state transducer for classifying whitelisted tokens
        e.g. usted -> tokens { name: "ud." }
    This class has highest priority among all classifier grammars.

    Whitelisted tokens are defined and loaded from "data/whitelist.tsv" (unless input_file specified).

    Args:
        input_case: accepting either "lower_cased" or "cased" input.
        input_file: path to a file with whitelist replacements (each line of the file: written_form\tspoken_form\n),
        e.g. nemo_text_processing/inverse_text_normalization/es/data/whitelist.tsv
    """

    def __init__(self, input_case: str = INPUT_LOWER_CASED, input_file: str = None):
        super().__init__(name="whitelist", kind="classify")

        def get_whitelist_graph(input_file: str):
            labels = load_labels(input_file)

            if input_case == INPUT_CASED:
                additional_labels = []
                for written, spoken in labels:
                    written_capitalized = written[0].upper() + written[1:]
                    additional_labels.extend(
                        [
                            [written_capitalized, spoken.capitalize()],  # first letter capitalized
                            [
                                written_capitalized,
                                spoken.upper().replace(" Y ", " y "),
                            ],  # # add pairs with the all letters capitalized
                        ]
                    )

                    spoken_no_space = spoken.replace(" ", "")
                    # add abbreviations without spaces (both lower and upper case), i.e. "BMW" not "B M W"
                    if len(spoken) == (2 * len(spoken_no_space) - 1):
                        additional_labels.extend(
                            [[written, spoken_no_space], [written_capitalized, spoken_no_space.upper()]]
                        )

                labels += additional_labels

            whitelist = pynini.string_map(labels).invert().optimize()
            return whitelist

        if input_file is None:
            input_file = get_abs_path("data/whitelist.tsv")

        if not os.path.exists(input_file):
            raise ValueError(f"Whitelist file {input_file} not found")

        whitelist = get_whitelist_graph(input_file)

        graph = pynutil.insert("name: \"") + convert_space(whitelist) + pynutil.insert("\"")
        self.fst = graph.optimize()
