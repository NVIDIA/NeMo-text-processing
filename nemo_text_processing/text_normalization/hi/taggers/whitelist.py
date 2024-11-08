# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.hi.graph_utils import (
    INPUT_LOWER_CASED,
    NEMO_UPPER,
    GraphFst,
    convert_space,
)
from nemo_text_processing.text_normalization.hi.utils import (
    augment_labels_with_punct_at_end,
    get_abs_path,
    load_labels,
)


class WhiteListFst(GraphFst):
    def __init__(self, input_case: str, deterministic: bool = True, input_file: str = None):
        super().__init__(name="whitelist", kind="classify", deterministic=deterministic)

        def _get_whitelist_graph(input_case, file, keep_punct_add_end: bool = False):
            whitelist = load_labels(file)
            if input_case == INPUT_LOWER_CASED:
                whitelist = [[x.lower(), y] for x, y in whitelist]
            else:
                whitelist = [[x, y] for x, y in whitelist]

            if keep_punct_add_end:
                whitelist.extend(augment_labels_with_punct_at_end(whitelist))

            graph = pynini.string_map(whitelist)
            return graph

        graph = _get_whitelist_graph(input_case, get_abs_path("data/whitelist/abbreviations.tsv"))

        if deterministic:
            graph |= graph.optimize()
        else:
            graph |= _get_whitelist_graph(
                input_case, get_abs_path("data/whitelist/abbreviations.tsv"), keep_punct_add_end=True
            )

        for x in [".", ". "]:
            graph |= (
                NEMO_UPPER
                + pynini.closure(pynutil.delete(x) + NEMO_UPPER, 2)
                + pynini.closure(pynutil.delete("."), 0, 1)
            )

        if input_file:
            whitelist_provided = _get_whitelist_graph(input_case, input_file)
            if not deterministic:
                graph |= whitelist_provided
            else:
                graph = whitelist_provided

        self.graph = (convert_space(graph)).optimize()

        self.fst = (pynutil.insert("name: \"") + self.graph + pynutil.insert("\"")).optimize()
