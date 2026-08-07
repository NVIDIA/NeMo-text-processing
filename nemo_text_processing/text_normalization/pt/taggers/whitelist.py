# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.en.utils import augment_labels_with_punct_at_end
from nemo_text_processing.text_normalization.pt.graph_utils import (
    INPUT_CASED,
    INPUT_LOWER_CASED,
    NEMO_SIGMA,
    NEMO_UPPER,
    GraphFst,
    convert_space,
)
from nemo_text_processing.text_normalization.pt.utils import get_abs_path, load_labels


def _empty_fst() -> "pynini.FstLike":
    """FST that accepts nothing (no whitelist rows)."""
    return pynini.intersect(pynini.accep("a"), pynini.accep("b")).optimize()


def get_formats(input_f, input_case=INPUT_CASED, is_default=True):
    """Abbreviation format variants (same idea as EN whitelist)."""
    multiple_formats = load_labels(input_f)
    if not multiple_formats:
        return _empty_fst()
    additional_options = []
    for x, y in multiple_formats:
        if input_case == INPUT_LOWER_CASED:
            x = x.lower()
        additional_options.append((f"{x}.", y))
        additional_options.append((f"{x[0].upper() + x[1:]}", f"{y[0].upper() + y[1:]}"))
        additional_options.append((f"{x[0].upper() + x[1:]}.", f"{y[0].upper() + y[1:]}"))
    multiple_formats.extend(additional_options)

    if not is_default:
        multiple_formats = [(x, f"|raw_start|{x}|raw_end||norm_start|{y}|norm_end|") for (x, y) in multiple_formats]

    return pynini.string_map(multiple_formats)


class WhiteListFst(GraphFst):
    """
    Whitelist classifier for pt-BR TN. Data lives under pt/data/whitelist/ (may be empty).
    """

    def __init__(self, input_case: str, deterministic: bool = True, input_file: str = None):
        super().__init__(name="whitelist", kind="classify", deterministic=deterministic)

        def _get_whitelist_graph(input_case, file, keep_punct_add_end: bool = False):
            whitelist = load_labels(file) if os.path.isfile(file) else []
            if not whitelist:
                return _empty_fst()
            if input_case == INPUT_LOWER_CASED:
                whitelist = [[x.lower(), y] for x, y in whitelist]
            else:
                whitelist = [[x, y] for x, y in whitelist]

            if keep_punct_add_end:
                whitelist.extend(augment_labels_with_punct_at_end(whitelist))

            return pynini.string_map(whitelist)

        graph = _get_whitelist_graph(input_case, get_abs_path("data/whitelist/tts.tsv"))

        symbol_path = get_abs_path("data/whitelist/symbol.tsv")
        if os.path.isfile(symbol_path) and load_labels(symbol_path):
            graph |= pynini.compose(
                pynini.difference(NEMO_SIGMA, pynini.accep("/")).optimize(),
                _get_whitelist_graph(input_case, symbol_path),
            ).optimize()

        for x in [".", ". "]:
            graph |= (
                NEMO_UPPER
                + pynini.closure(pynutil.delete(x) + NEMO_UPPER, 2)
                + pynini.closure(pynutil.delete("."), 0, 1)
            )

        if not deterministic:
            alt_path = get_abs_path("data/whitelist/alternatives.tsv")
            if os.path.isfile(alt_path) and load_labels(alt_path):
                graph |= _get_whitelist_graph(input_case, alt_path, keep_punct_add_end=True)
            fmt_path = get_abs_path("data/whitelist/alternatives_all_format.tsv")
            if os.path.isfile(fmt_path) and load_labels(fmt_path):
                graph |= get_formats(fmt_path, input_case=input_case)

        if input_file:
            whitelist_provided = _get_whitelist_graph(input_case, input_file)
            if not deterministic:
                graph |= whitelist_provided
            else:
                graph = whitelist_provided

        self.graph = convert_space(graph).optimize()
        self.fst = (pynutil.insert("name: \"") + self.graph + pynutil.insert("\"")).optimize()
