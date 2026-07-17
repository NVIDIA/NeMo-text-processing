# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.ja.graph_utils import (
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_NARROW_NON_BREAK_SPACE,
    NEMO_NON_BREAKING_SPACE,
    NEMO_NOT_SPACE,
    NEMO_SIGMA,
    generator_main,
)
from nemo_text_processing.text_normalization.ja.utils import get_abs_path, load_labels
from nemo_text_processing.utils.logging import logger


class PostProcessingFst:
    """
    Finite state transducer that post-processing an entire sentence after verbalization is complete, e.g.
    removes extra spaces around punctuation marks
    " ( one hundred and twenty three ) " -> "(one hundred and twenty three)"

    Args:
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
    """

    def __init__(self, cache_dir: str = None, overwrite_cache: bool = False):

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(cache_dir, "ja_tn_post_processing.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["post_process_graph"]
            logger.info(f'Post processing graph was restored from {far_file}.')
        else:
            self.fst = self.get_punct_postprocess_graph()

            if far_file:
                generator_main(far_file, {"post_process_graph": self.fst})

    def get_punct_postprocess_graph(self):
        """
        Returns graph to post process Japanese TN output.

        Japanese verbalizers need ordinary inter-token spaces removed, but some
        classes intentionally use spaces internally. Protect those spaces as NBSP
        before deleting remaining technical spaces, then restore them as regular
        spaces in the final output.
        """

        protect_space = pynini.cross(" ", NEMO_NON_BREAKING_SPACE)
        ascii_char = NEMO_ALPHA | NEMO_DIGIT
        phone_digit = (
            pynini.project(pynini.string_file(get_abs_path("data/numbers/digit.tsv")), "output")
            | pynini.project(pynini.string_file(get_abs_path("data/numbers/zero.tsv")), "output")
            | pynini.project(pynini.string_file(get_abs_path("data/numbers/zero_decimal.tsv")), "output")
            | pynini.project(pynini.string_file(get_abs_path("data/numbers/zero_maru.tsv")), "output")
        ).optimize()
        space_sensitive_tokens = (
            pynini.project(pynini.string_file(get_abs_path("data/electronic/symbol.tsv")), "output")
            | pynini.project(pynini.string_file(get_abs_path("data/latin/letters.tsv")), "output")
            | pynini.project(pynini.string_file(get_abs_path("data/serial/words.tsv")), "output")
        ).optimize()
        title_tokens = pynini.union(
            *{spoken for _, spoken in load_labels(get_abs_path("data/whitelist_title.tsv"))}
        ).optimize()
        ten = dict(load_labels(get_abs_path("data/numbers/teen.tsv")))["10"]
        japanese_number = phone_digit | pynini.accep(ten)
        sentence_suffix = pynini.string_file(get_abs_path("data/post_processing/sentence_suffix.tsv"))
        collapse_double_space = pynini.cdrewrite(pynini.cross("  ", " "), "", "", pynini.closure(NEMO_SIGMA))

        protect_whitelist_internal_space = pynini.closure(NEMO_SIGMA)
        for spoken in {spoken for _, spoken in load_labels(get_abs_path("data/whitelist.tsv")) if " " in spoken}:
            parts = spoken.split()
            for left, right in zip(parts, parts[1:]):
                protect_whitelist_internal_space @= pynini.cdrewrite(
                    protect_space, left, right, pynini.closure(NEMO_SIGMA)
                )
        delete_ascii_inner_space = pynini.cdrewrite(
            pynutil.delete(" "), ascii_char, ascii_char, pynini.closure(NEMO_SIGMA)
        )
        protect_ascii_word_space = pynini.cdrewrite(
            protect_space, ascii_char**2, ascii_char**2, pynini.closure(NEMO_SIGMA)
        )
        protect_ascii_before_desu = pynini.cdrewrite(
            protect_space, ascii_char**2, sentence_suffix, pynini.closure(NEMO_SIGMA)
        )
        protect_ascii_before_japanese_number = pynini.cdrewrite(
            protect_space, ascii_char**2, japanese_number, pynini.closure(NEMO_SIGMA)
        )
        protect_title_before_ascii = pynini.cdrewrite(
            protect_space, title_tokens, ascii_char**2, pynini.closure(NEMO_SIGMA)
        )
        protect_after_space_sensitive_token = pynini.cdrewrite(
            protect_space, space_sensitive_tokens, "", pynini.closure(NEMO_SIGMA)
        )
        protect_before_space_sensitive_token = pynini.cdrewrite(
            protect_space, "", space_sensitive_tokens, pynini.closure(NEMO_SIGMA)
        )
        delete_technical_space = pynini.cdrewrite(
            pynutil.delete(" "), NEMO_NOT_SPACE, NEMO_NOT_SPACE, pynini.closure(NEMO_SIGMA)
        )
        restore_protected_space = pynini.cdrewrite(
            pynini.cross(pynini.union(NEMO_NON_BREAKING_SPACE, NEMO_NARROW_NON_BREAK_SPACE), " "),
            "",
            "",
            pynini.closure(NEMO_SIGMA),
        )

        graph = (
            collapse_double_space
            @ collapse_double_space
            @ protect_whitelist_internal_space
            @ protect_ascii_word_space
            @ delete_ascii_inner_space
            @ protect_ascii_before_desu
            @ protect_ascii_before_japanese_number
            @ protect_title_before_ascii
            @ protect_after_space_sensitive_token
            @ protect_before_space_sensitive_token
            @ delete_technical_space
            @ restore_protected_space
        ).optimize()

        return graph
