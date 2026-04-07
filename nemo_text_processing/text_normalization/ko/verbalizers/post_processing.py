# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.ko.graph_utils import NEMO_SIGMA, NEMO_SPACE, generator_main
from nemo_text_processing.utils.logging import logger


class PostProcessingFst:
    def __init__(self, cache_dir: str = None, overwrite_cache: bool = False):
        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(cache_dir, "ko_tn_post_processing.far")

        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["post_process_graph"]
            logger.info(f"Post processing graph was restored from {far_file}.")
        else:
            self.fst = self.get_postprocess_graph()
            if far_file:
                generator_main(far_file, {"post_process_graph": self.fst})

    def get_postprocess_graph(self):
        delete_space = pynutil.delete(NEMO_SPACE)

        vowel_final = pynini.union(
            "아", "야", "어", "여", "오", "요", "우", "유", "이", "애", "에",
            "사", "오", "구"
        )

        rule_i_to_ga = pynini.cdrewrite(
            delete_space + pynini.cross("이 ", "가 "),
            vowel_final,
            "",
            NEMO_SIGMA,
        )

        rule_eun_to_neun = pynini.cdrewrite(
            delete_space + pynini.cross("은 ", "는 "),
            vowel_final,
            "",
            NEMO_SIGMA,
        )

        rule_eul_to_reul = pynini.cdrewrite(
            delete_space + pynini.cross("을 ", "를 "),
            vowel_final,
            "",
            NEMO_SIGMA,
        )

        graph = rule_i_to_ga @ rule_eun_to_neun @ rule_eul_to_reul
        return graph.optimize()