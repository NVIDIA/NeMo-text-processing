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

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.ko.graph_utils import NEMO_NOT_QUOTE, NEMO_SPACE, GraphFst, delete_space

class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing Korean fractions, e.g.
    tokens { fraction { numerator: "3" denominator: "5" } } → 5분의3
    tokens { fraction { integer_part: "2" numerator: "7" denominator: "9" } } → 2과 9분의7
    tokens { fraction { denominator: "√8" numerator: "4" } } → 루트8분의4
    tokens { fraction { denominator: "2.75" numerator: "125" } } → 2.75분의125
    tokens { fraction { negative: "마이너스" numerator: "10" denominator: "11" } } → 마이너스11분의10
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="fraction", kind="verbalize", deterministic=deterministic)

        # Handles square root symbols like "√3" → "루트3"
        denominator_root = pynini.cross("√", "루트") + pynutil.insert(NEMO_SPACE) + pynini.closure(NEMO_NOT_QUOTE)
        numerator_root = pynini.cross("√", "루트") + pynutil.insert(NEMO_SPACE) + pynini.closure(NEMO_NOT_QUOTE)

        # Matches non-root numeric content
        denominator = pynini.closure(NEMO_NOT_QUOTE - "√")
        numerator = pynini.closure(NEMO_NOT_QUOTE - "√")

        # Delete FST field: denominator and extract value
        denominator_component = (
            pynutil.delete('denominator: "') + (denominator_root | denominator) + pynutil.delete('"')
        )
        numerator_component = pynutil.delete('numerator: "') + (numerator_root | numerator) + pynutil.delete('"')

        # Match fraction form: "denominator + 분의 + numerator"
        # Also deletes optional morphosyntactic_features: "분의" if present
        graph_fraction = (
            denominator_component
            + pynutil.delete(NEMO_SPACE)
            + pynini.closure(
                pynutil.delete('morphosyntactic_features:') + delete_space + pynutil.delete('"분의"') + delete_space,
                0,
                1,
            )
            + pynutil.insert("분의")
            + pynutil.insert(NEMO_SPACE)
            + numerator_component
        )

        # Handle subject particle feature (분의_subject)
        # Insert default particle "이" (will be corrected later via rewrite rules)
        subject_suffix = (
            pynutil.delete(NEMO_SPACE)
            + pynutil.delete('morphosyntactic_features:')
            + delete_space
            + pynutil.delete('"분의_subject"')
            + delete_space
            + pynutil.insert("이")   # 일단 기본값
        )

        # Handle topic particle feature (분의_topic)
        topic_suffix = (
            pynutil.delete(NEMO_SPACE)
            + pynutil.delete('morphosyntactic_features:')
            + delete_space
            + pynutil.delete('"분의_topic"')
            + delete_space
            + pynutil.insert("은")
        )

        # Handle object particle feature (분의_object)
        object_suffix = (
            pynutil.delete(NEMO_SPACE)
            + pynutil.delete('morphosyntactic_features:')
            + delete_space
            + pynutil.delete('"분의_object"')
            + delete_space
            + pynutil.insert("을")
        )
        
        # Combine fraction + optional particle suffix
        # Particle is always inserted first in default form and later corrected
        graph_fraction_all = (
            graph_fraction
            + pynini.closure(subject_suffix | topic_suffix | object_suffix, 0, 1)
        )
               
        # Handle integer + fraction (e.g., "2과 3/4")
        # integer_part is removed and replaced with proper spacing
        graph_integer = (
            pynutil.delete('integer_part:')
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(pynini.union("√", ".", NEMO_NOT_QUOTE - '"'))
            + pynutil.delete('"')
            + pynutil.insert(NEMO_SPACE)
        )
        # Combine integer part with fraction
        graph_integer_fraction = graph_integer + delete_space + graph_fraction_all
        
        # Handle optional negative prefix (e.g., "마이너스")
        optional_sign = (
            pynutil.delete('negative:')
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_NOT_QUOTE - '"')
            + pynutil.delete('"')
            + delete_space
            + pynutil.insert(NEMO_SPACE)
        )

        # Final structure:
        # [optional negative] + (integer + fraction OR fraction only)
        graph = pynini.closure(optional_sign, 0, 1) + (graph_integer_fraction | graph_fraction_all)
        
        # Remove token wrappers
        final_graph = self.delete_tokens(graph)
        
        # Sigma for rewrite context (entire string)
        sigma = pynini.closure(NEMO_NOT_QUOTE | NEMO_SPACE)
        
        # Fix subject particle agreement (이 → 가 for vowel-ending numerals)
        # e.g., 사이 → 사가, 구이 → 구가
        subject_rewrite = pynini.cdrewrite(
            pynini.union(
                pynini.cross("이이", "이가"),
                pynini.cross("사이", "사가"),
                pynini.cross("오이", "오가"),
                pynini.cross("구이", "구가"),
            ),
            "",
            "",
            sigma,
        )
        
        # Fix topic particle agreement (은 → 는)
        # e.g., 이은 → 이는, 사은 → 사는
        topic_rewrite = pynini.cdrewrite(
            pynini.union(
                pynini.cross("이은", "이는"),
                pynini.cross("사은", "사는"),
                pynini.cross("오은", "오는"),
                pynini.cross("구은", "구는"),
            ),
            "",
            "",
            sigma,
        )

        # Fix object particle agreement (을 → 를)
        # e.g., 오을 → 오를, 이을 → 이를
        object_rewrite = pynini.cdrewrite(
            pynini.union(
                pynini.cross("이을", "이를"),
                pynini.cross("사을", "사를"),
                pynini.cross("오을", "오를"),
                pynini.cross("구을", "구를"),
            ),
            "",
            "",
            sigma,
        )
        
        # Apply all rewrite rules sequentially and final optimized FST
        final_graph = final_graph @ subject_rewrite @ topic_rewrite @ object_rewrite
        self.fst = final_graph.optimize()