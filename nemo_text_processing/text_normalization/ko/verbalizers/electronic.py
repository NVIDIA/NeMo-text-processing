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
from pynini.examples import plurals
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.ko.graph_utils import (
    NEMO_ALPHA,
    NEMO_CHAR,
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    NEMO_SPACE,
    NEMO_DIGIT,
    GraphFst,
    delete_extra_space,
    delete_space,
    insert_space,
    
)
from nemo_text_processing.text_normalization.ko.utils import get_abs_path

class ElectronicFst(GraphFst):
    """
    전자식 텍스트 읽기(verbalize):
      tokens { electronic { username: "cdf1" domain: "abc.edu" } }
      -> c d f 일 골뱅이 a b c 닷 e d u  (정책에 따라 다름)

    deterministic=True: 단일 출력
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="verbalize", deterministic=deterministic)

        # 1. 숫자 (0~9): ko/data/number/digit.tsv
        # (BUG FIX 1: 'invert' 제거)
        graph_digit_no_zero = pynini.string_file(
            get_abs_path("data/number/digit.tsv")
        ).optimize()
        
        graph_zero = pynini.cross("0", "영")
        if not deterministic:
            graph_zero |= pynini.cross("0", "공")
        graph_digit = (graph_digit_no_zero | graph_zero).optimize()

        # 2. 심볼: ko/data/electronic/symbol.tsv (예: ".\t점")
        graph_symbols = pynini.string_file(
            get_abs_path("data/electronic/symbol.tsv")
        ).optimize()

        NEMO_NOT_BRACKET = pynini.difference(
            NEMO_CHAR, pynini.union("{", "}")
        ).optimize()

        # 3. 기본 띄어쓰기 규칙 (Fallback용)
        # (BUG FIX 2: 'NEMO_ALPHA' 추가)
        default_chars_symbols = pynini.cdrewrite(
            pynutil.insert(" ")
            + (graph_symbols | graph_digit | NEMO_ALPHA)
            + pynutil.insert(" "),
            "",
            "",
            NEMO_SIGMA,
        )
        default_chars_map = pynini.compose(
            pynini.closure(NEMO_NOT_BRACKET), default_chars_symbols
        ).optimize()

        # 4. username (띄어쓰기 규칙 적용)
        user_name = (
            pynutil.delete("username:")
            + delete_space
            + pynutil.delete('"')
            + default_chars_map
            + pynutil.delete('"')
        )

        # 5. domain (domain.tsv 우선 적용)
        # [수정] domain.tsv 파일을 로드
        domain_common_pairs = pynini.string_file(
            get_abs_path("data/electronic/domain.tsv")
        ).optimize()

        # ".com", ".co.kr" 같은 패턴을 원문 어디서든 우선 치환
        tld_rewrite = pynini.cdrewrite(
            domain_common_pairs,  # 입력: ".com"  출력: "닷컴"  등
            "",                   # 왼쪽 문맥 없음
            "",                   # 오른쪽 문맥 없음
            NEMO_SIGMA,           # 전체 문맥에서 적용
        )

        add_space_before_dot = pynini.cdrewrite(
            pynini.cross("닷", " 닷"),                 # '닷' -> ' 닷'
            (NEMO_ALPHA | NEMO_DIGIT | NEMO_CHAR),    # 왼쪽 문맥: 글자/숫자/한글 등
            "",
            NEMO_SIGMA,
        )

        domain = (
            pynutil.delete("domain:")
            + delete_space
            + pynutil.delete('"')
            + (tld_rewrite @ add_space_before_dot)
            + delete_space
            + pynutil.delete('"')
        ).optimize()
        
        # 6. protocol (태거에서 이미 변환된 값을 그대로 출력)
        protocol = (
            pynutil.delete('protocol: "')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        # 7. 조합: (옵션)프로토콜 + (옵션)이메일 아이디 + 도메인
        graph = (
            pynini.closure(protocol + delete_space, 0, 1)
            + pynini.closure(
                user_name + delete_space + pynutil.insert(" 골뱅이 ") + delete_space, 0, 1
            )
            + domain
            + delete_space
        ).optimize() @ pynini.cdrewrite(
            delete_extra_space, "", "", NEMO_SIGMA
        )  # 최종 여분 공백 제거

        self.fst = self.delete_tokens(graph).optimize()