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

# Copyright (c) 2025, NVIDIA
# Licensed under the Apache License, Version 2.0

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.ko.graph_utils import (
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_NOT_SPACE,
    NEMO_SIGMA,
    GraphFst,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.ko.utils import get_abs_path


class ElectronicFst(GraphFst):
    """
    전자식 텍스트 분류 (이메일/URL/카드 큐 등)
      예) abc@nvidia.co.kr
        -> tokens { electronic { username: "abc" domain: "nvidia.co.kr" } }
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="electronic", kind="classify", deterministic=deterministic)

        # ---------- 기본 렌지/심볼 ----------
        LOWER = pynini.union(*[pynini.accep(c) for c in "abcdefghijklmnopqrstuvwxyz"])
        UPPER = pynini.union(*[pynini.accep(c) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"])
        ASCII_LETTER = (LOWER | UPPER).optimize()
        ASCII_ALNUM = (ASCII_LETTER | NEMO_DIGIT).optimize()

        HYPHEN = pynini.accep("-")
        DOT = pynini.accep(".")
        SLASH = pynini.accep("/")
        AT = pynini.accep("@")

        # 숫자 읽기 모드
        numbers = NEMO_DIGIT if deterministic else (pynutil.insert(" ") + cardinal.long_numbers + pynutil.insert(" "))

        # 리소스 로드
        cc_cues = pynini.string_file(get_abs_path("data/electronic/cc_cues.tsv"))
        accepted_symbols = pynini.project(pynini.string_file(get_abs_path("data/electronic/symbol.tsv")), "input")
        accepted_common_domains = pynini.project(
            pynini.string_file(get_abs_path("data/electronic/domain.tsv")), "input"
        )
        graph_symbols = pynini.string_file(get_abs_path("data/electronic/symbol.tsv")).optimize()

        # ---------- username ----------
        # '@'는 username에 포함되면 안 되므로 제외
        username_symbols = pynini.difference(accepted_symbols, AT)
        # 영문/숫자로 시작 + (영문/숫자/허용심볼/숫자읽기) 반복
        username_core = ASCII_ALNUM + pynini.closure(ASCII_ALNUM | numbers | username_symbols)
        username = pynutil.insert('username: "') + username_core + pynutil.insert('"') + pynini.cross("@", " ")

        # ---------- domain ----------
        # RFC 단순화: label = [A-Za-z0-9-]+ , TLD = '.' [A-Za-z0-9]{2,}
        label = pynini.closure(ASCII_ALNUM | HYPHEN, 1)
        tld = DOT + pynini.closure(ASCII_ALNUM, 2)
        # 전체 도메인: (label + (tld)+) 또는 (tld만)  → ".com" 같은 케이스 허용
        domain_core = (label + pynini.closure(tld, 1)) | tld

        # 도메인 뒤 경로(/...) 1회 옵션
        domain_with_opt_path = domain_core + pynini.closure(SLASH + pynini.closure(NEMO_NOT_SPACE, 1), 0, 1)

        domain_graph_with_class_tags = (
            pynutil.insert('domain: "') + domain_with_opt_path.optimize() + pynutil.insert('"')
        )

        # ---------- protocol ----------
        protocol_symbols = pynini.closure((graph_symbols | pynini.cross(":", "colon")) + pynutil.insert(" "))
        protocol_start = (pynini.cross("https", "HTTPS ") | pynini.cross("http", "HTTP ")) + (
            pynini.accep("://") @ protocol_symbols
        )
        protocol_file_start = pynini.accep("file") + insert_space + (pynini.accep(":///") @ protocol_symbols)
        protocol_end = pynutil.add_weight(pynini.cross("www", "WWW ") + pynini.accep(".") @ protocol_symbols, -1000)
        protocol = protocol_file_start | protocol_start | protocol_end | (protocol_start + protocol_end)
        protocol = pynutil.insert('protocol: "') + protocol + pynutil.insert('"')

        # ---------- 그래프 조합 ----------
        graph = pynini.Fst()  # empty

        # (1) 이메일
        email_guard = NEMO_SIGMA + AT + NEMO_SIGMA + DOT + NEMO_SIGMA
        graph |= pynini.compose(email_guard, username + domain_graph_with_class_tags)

        # (2) 단독 도메인 (프로토콜 없이)
        #   money 그래프 충돌 방지를 위해 '$' 제외, 이메일 구분자 '@' 제외
        dollar_accep = pynini.accep("$")
        excluded_symbols = DOT | dollar_accep | AT
        filtered_symbols = pynini.difference(accepted_symbols, excluded_symbols)
        accepted_characters = ASCII_ALNUM | filtered_symbols
        # label + (TLD)+ 또는 TLD 단독 (위 domain_core와 동일 정의 사용)
        graph_domain = (pynutil.insert('domain: "') + domain_core + pynutil.insert('"')).optimize()
        graph |= graph_domain

        # (3) URL (프로토콜 포함)
        graph |= protocol + pynutil.insert(" ") + domain_graph_with_class_tags

        # (4) 크레딧카드 cue + 숫자(4~16)
        if deterministic:
            cc_digits = pynini.closure(NEMO_DIGIT, 4, 16)
            cc_phrases = (
                pynutil.insert('protocol: "')
                + cc_cues
                + pynutil.insert('" domain: "')
                + delete_space
                + cc_digits
                + pynutil.insert('"')
            )
            graph |= cc_phrases

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
