# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.ko.graph_utils import (
    NEMO_SIGMA,
    GraphFst,
    insert_space,
    delete_space,
)

# ===== 공백/토큰 유틸 =====
SP = pynini.closure(delete_space)  # absorb 0+ spaces
NOT_QUOTE = pynini.difference(NEMO_SIGMA, pynini.accep('"'))
FIELD_VAL = pynini.closure(NOT_QUOTE, 1)

def del_key_val(key: str):
    """SP + key: "<VAL>" → <VAL>"""
    return (SP + pynutil.delete(f'{key}: "') + FIELD_VAL + pynutil.delete('"')).optimize()

def drop_key_val(key: str):
    """SP + key: "<ANY>" → (삭제)"""
    return (SP + pynutil.delete(f'{key}: "') + pynini.closure(NOT_QUOTE, 1) + pynutil.delete('"')).optimize()

def drop_key_exact(key: str, val: str):
    """SP + key: "val" → (삭제)"""
    return (SP + pynutil.delete(f'{key}: "{val}"')).optimize()


class MoneyFst(GraphFst):
    """
    Verbalize Korean money.

    Input tokens:
      tokens { money { integer_part: "..." currency_maj: "..." [minor_part: "..."] [period: "..."] } }

    Output examples:
      integer_part: "십" currency_maj: "원" period: "월"   -> "십원 매월"
      integer_part: "삼백오십" currency_maj: "원" period: "년" -> "삼백오십원 매년"
      integer_part: "삼십억" currency_maj: "원"           -> "삼십억원"
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="money", kind="verbalize", deterministic=deterministic)

        # --- 필드 파서 ---
        integer_part     = del_key_val("integer_part")
        minor_part_drop  = drop_key_val("minor_part")          # 원화 테스트에서는 소수부 무시(삭제)
        currency_val_any = del_key_val("currency_maj")         # ex) "원", "달러", "유로", "엔"
        won_key_drop     = drop_key_exact("currency_maj", "원")# "원" 키는 출력 없이 삭제

        # period: "월"|"년"|"주"|"일"|"시간" → " 매월" 등으로 매핑
        period_val = del_key_val("period")
        period_map = pynini.string_map([
            ("월", " 매월"),
            ("년", " 매년"),
            ("주", " 매주"),
            ("일", " 매일"),
            ("시간", " 매시간"),
        ])
        period_out_opt = pynini.closure(period_val @ period_map, 0, 1)

        # ===== KRW(원) 경로 =====
        # (A) [integer] [원] → "{integer}원"
        won_a = integer_part + SP + won_key_drop + pynutil.insert("원")
        # (B) [원] [integer] → "{integer}원"
        won_b = won_key_drop + SP + integer_part + pynutil.insert("원")
        won_core = (won_a | won_b)
        won_core = (won_core + pynini.closure(minor_part_drop, 0, 1)).optimize()

        # (C) [integer] [period] [원] → "{integer}원 매{period}"
        def drop_period_exact(val: str):
            return (SP + pynutil.delete(f'period: "{val}"')).optimize()

        won_between = integer_part + (
            drop_period_exact("월")   + SP + won_key_drop + pynutil.insert("원 매월")   |
            drop_period_exact("년")   + SP + won_key_drop + pynutil.insert("원 매년")   |
            drop_period_exact("주")   + SP + won_key_drop + pynutil.insert("원 매주")   |
            drop_period_exact("일")   + SP + won_key_drop + pynutil.insert("원 매일")   |
            drop_period_exact("시간") + SP + won_key_drop + pynutil.insert("원 매시간")
        )

        # ===== 기타 통화 =====
        # "{integer}{공백}{통화}" (한국어에선 원만 붙여 쓰고, 타 통화는 보통 띄움)
        other_core = (integer_part + insert_space + currency_val_any).optimize()
        other_core = (other_core + pynini.closure(minor_part_drop, 0, 1)).optimize()

        # ===== 결합 =====
        # KRW 경로 우선, 그 다음 "between" 경로, 그 다음 기타 통화
        graph_core = (
            pynutil.add_weight(won_core, 0.0) |
            pynutil.add_weight(won_between, 0.1) |
            pynutil.add_weight(other_core, 0.5)
        ).optimize()

        # 기본: 금액 + (뒤에 period가 따로 오면 " 매…" 붙이기)
        graph = (graph_core + period_out_opt).optimize()

        # tokens { money { ... } } 제거
        self.fst = self.delete_tokens(graph).optimize()
