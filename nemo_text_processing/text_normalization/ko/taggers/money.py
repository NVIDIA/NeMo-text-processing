# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.ko.graph_utils import NEMO_DIGIT, GraphFst, delete_space, insert_space
from nemo_text_processing.text_normalization.ko.utils import get_abs_path, load_labels


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money.
    Creates tokens like:
      money { integer_part: "삼백오십" currency_maj: "원" period: "년" }
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="classify", deterministic=deterministic)

        graph_cardinal = cardinal.graph
        SP = pynini.closure(delete_space)

        # --- 숫자 (정수/소수) ---
        # 정수부: "0" 또는 1-9 시작, 콤마 허용 (18,925,000 등)
        integer_part_fst = ((NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT | pynutil.delete(","))) | NEMO_DIGIT

        # integer_part (순수 수사)
        graph_integer_plain = (
            pynutil.insert('integer_part: "') + (integer_part_fst @ graph_cardinal) + pynutil.insert('" ')
        )

        # 소수부(두 자리) → minor_part (원화 테스트에선 주로 미사용이지만 형태 유지)
        decimal_part_fst = NEMO_DIGIT + NEMO_DIGIT
        graph_minor = pynutil.insert('minor_part: "') + (decimal_part_fst @ graph_cardinal) + pynutil.insert('" ')

        # 숫자 + (만|억|조) 접미 —— ★ 우선순위/괄호 버그 방지: 전체를 감싸 integer_part로 넣기
        scale_unit = pynini.union("만", "억", "조")
        value_with_scale = (integer_part_fst @ graph_cardinal) + scale_unit
        graph_integer_with_suffix = (
            pynutil.insert('integer_part: "') + value_with_scale + pynutil.insert('" ')
        ).optimize()

        # 정수(+선택 소수)
        number_component_plain = graph_integer_plain + pynini.closure(pynutil.delete(".") + graph_minor, 0, 1)
        number_component = (graph_integer_with_suffix | number_component_plain).optimize()

        # --- 통화 (선행/후행 모두) ---
        # currency_major.tsv 예:
        #   ₩   원
        #   KRW 원
        #   원  원
        maj_labels = load_labels(get_abs_path("data/money/currency_major.tsv"))

        # 선행 통화 (₩, KRW 등)
        currency_major_prepended = pynini.union(
            *[pynutil.delete(surface) + pynutil.insert(f'currency_maj: "{unit}" ') for surface, unit in maj_labels]
        ).optimize()

        # 후행 통화 (…원, …달러 등)
        currency_major_appended = pynini.union(
            *[pynutil.delete(unit) + pynutil.insert(f'currency_maj: "{unit}" ') for _, unit in maj_labels]
        ).optimize()

        # --- 기간(/월, /년, /주, /일, /시간) ---
        period_map = pynini.union(
            pynutil.delete("/월") + pynutil.insert('period: "월"'),
            pynutil.delete("/년") + pynutil.insert('period: "년"'),
            pynutil.delete("/주") + pynutil.insert('period: "주"'),
            pynutil.delete("/일") + pynutil.insert('period: "일"'),
            pynutil.delete("/시간") + pynutil.insert('period: "시간"'),
        )
        # 토큰 사이 출력 공백을 안정적으로 보장 (필드 내부에 이미 `" "`를 넣었기 때문에 여기선 앞에만 한 칸)
        period_opt = pynini.closure(pynutil.insert(" ") + period_map, 0, 1)

        # --- 결합 ---
        # 선행 통화: [통화] [숫자] [기간?]
        graph_prepend = (currency_major_prepended + SP + number_component + period_opt).optimize()

        # 후행 통화: [숫자] [통화] [기간?]
        graph_append = (number_component + currency_major_appended + period_opt).optimize()

        graph = (graph_prepend | graph_append).optimize()

        self.fst = self.add_tokens(graph).optimize()
