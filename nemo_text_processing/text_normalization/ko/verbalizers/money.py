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

from nemo_text_processing.text_normalization.ko.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space, insert_space

# ===== whitespace & token helpers =====
sp = pynini.closure(delete_space)  # absorb 0+ spaces
FIELD_VAL = pynini.closure(NEMO_NOT_QUOTE, 1)


def del_key_val(key: str):
    """
    Delete the token field prefix and quotes, keep only the value.

    Input format:  [sp] key: "<VAL>"
    Output:        <VAL>

    Example:
      input  'integer_part: "삼백오십"'
      output '삼백오십'
    """
    return (sp + pynutil.delete(f'{key}: "') + FIELD_VAL + pynutil.delete('"')).optimize()


def drop_key_val(key: str):
    """
    Delete the entire key-value pair (key and its quoted value).

    Input format:  [sp] key: "<ANY>"
    Output:        (nothing)

    Example:
      input  'minor_part: "십"'
      output ''
    """
    return (sp + pynutil.delete(f'{key}: "') + FIELD_VAL + pynutil.delete('"')).optimize()
  

def drop_key_exact(key: str, val: str):
    """
    Delete the exact key-value pair if it matches the given value.

    Input format:  [sp] key: "val"
    Output:        (nothing)

    Example:
      input  'currency_maj: "원"'
      output ''
    """
    return (sp + pynutil.delete(f'{key}: "{val}"')).optimize()


class MoneyFst(GraphFst):
    """
    Verbalize Korean money.

    Input tokens:
      tokens { money { integer_part: "..." currency_maj: "..." [minor_part: "..."] } }

    Period (e.g., /월, /년, …) is intentionally NOT handled here.
    Output examples:
      integer_part: "십" currency_maj: "원"          -> "십원"
      integer_part: "삼십억" currency_maj: "원"     -> "삼십억원"
      integer_part: "이백" currency_maj: "달러"     -> "이백 달러"
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="money", kind="verbalize", deterministic=deterministic)

        # --- fields ---
        integer_part = del_key_val("integer_part")
        minor_part_drop = drop_key_val("minor_part")  # ignore minor for KRW
        currency_val_any = del_key_val("currency_maj")  # ex) "원", "달러", "유로"
        won_key_drop = drop_key_exact("currency_maj", "원")  # don't print the key for KRW

        # ===== KRW (원) =====
        # (A) [integer] [원] -> "{integer}원"
        won_a = integer_part + sp + won_key_drop + pynutil.insert("원")
        # (B) [원] [integer] -> "{integer}원"
        won_b = won_key_drop + sp + integer_part + pynutil.insert("원")
        won_core = won_a | won_b
        won_core = (won_core + pynini.closure(minor_part_drop, 0, 1)).optimize()

        # ===== Other currencies =====
        # "{integer} {currency}" (KRW sticks; others are spaced)
        other_core = (integer_part + insert_space + currency_val_any).optimize()
        other_core = (other_core + pynini.closure(minor_part_drop, 0, 1)).optimize()

        # ===== combine (no period) =====
        graph_core = (pynutil.add_weight(won_core, 0.0) | pynutil.add_weight(other_core, 0.5)).optimize()

        # no trailing period mapping
        graph = graph_core

        # strip tokens wrapper
        self.fst = self.delete_tokens(graph).optimize()
