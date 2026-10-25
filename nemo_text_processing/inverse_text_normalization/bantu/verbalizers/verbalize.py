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

from nemo_text_processing.inverse_text_normalization.bantu.profile import CardinalProfile
from nemo_text_processing.inverse_text_normalization.bantu.verbalizers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.bantu.verbalizers.money import MoneyFst
from nemo_text_processing.text_normalization.en.graph_utils import GraphFst


class VerbalizeFst(GraphFst):
    """
    Composes the verbalizer grammars. Values arrive fully resolved from the
    classifier, so verbalizing is formatting only.

    Args:
        profile: the language profile (sets the largest value)
        thousands_separator: written separator between digit groups
    """

    def __init__(self, profile: CardinalProfile, thousands_separator: str = ","):
        super().__init__(name="verbalize", kind="verbalize")
        cardinal = CardinalFst(profile.max_value, thousands_separator).fst
        money = MoneyFst(profile.max_value, thousands_separator).fst
        self.fst = (cardinal | money).optimize()
