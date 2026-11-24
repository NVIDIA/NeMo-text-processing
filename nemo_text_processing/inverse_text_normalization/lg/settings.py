# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright (c) 2026, Crane AI Labs. All rights reserved.
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

"""
Luganda (lg) on the shared Bantu kernel.

The word lists are in data/; this file holds the shape of the grammar: noun classes,
the word for ten, join words, places, counted scales and their switches. Every word list
and every decision here was reviewed by a native speaker.

What Luganda needs from the kernel:
- number words agree with the noun before them: abantu babiri, emiti ebiri, amagi abiri = 2
- the same word can mean two numbers: amagi abiri = 2 eggs, but after a currency
  phrase abiri is a shortened ten, ssiringi za uganda abiri = 20 UGX (data/numbers/roles.tsv)
- teens join with "na" (kkumi na bbiri = 12), larger parts with "mu"
  (amakumi abiri mu ttaano = 25); "mu" is also an ordinary word, "in"
- counted scales: mitwalo (ten-thousands), bukadde (millions), buwumbi (billions),
  each followed by its count: mitwalo ebiri = 20,000
- payment defaults: bare shillings (ssiringi, siringi) and bare scale amounts mean UGX
"""
import os

from nemo_text_processing.inverse_text_normalization.bantu.data_loader import load_profile
from nemo_text_processing.inverse_text_normalization.bantu.profile import (
    CardinalProfile,
    CountedScale,
    NounContext,
    Place,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Payment policy, not a fact about Luganda: bare scale amounts (mitwalo, bukadde, buwumbi)
# mean UGX wherever they appear. Augmented forms (emitwalo, obukadde) mean UGX only with
# a money signal. Decided with the native speaker.
IMPLIED_CURRENCY = "UGX"


def _big_scale(stem: str, power: int) -> CountedScale:
    """Millions (kadde) and billions (wumbi).

    kakadde kamu / bukadde + count is money; the count agrees with class 14 (as obutabo):
    obukadde abiri = 20 million (short tens), obukadde kikumi mu ataano = 150 million.
    Bare forms are always money. Augmented forms are money at the start of the input or
    after a money verb, and otherwise count the word before them; counting one needs kamu
    (ebitabo akakadde kamu), while akakadde alone is money only.
    """
    return CountedScale(
        singular="ka" + stem,
        singular_count="kamu",
        plural="bu" + stem,
        power=power,
        count_class="14",
        tens=(),
        remainder_class="10",
        implied_currency=IMPLIED_CURRENCY,
        counting_plural="obu" + stem,
        counting_ones=(f"aka{stem} kamu",),
        money_only_ones=("aka" + stem,),
        implied_anywhere=True,
        counting_money_at_start=True,
        counts_any_word=True,
        after_counted_nouns=True,
    )


def _plural(class_id: str, noun: str) -> NounContext:
    return NounContext(class_id, noun, (), ())


def _singular(class_id: str, noun: str) -> NounContext:
    return NounContext(class_id, noun, (), (), compounds=False)


SKELETON = CardinalProfile(
    language="lg",
    # One example noun per class; the number words for each class are in data/numbers/units.tsv.
    contexts=(
        _singular("1", "omuntu"),
        _plural("2", "abantu"),
        _singular("3", "omuti"),
        _plural("4", "emiti"),
        _singular("5", "eggi"),
        _plural("6", "amagi"),
        _singular("7", "ekitabo"),
        _plural("8", "ebitabo"),
        _singular("9", "ente"),
        _plural("10", "ente"),
        _singular("11", "olulimi"),
        _singular("12", "akatabo"),
        _plural("14", "obutabo"),
        _singular("15", "okutu"),
    ),
    ten="kkumi",
    teen_join=" na ",
    places=(Place(1, (), " mu "), Place(2, (), " mu "), Place(3, (), " mu "), Place(4, (), " mu ")),
    exact_boundaries=(),
    roles=(),
    max_value=999_999_999_999,
    counted_scales=(
        # Ten-thousands: mutwalo gumu / mitwalo + count (agrees with class 4, as emiti),
        # then an optional "mu" remainder. Tens inside the count are short: mitwalo
        # abiri = 200,000. Bare mitwalo is always money; any other word + emitwalo counts
        # that word (embuzi emitwalo ebiri = 20,000 goats), except after a money verb.
        CountedScale(
            singular="mutwalo",
            singular_count="gumu",
            plural="mitwalo",
            power=4,
            count_class="4",
            tens=(),
            remainder_class="10",
            contracted_teen_joins=("n'", "n"),  # na + ebiri -> n'ebiri or nebiri
            implied_currency=IMPLIED_CURRENCY,
            counting_plural="emitwalo",
            implied_anywhere=True,
            counts_any_word=True,
            after_counted_nouns=True,
        ),
        _big_scale("kadde", 6),  # million
        _big_scale("wumbi", 9),  # billion
    ),
)


def load() -> CardinalProfile:
    """Luganda's full profile: the skeleton above filled from data/."""
    return load_profile(DATA_DIR, SKELETON)
