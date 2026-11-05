# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright (c) 2026, Crane Labs. All rights reserved.
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
A small made-up language for testing the shared Bantu kernel on its own.

Every word here is invented. The language is shaped like a Bantu language so that
each kernel feature is exercised (Luganda equivalents in brackets):
- noun classes with agreeing number words: bakopa bapi / zikopa zipi = 2 [abantu babiri / emiti ebiri]
- 6-9 and ten do not agree: sei, siti, oto, nona, deka [mukaaga ... kkumi]
- teens join with "ke", larger parts with "vo" [na, mu]
- "vo" is also an ordinary word, "in" [mu]
- a currency phrase that licenses a special reading: koin mapi = 20 XTS, but makopa mapi = 2 [abiri]
- other currency phrases, each agreeing with its own class: dolari zipi = 2 USD,
  yuro mafu = 5 EUR, yeni efu = 5 JPY
- a counted scale of ten-thousands: mina + count [mitwalo]; after a listed noun,
  emina counts that noun, also as an ordinary number place for 1-9:
  bakopa emina epi = 20,000, bakopa emina pideka = 200,000 [abantu emitwalo ebiri / abiri]
- implied currency: bare "mina" is always money; word + "emina" counts that word,
  except after a money verb [mitwalo / emitwalo]
- a held-back word that continues a number but is not handled: halfu [n'ekitundu]
- a count class with vowel-initial forms, so ke + epi contracts to k'epi [n'ebiri]
- a larger scale of millions: bumega + count, with augmented forms that count a noun,
  "one" forms, hundreds inside the count, and money at the start [obukadde / akakadde]
- a fixed phrase (tausi = 1,000), a special teen form (dekafu = 15) and a spelling
  variant (piideka = pideka) [akakadde, kkumi nataano, abiiri]

A failing test with this language points at the kernel, not at real language data.
XTS is the ISO 4217 code reserved for testing.

The word lists are in data/; this file holds the shape of the grammar: noun classes,
join words, places, counted scales and their switches. Hidden from the command line.
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

SKELETON = CardinalProfile(
    language="toy",
    contexts=(
        NounContext("1", "kopa", (), (), compounds=False),
        NounContext("2", "bakopa", (), ()),
        NounContext("4", "zikopa", (), ()),
        NounContext("6", "makopa", (), ()),
        NounContext("8", "ekopa", (), ()),
    ),
    ten="deka",
    teen_join=" ke ",
    places=(Place(1, (), " vo "), Place(2, (), " vo "), Place(3, (), " vo "), Place(4, (), " vo ")),
    exact_boundaries=(),
    roles=(),
    max_value=999_999_999,
    counted_scales=(
        # Ten-thousands [mitwalo]: bare mina is money anywhere; emina counts a word.
        CountedScale(
            singular="muna",
            singular_count="ema",
            plural="mina",
            power=4,
            count_class="8",
            tens=(),
            remainder_class="4",
            contracted_teen_joins=("k'", "k"),
            implied_currency="XTS",
            counting_plural="emina",
            implied_anywhere=True,
            counts_any_word=True,
            after_counted_nouns=True,
        ),
        # Millions [obukadde]: augmented forms are also money at the start of input,
        # count listed nouns with agreement, and need "one" to count (akamega kaima).
        CountedScale(
            singular="kamega",
            singular_count="kaima",
            plural="bumega",
            power=6,
            count_class="2",
            tens=(),
            remainder_class="4",
            implied_currency="XTS",
            counting_plural="obumega",
            counting_ones=("akamega kaima",),
            money_only_ones=("akamega",),
            implied_anywhere=True,
            counting_money_at_start=True,
            counts_any_word=True,
            after_counted_nouns=True,
        ),
    ),
)


def load() -> CardinalProfile:
    """The test language's full profile: the skeleton above filled from data/."""
    return load_profile(DATA_DIR, SKELETON)
