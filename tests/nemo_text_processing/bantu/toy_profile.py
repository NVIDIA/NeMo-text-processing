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

"""
A small made-up language for testing the shared Bantu kernel on its own.

Every word here is invented. The language is shaped like a Bantu language so that
each kernel feature is exercised (Luganda equivalents in brackets):
- noun classes with agreeing number words: bakopa bapi / zikopa zipi = 2 [abantu babiri / emiti ebiri]
- 6-9 and ten do not agree: sei, siti, oto, nona, deka [mukaaga ... kkumi]
- teens join with "ke", larger parts with "vo" [na, mu]
- "vo" is also an ordinary word, "in" [mu]
- a currency phrase that licenses a special reading: koin mapi = 20 XTS, but makopa mapi = 2 [abiri]
- a counted scale of ten-thousands: mina + count [mitwalo]
- implied currency: bare "mina" is always money; word + "emina" counts that word,
  except after a money verb [mitwalo / emitwalo]
- a held-back word that continues a number but is not handled: halfu [n'ekitundu]
- a count class with vowel-initial forms, so ke + epi contracts to k'epi [n'ebiri]
- a larger scale of millions: bumega + count, with augmented forms that count a noun,
  "one" forms, hundreds inside the count, and money at the start [obukadde / akakadde]
- a fixed phrase (tausi = 1,000), a special teen form (dekafu = 15) and a spelling
  variant (piideka = pideka) [akakadde, kkumi nataano, abiiri]

A failing test with this profile points at the kernel, not at real language data.
XTS is the ISO 4217 code reserved for testing.
"""
from nemo_text_processing.inverse_text_normalization.bantu.profile import (
    CardinalProfile,
    CountedScale,
    CurrencyContext,
    MoneyCues,
    NounContext,
    Place,
    with_spelling_variants,
)

INVARIANT = (("sei", "6"), ("siti", "7"), ("oto", "8"), ("nona", "9"))
STEMS = (("pi", "2"), ("ko", "3"), ("de", "4"), ("fu", "5"))
TENS = (
    ("pideka", "2"),
    ("kodeka", "3"),
    ("dedeka", "4"),
    ("fudeka", "5"),
    ("seideka", "6"),
    ("sitideka", "7"),
    ("otodeka", "8"),
    ("nonadeka", "9"),
)


def plural(class_id, noun, prefix, final_one=None, teen_variants=()):
    units = tuple((prefix + stem, value) for stem, value in STEMS) + INVARIANT
    final = units + (((final_one, "1"),) if final_one else ())
    return NounContext(class_id, noun, units + (("deka", "10"),), final, teen_variants=teen_variants)


def multiples(scale):
    words = (("pi", "2"), ("ko", "3"), ("de", "4"), ("fu", "5")) + INVARIANT
    return ((scale, "1"),) + tuple((f"{scale} {word}", value) for word, value in words)


TOY_PROFILE = with_spelling_variants(
    CardinalProfile(
        language="toy",
        contexts=(
            NounContext("1", "kopa", (("ima", "1"),), (), compounds=False),
            plural("2", "bakopa", "ba", final_one="baima"),
            plural("4", "zikopa", "zi", final_one="zima"),
            plural("6", "makopa", "ma", teen_variants=(("dekafu", "15"),)),
            plural("8", "ekopa", "e", final_one="ema"),
        ),
        ten="deka",
        teen_join=" ke ",
        places=(
            Place(1, TENS, " vo "),
            Place(2, multiples("hanto"), " vo "),
            Place(3, multiples("milo"), " vo "),
        ),
        exact_boundaries=(("tausi", "1000"),),
        roles=(("short-tens", (("mapi", "20"),)),),
        max_value=999_999_999,
        currencies=(CurrencyContext("koin", "XTS", agreement_class="4", roles=("short-tens",)),),
        counted_scales=(
            # Ten-thousands [mitwalo]: bare mina is money anywhere; emina counts a word.
            CountedScale(
                singular="muna",
                singular_count="ema",
                plural="mina",
                power=4,
                count_class="8",
                tens=(("deka", "1"),) + TENS,
                remainder_class="4",
                contracted_teen_joins=("k'", "k"),
                implied_currency="XTS",
                counting_plural="emina",
                implied_anywhere=True,
                counts_any_word=True,
            ),
            # Millions [obukadde]: augmented forms are also money at the start of input,
            # count listed nouns with agreement, and need "one" to count (akamega kaima).
            CountedScale(
                singular="kamega",
                singular_count="kaima",
                plural="bumega",
                power=6,
                count_class="2",
                tens=(("deka", "1"),) + TENS,
                remainder_class="4",
                count_ones=(("kaima", "1"),),
                count_hundreds=multiples("hanto"),
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
        money_cues=MoneyCues(stems=("lipo",), words=("dalo",), not_cues=("olipon",)),
        held_back_words=("halfu",),
    ),
    {"pideka": ("piideka",)},
)
