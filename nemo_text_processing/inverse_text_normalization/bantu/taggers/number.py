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

"""Place-value number FSTs built from a profile, shared by the cardinal, money and scale taggers.

Values are composed digit by digit inside the FST; there is no runtime arithmetic.
"""
import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.bantu.profile import CardinalProfile, Forms, NounContext

DIGIT = pynini.union(*"0123456789").optimize()
NONZERO = pynini.union(*"123456789").optimize()


def mapping(forms: Forms) -> pynini.Fst:
    return pynini.string_map(forms).optimize() if forms else pynini.Fst()


def integer_domain(max_value: int) -> pynini.Fst:
    """Canonical positive decimal strings <= bound, without integer enumeration."""
    if isinstance(max_value, bool) or not isinstance(max_value, int) or max_value < 1:
        raise ValueError("max_value must be a positive integer")
    bound = str(max_value)
    branches = [pynini.accep(bound)]
    if len(bound) > 1:
        branches.append(NONZERO + pynini.closure(DIGIT, 0, len(bound) - 2))
    for index, limit in enumerate(bound):
        minimum = 1 if index == 0 else 0
        if int(limit) > minimum:
            lower = pynini.union(*(str(n) for n in range(minimum, int(limit))))
            branches.append(
                pynini.accep(bound[:index])
                + lower
                + pynini.closure(DIGIT, len(bound) - index - 1, len(bound) - index - 1)
            )
    return pynini.union(*branches).optimize()


def build_cardinal(profile: CardinalProfile, context: NounContext) -> pynini.Fst:
    """Compose fixed-width places, supply skipped zeros, then canonicalize once."""
    atomic = mapping(context.atomic_units)
    if not context.compounds:
        return (atomic @ integer_domain(profile.max_value)).optimize()
    units = mapping(context.final_units)
    lower = units
    expected_power = 1
    for place in profile.places:
        if place.power != expected_power:
            raise ValueError("places must be consecutive ascending decimal powers")
        if any(value not in "123456789" or len(value) != 1 for _, value in place.forms):
            raise ValueError("place coefficients must be single nonzero digits")
        # A present high place either consumes a smaller remainder or emits zeros.
        high = mapping(place.forms) + (
            pynutil.delete(place.remainder_join) + lower | pynutil.insert("0" * place.power)
        )
        if place.power == 1:
            high |= pynini.cross(profile.ten, "10")
            high |= pynini.cross(profile.ten, "1") + pynutil.delete(profile.teen_join) + units
            high |= mapping(context.teen_variants)
        lower = (pynutil.insert("0") + lower | high).optimize()
        expected_power += 1
    canonical = pynutil.delete(pynini.closure("0")) + NONZERO + pynini.closure(DIGIT)
    compounds = lower @ canonical
    # A final-one form need not be licensed as a standalone plural count.
    compounds @= NONZERO + pynini.closure(DIGIT, 1)
    result = atomic | compounds | mapping(profile.exact_boundaries)
    return (result @ integer_domain(profile.max_value)).optimize()


def build_role(profile: CardinalProfile, role: str) -> pynini.Fst:
    """Explicit role probe; not a standalone sentence or currency grammar."""
    for name, forms in profile.roles:
        if name == role:
            return mapping(forms)
    raise KeyError(role)
