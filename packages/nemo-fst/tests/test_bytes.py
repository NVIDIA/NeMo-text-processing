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

"""Byte coverage.

The `olabel_lookahead` conversion renumbers a 244-value alphabet, and a byte
relabelled wrongly does not raise -- it silently fails to match, and the tagger
quietly returns something shorter.  Tab, newline, carriage return and the
multi-byte UTF-8 lead bytes are the interesting ones: they are exactly the
labels a pure-Python reconstruction of the map could not recover (it got 240 of
244, and two of the stragglers share an image so set reasoning cannot close the
gap).  Getting them from the C++ side is a large part of why this package
exists, so the tests are behavioural, not just structural.
"""

from __future__ import annotations

import pytest

MULTIBYTE = [
    "Résumé costs €25.",            # 2-byte lead bytes (C3, E2 sequences)
    "Der Preis beträgt 30 €.",
    "Naïve café — 5 items.",        # em dash, 3-byte
    "Emoji \U0001F600 and 7 things.",  # 4-byte lead byte
    "日本語 12 items.",
]

CONTROL = [
    "Line one 5 things.\nLine two 7 things.",
    "Tabbed\t12 items\tand $3.",
    "CR here\r and 9 things.",
    "Mixed\r\n\t 42 items.",
]


def pynini_tag(normalizer, text: str) -> str:
    import pynini

    lattice = pynini.escape(text) @ normalizer.tagger.fst
    return pynini.shortestpath(lattice, nshortest=1, unique=True).string()


def squash(text: str) -> str:
    return " ".join(text.split())


@pytest.mark.parametrize("text", MULTIBYTE + CONTROL)
def test_tagging_matches_pynini(text, tagger, normalizer):
    """A byte relabelled wrongly does not raise -- it silently fails to match."""
    assert squash(tagger.tag(text)) == squash(pynini_tag(normalizer, text))


@pytest.mark.parametrize("text", MULTIBYTE + CONTROL)
def test_tagger_consumes_the_whole_input(text, tagger):
    """A mis-relabelled byte shows up as an empty tag, not an exception."""
    tagged = tagger.tag(text)
    assert tagged.startswith("tokens {"), (text, tagged)
    # Every non-whitespace run of the input should be represented; the cheapest
    # proxy is that the tagged string is not truncated relative to the input.
    assert len(tagged) >= len(text.strip()), (text, tagged)


def test_every_byte_of_the_alphabet_round_trips(tagger, normalizer):
    """Sweep the printable/control range one byte at a time.

    The relabelling map covers a 244-value alphabet, and a byte mapped wrongly
    produces no error -- just a tag that quietly does not match.
    """
    disagree = []
    for value in list(range(1, 128)) + [0xC3, 0xE2]:
        try:
            char = bytes([value]).decode("utf-8")
        except UnicodeDecodeError:
            continue
        text = f"a{char}b 12 items."
        if squash(tagger.tag(text)) != squash(pynini_tag(normalizer, text)):
            disagree.append((value, text))
    assert not disagree, disagree[:10]


def test_control_bytes_are_actually_relabelled(tagger):
    """Structural check: the three the prototype could not recover.

    If these ever come back as identity the behavioural tests above are the only
    thing standing between a silent regression and production.
    """
    pairs = tagger.relabel_pairs
    for byte in (0x09, 0x0A, 0x0D):
        assert byte in pairs, f"byte {byte:#04x} missing from the relabel map"
        assert pairs[byte] != byte, f"byte {byte:#04x} was not renumbered"
    assert len(pairs) >= 244, len(pairs)
