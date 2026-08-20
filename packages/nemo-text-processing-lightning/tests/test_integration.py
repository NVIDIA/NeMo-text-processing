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

"""Drive `nemo_text_processing.Normalizer` through this package's tagger.

Nothing in `nemo_text_processing` calls nemo-text-processing-lightning yet. This substitutes the
tagging step at runtime, so the whole pipeline -- token parser, permuter,
verbalizer, post-processor -- runs on nemo-text-processing-lightning's output and the end of it can be
compared against the stock pipeline.

Two methods are replaced together because `normalize()` calls `find_tags` to get
a lattice and then `Normalizer.select_tag` to take its one-best, whereas
nemo-text-processing-lightning does both in one call. Passing the tagged string through `find_tags`
and making `select_tag` the identity threads it through untouched.
"""

from __future__ import annotations

import pytest

from test_differential import squash, weight_of


def use_lightning(monkeypatch, tagger):
    """Substitute the tagging step on the Normalizer class.

    Two methods together: `normalize()` calls `find_tags` for a lattice and then
    `Normalizer.select_tag` -- by class, not through self -- to take its
    one-best, whereas nemo-text-processing-lightning does both in one call. Returning the tagged
    string from `find_tags` and making `select_tag` the identity threads it
    through untouched.
    """
    from nemo_text_processing.text_normalization.normalize import Normalizer

    monkeypatch.setattr(Normalizer, "find_tags", lambda self, text: tagger.tag(text))
    monkeypatch.setattr(Normalizer, "select_tag", staticmethod(lambda tagged: tagged))


@pytest.fixture
def patched_normalizer(normalizer, tagger, monkeypatch):
    use_lightning(monkeypatch, tagger)
    return normalizer


def test_normalize_end_to_end_matches_stock_pipeline(normalizer, tagger, inputs, monkeypatch):
    """Normalized output must match, except where the tagger had a genuine tie.

    The stock outputs are collected *before* the substitution, or both sides
    would be nemo-text-processing-lightning and the comparison would be vacuous.
    """
    import pynini

    stock = {text: normalizer.normalize(text) for text in inputs}

    use_lightning(monkeypatch, tagger)
    probe = "It costs $25.50."
    assert normalizer.find_tags(probe) == tagger.tag(probe), "substitution is not in the path"

    ours = {text: normalizer.normalize(text) for text in inputs}

    differing = [t for t in inputs if stock[t] != ours[t]]
    untied = []
    for text in differing:
        lattice = pynini.escape(text) @ normalizer.tagger.fst
        base_tag = pynini.shortestpath(lattice, nshortest=1, unique=True).string()
        w_base, w_ours = weight_of(lattice, base_tag), weight_of(lattice, tagger.tag(text))
        if w_ours is None or w_base is None or w_ours != w_base:
            untied.append((text, stock[text], ours[text], w_base, w_ours))

    n = len(inputs)
    print(
        f"\n{n - len(differing)}/{n} normalized outputs identical; "
        f"{len(differing) - len(untied)} differ from a weight-tied tagging; {len(untied)} unexplained"
    )
    for text in differing[:5]:
        print(
            f"  tie  {text!r}\n    stock   : {stock[text][:70]!r}\n    nemo-text-processing-lightning: {ours[text][:70]!r}"
        )
    for text, a, b, wa, wb in untied[:5]:
        print(f"  UNEXPLAINED {text!r}: stock {wa} -> {a[:60]!r}, nemo-text-processing-lightning {wb} -> {b[:60]!r}")
    assert not untied, f"{len(untied)} outputs differ without a tie to explain them"


def test_patched_pipeline_still_produces_wellformed_output(patched_normalizer):
    """A sanity check that the substitution really is in the path."""
    got = patched_normalizer.normalize("It costs $25.50 on 3/4/2023.")
    assert "twenty five dollars fifty cents" in got, got
    assert "march fourth" in got, got


def test_tagged_string_feeds_the_token_parser(tagger, normalizer):
    """nemo-text-processing-lightning's tagged string must satisfy the parser the pipeline hands it to."""
    from nemo_text_processing.text_normalization.token_parser import TokenParser

    parser = TokenParser()
    for text in ("It costs $25.50.", "Call 555-0105 at 5:30 p.m.", "Plain words only."):
        tagged = tagger.tag(text)
        parser(tagged)
        tokens = parser.parse()
        assert tokens, (text, tagged)
        assert squash(tagged).startswith("tokens {"), tagged
