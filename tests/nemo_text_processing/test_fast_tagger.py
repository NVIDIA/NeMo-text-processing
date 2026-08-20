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

"""The optional nemo-fst tagging path, and its fallback.

These run whether or not nemo-fst is installed. It is used automatically when
available; the point of the tests is that being unable to use it degrades to
pynini rather than failing, and that the choice is overridable either way.
"""

import logging

import pytest

from nemo_text_processing.text_normalization.normalize import Normalizer

from .utils import CACHE_DIR


@pytest.fixture(scope="module")
def normalizer():
    return Normalizer(input_case="cased", lang="en", cache_dir=CACHE_DIR)


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_used_automatically_when_available(normalizer):
    """No flag needed: if the package is importable and there is a FAR, use it."""
    try:
        import nemo_fst
    except ImportError:
        assert normalizer._fst_tagger is None
        return
    assert (normalizer._fst_tagger is not None) == nemo_fst.has_lookahead()


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_can_be_turned_off():
    """`fast_tagger=False` forces the pynini path even when the package is there."""
    norm = Normalizer(input_case="cased", lang="en", cache_dir=CACHE_DIR, fast_tagger=False)
    assert norm._fst_tagger is None
    assert norm.normalize("It costs $25.50.") == "It costs twenty five dollars fifty cents."


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_missing_package_degrades_with_a_warning(monkeypatch, caplog):
    """Asking for the fast path without the package warns and keeps working."""
    import builtins

    real_import = builtins.__import__

    def no_nemo_fst(name, *args, **kwargs):
        if name == "nemo_fst":
            raise ImportError("simulated: nemo-fst not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_nemo_fst)
    with caplog.at_level(logging.WARNING):
        # fast_tagger=True, so being unable to use it is worth a warning; left
        # to itself the same situation is only a debug line.
        norm = Normalizer(input_case="cased", lang="en", cache_dir=CACHE_DIR, fast_tagger=True)

    assert norm._fst_tagger is None
    assert any("nemo-fst is not installed" in r.message for r in caplog.records), caplog.text
    assert norm.normalize("It costs $25.50.") == "It costs twenty five dollars fifty cents."


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_opting_out_of_the_cache_degrades_with_a_warning(caplog):
    """cache_dir="None" means recompile every time, so there is no FAR to load."""
    with caplog.at_level(logging.WARNING):
        norm = Normalizer(input_case="cased", lang="en", cache_dir="None", fast_tagger=True)
    assert norm._fst_tagger is None
    assert any("grammar cache" in r.message for r in caplog.records), caplog.text


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_cache_dir_defaults_to_a_writable_location(tmp_path, monkeypatch):
    """Unset means the default cache, not "no cache"."""
    from nemo_text_processing.text_normalization.normalize import default_cache_dir

    monkeypatch.setenv("NEMO_TEXT_PROCESSING_CACHE_DIR", str(tmp_path / "grammars"))
    assert default_cache_dir() == str(tmp_path / "grammars")
    norm = Normalizer(input_case="cased", lang="en")
    assert norm.cache_dir == str(tmp_path / "grammars")
    assert (tmp_path / "grammars").is_dir()


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_unwritable_default_cache_is_not_fatal(monkeypatch, caplog):
    """A read-only home is a warning and slower grammars, not a failure."""
    monkeypatch.setenv("NEMO_TEXT_PROCESSING_CACHE_DIR", "/proc/nonexistent/cache")
    with caplog.at_level(logging.WARNING):
        norm = Normalizer(input_case="cased", lang="en", cache_dir=None)
    assert norm.cache_dir is None
    assert any("default grammar cache" in r.message for r in caplog.records), caplog.text
    assert norm.normalize("It costs $25.50.") == "It costs twenty five dollars fifty cents."


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_fast_path_agrees_with_pynini_where_the_grammar_is_unambiguous(normalizer):
    """When enabled, output matches on inputs with a single cheapest parse.

    Inputs whose parse is genuinely tied are excluded: both implementations
    return a lowest-cost reading, but not necessarily the same one. The
    nemo-fst package's own differential test is what pins the stronger property
    -- never a costlier path than pynini's.
    """
    pytest.importorskip("nemo_fst")
    fast = Normalizer(input_case="cased", lang="en", cache_dir=CACHE_DIR, fast_tagger=True)
    slow = Normalizer(input_case="cased", lang="en", cache_dir=CACHE_DIR, fast_tagger=False)
    if fast._fst_tagger is None:
        pytest.skip("nemo-fst present but unusable in this environment")

    for text in (
        "It costs $25.50 on 3/4/2023.",
        "Call 555-0105 before 5:30 p.m.",
        "The quick brown fox jumps over 3 lazy dogs.",
        "Résumé costs €25.",
        "Meeting at 10:30 a.m. on January 5th, 2021.",
    ):
        assert fast.normalize(text) == slow.normalize(text), text
