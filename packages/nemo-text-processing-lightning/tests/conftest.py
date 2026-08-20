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

"""Fixtures for the nemo-text-processing-lightning tests.

Everything here needs a compiled English tagger FAR. Building one from source
takes ~20 s, so point `--tn_cache_dir` at an existing cache to reuse it, the
same way the main test suite does:

    pytest --tn_cache_dir=/path/to/grammars
"""

from __future__ import annotations

import glob
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
CORPUS = REPO / "tests" / "nemo_text_processing" / "en" / "data_text_normalization"


def pytest_addoption(parser):
    parser.addoption(
        "--tn_cache_dir",
        action="store",
        default=None,
        help="path to a directory with cached .far grammar files",
    )


@pytest.fixture(scope="session")
def cache_dir(request, tmp_path_factory) -> Path:
    """Where compiled grammars live. A shared cache if one was given."""
    given = request.config.getoption("--tn_cache_dir")
    return Path(given) if given else tmp_path_factory.mktemp("grammars")


@pytest.fixture(scope="session")
def artifact_dir(tmp_path_factory) -> Path:
    """Where nemo-text-processing-lightning writes its prepared lookahead artifacts."""
    return tmp_path_factory.mktemp("nemo_text_processing_lightning_artifacts")


TAGGER_FAR = "en_tn_True_deterministic_cased__tokenize.far"
TOY_FAR = Path(__file__).parent / "data" / "toy_tagger.far"
TOY_OTHER_FAR = Path(__file__).parent / "data" / "toy_other.far"


@pytest.fixture(scope="session")
def normalizer(cache_dir):
    """NeMo's own normalizer, the oracle the differential tests compare against."""
    pytest.importorskip("pynini")
    from nemo_text_processing.text_normalization.normalize import Normalizer

    return Normalizer(input_case="cased", lang="en", cache_dir=str(cache_dir))


@pytest.fixture(scope="session")
def far_path(request, cache_dir) -> Path:
    """A compiled tagger FAR.

    A FAR is a data file, so an existing one is usable with no pynini in the
    environment -- which is the case in every wheel-test container, since pynini
    publishes manylinux x86_64 wheels only. pynini is pulled in only when one
    has to be compiled.
    """
    far = cache_dir / TAGGER_FAR
    if far.exists():
        return far
    request.getfixturevalue("normalizer")  # compiles it, and needs pynini
    if not far.exists():
        pytest.skip(f"no tagger FAR at {far}")
    return far


@pytest.fixture(scope="session")
def tagger(far_path, artifact_dir):
    import nemo_text_processing_lightning

    return nemo_text_processing_lightning.Tagger.from_far(far_path, cache_dir=artifact_dir)


@pytest.fixture(scope="session")
def inputs() -> list:
    """Left-hand sides of the English text-normalization test cases."""
    texts = []
    for path in sorted(glob.glob(str(CORPUS / "test_cases_*.txt"))):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "~" in line:
                texts.append(line.split("~", 1)[0])
    if not texts:
        pytest.skip(f"no test-case files under {CORPUS}")
    return texts


@pytest.fixture(scope="session")
def toy_far() -> Path:
    """A tiny checked-in grammar: digits to cardinal, letters to name.

    Anything structural -- caching, determinism, error handling, thread safety --
    needs an FST but not a particular one. Using this rather than the English
    tagger means those tests run with no pynini and no compiled grammars, which
    is the situation in every wheel-test container.
    """
    if not TOY_FAR.exists():
        pytest.skip(f"toy grammar missing at {TOY_FAR}")
    return TOY_FAR


@pytest.fixture(scope="session")
def toy_tagger(toy_far, artifact_dir):
    import nemo_text_processing_lightning

    return nemo_text_processing_lightning.Tagger.from_far(toy_far, cache_dir=artifact_dir / "toy")


@pytest.fixture(scope="session")
def toy_other_far() -> Path:
    """A second toy grammar, under a different key, for cache-keying tests."""
    if not TOY_OTHER_FAR.exists():
        pytest.skip(f"second toy grammar missing at {TOY_OTHER_FAR}")
    return TOY_OTHER_FAR
