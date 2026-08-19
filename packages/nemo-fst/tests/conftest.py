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

"""Fixtures for the nemo-fst tests.

Everything here needs a compiled English tagger FAR. Building one from source
takes ~20 s, so point `--tn_cache_dir` at an existing cache to reuse it, the
same way the main test suite does:

    pytest --tn_cache_dir=/path/to/grammars
"""

from __future__ import annotations

from pathlib import Path

import pytest


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
    """Where nemo-fst writes its prepared lookahead artifacts."""
    return tmp_path_factory.mktemp("nemo_fst_artifacts")


@pytest.fixture(scope="session")
def normalizer(cache_dir):
    """NeMo's own normalizer, which compiles the FAR if it is not cached yet."""
    pytest.importorskip("pynini")
    from nemo_text_processing.text_normalization.normalize import Normalizer

    return Normalizer(input_case="cased", lang="en", cache_dir=str(cache_dir))


@pytest.fixture(scope="session")
def far_path(normalizer, cache_dir) -> Path:
    far = cache_dir / "en_tn_True_deterministic_cased__tokenize.far"
    if not far.exists():
        pytest.skip(f"no tagger FAR at {far}")
    return far


@pytest.fixture(scope="session")
def tagger(far_path, artifact_dir):
    import nemo_fst

    return nemo_fst.Tagger.from_far(far_path, cache_dir=artifact_dir)
