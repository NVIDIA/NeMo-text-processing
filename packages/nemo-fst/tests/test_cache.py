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

"""Cache invalidation.

A stale prepared artifact against a new grammar is the worst failure mode
available: plausible output, wrong grammar, no error.  The key is a hash of the
FAR's *contents* (plus the FAR key and an artifact-format version), so a
regenerated grammar cannot hit it -- and, deliberately, a mere `touch` can.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import nemo_fst
import pytest


def test_prepare_then_reuse(toy_far, tmp_path):
    first = nemo_fst.Tagger.from_far(toy_far, cache_dir=tmp_path)
    assert first.prepared is True
    artifact = Path(first.artifact_path)
    assert artifact.exists()
    assert artifact.with_suffix(artifact.suffix + ".relabel").exists()

    second = nemo_fst.Tagger.from_far(toy_far, cache_dir=tmp_path)
    assert second.prepared is False
    assert second.artifact_path == first.artifact_path
    assert second.relabel_pairs == first.relabel_pairs
    assert second.tag("alpha 12 beta") == first.tag("alpha 12 beta")


def test_touch_does_not_invalidate(toy_far, tmp_path):
    """The key is content, not mtime -- rebuilding an identical FAR is free."""
    grammar = tmp_path / "grammar.far"
    shutil.copy(toy_far, grammar)
    first = nemo_fst.Tagger.from_far(grammar, cache_dir=tmp_path)
    assert first.prepared is True

    os.utime(grammar, (time.time() + 60, time.time() + 60))
    second = nemo_fst.Tagger.from_far(grammar, cache_dir=tmp_path)
    assert second.prepared is False
    assert second.artifact_path == first.artifact_path


def test_regenerated_grammar_invalidates(toy_far, toy_other_far, tmp_path):
    """Same path, different contents: the artifact must be rebuilt, not reused."""
    grammar = tmp_path / "grammar.far"
    shutil.copy(toy_far, grammar)
    original = nemo_fst.Tagger.from_far(grammar, cache_dir=tmp_path)
    assert original.prepared is True

    # Stand-in for a regenerated grammar: a different FST under the same path.
    shutil.copy(toy_other_far, grammar)
    regenerated = nemo_fst.Tagger.from_far(grammar, key="verbalize",
                                           cache_dir=tmp_path)
    assert regenerated.prepared is True, "stale artifact was reused"
    assert regenerated.artifact_path != original.artifact_path
    assert regenerated.num_states != original.num_states

    # And the old artifact is still there and still valid for the old contents.
    shutil.copy(toy_far, grammar)
    restored = nemo_fst.Tagger.from_far(grammar, cache_dir=tmp_path)
    assert restored.prepared is False
    assert restored.artifact_path == original.artifact_path


def test_key_is_part_of_the_cache_key(toy_far, toy_other_far, tmp_path):
    a = nemo_fst.Tagger.from_far(toy_far, cache_dir=tmp_path)
    b = nemo_fst.Tagger.from_far(toy_other_far, key="verbalize",
                                 cache_dir=tmp_path)
    assert a.artifact_path != b.artifact_path


def test_truncated_artifact_is_rebuilt(toy_far, tmp_path):
    first = nemo_fst.Tagger.from_far(toy_far, cache_dir=tmp_path)
    artifact = Path(first.artifact_path)
    # A fraction of the actual size: a fixed offset is a no-op on a small
    # artifact, which makes the test silently prove nothing.
    data = artifact.read_bytes()
    artifact.write_bytes(data[: len(data) // 2])
    second = nemo_fst.Tagger.from_far(toy_far, cache_dir=tmp_path)
    assert second.prepared is True
    assert second.tag("alpha 12 beta") == first.tag("alpha 12 beta")


def test_missing_relabel_map_is_rebuilt(toy_far, tmp_path):
    """The map does not survive Write/Read inside the FST, so it lives beside it.

    Losing it is not a crash: LabelReachableData hands back an empty map, which
    is an identity relabelling, which composes to nothing.  So a cache entry
    without its map has to be treated as a miss.
    """
    first = nemo_fst.Tagger.from_far(toy_far, cache_dir=tmp_path)
    Path(first.artifact_path + ".relabel").unlink()
    second = nemo_fst.Tagger.from_far(toy_far, cache_dir=tmp_path)
    assert second.prepared is True
    assert second.relabel_pairs == first.relabel_pairs


def test_cache_dir_false_never_touches_disk(toy_far, tmp_path):
    tagger = nemo_fst.Tagger.from_far(toy_far, cache_dir=False)
    assert tagger.prepared is True
    assert tagger.artifact_path == ""
    assert tagger.tag("It costs $25.50.").startswith("tokens {")


def test_missing_far_and_missing_key(toy_far, tmp_path):
    with pytest.raises(FileNotFoundError):
        nemo_fst.Tagger.from_far(tmp_path / "nope.far", cache_dir=tmp_path)
    with pytest.raises(RuntimeError, match="not in FAR"):
        nemo_fst.Tagger.from_far(toy_far, key="no_such_key", cache_dir=tmp_path)
