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

"""What can be checked with no pynini in the environment.

This is what verifies a built wheel. pynini publishes manylinux x86_64 wheels
only, so the container that tests an aarch64, macOS or Windows wheel cannot
install it, and every assertion that needs an oracle is unavailable there. These
need nothing but the extension and, for most of them, a compiled grammar.

They are properties rather than comparisons: that the build carries a working
lookahead OpenFst, exports nothing it should not, produces well-formed and
stable output, releases the GIL, and fails cleanly on bad input.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time

import pytest

import nemo_fst


def test_lookahead_is_actually_compiled_in():
    """The one assertion a wheel must not ship without.

    An OpenFst built without --enable-lookahead-fsts produces a package that
    imports, has an API, and is worth nothing.
    """
    assert nemo_fst.has_lookahead()


def test_openfst_version_is_reported():
    assert nemo_fst.__openfst_version__.startswith("1.8.")


def test_exports_only_the_module_init_symbol():
    """Two OpenFst copies in one process are safe only if ours is invisible."""
    if not sys.platform.startswith("linux"):
        pytest.skip("nm -D is Linux-specific")
    so = nemo_fst._nemo_fst.__file__
    out = subprocess.run(["nm", "-D", "--defined-only", so], capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("nm unavailable")
    exported = [line.split()[-1] for line in out.stdout.splitlines() if line.strip()]
    assert exported == ["PyInit__nemo_fst"], exported


def test_no_external_openfst_dependency():
    """A relocatable wheel carries its OpenFst; it does not look for one."""
    if not sys.platform.startswith("linux"):
        pytest.skip("readelf is Linux-specific")
    so = nemo_fst._nemo_fst.__file__
    out = subprocess.run(["readelf", "-d", so], capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("readelf unavailable")
    assert "libfst" not in out.stdout, out.stdout


def test_tagged_output_is_wellformed(tagger):
    """Structure, not content: no oracle needed to know this much."""
    tagged = tagger.tag("It costs $25.50 on 3/4/2023.")
    assert tagged.startswith("tokens {"), tagged
    assert tagged.count("{") == tagged.count("}"), tagged
    assert "money" in tagged, tagged
    assert "date" in tagged, tagged


def test_tagging_is_deterministic(tagger):
    """The same input must give the same answer every time.

    Worth asserting rather than assuming: the lookahead FST is shared across
    calls and carries precomputed reachability, so a bug that mutated it would
    show up here and nowhere else.
    """
    text = "Call 555-0105 before 5:30 p.m. about the $5,204.50 invoice."
    first = tagger.tag(text)
    assert all(tagger.tag(text) == first for _ in range(20))


def test_empty_and_whitespace_input(tagger):
    for text in ("", " ", "\n", "\t "):
        tagger.tag(text)  # must not raise


def test_multibyte_input_round_trips(tagger):
    """A byte relabelled wrongly does not raise, it silently fails to match."""
    for text in ("Résumé costs €25.", "日本語 12 items.", "Emoji 😀 and 7 things."):
        tagged = tagger.tag(text)
        assert tagged.startswith("tokens {"), (text, tagged)


def test_missing_far_raises_cleanly(tmp_path):
    with pytest.raises(Exception):  # noqa: B017 -- any clean failure, not a crash
        nemo_fst.Tagger.from_far(tmp_path / "does-not-exist.far", cache_dir=tmp_path)


def test_missing_key_raises_cleanly(far_path, tmp_path):
    with pytest.raises(RuntimeError, match="not in FAR"):
        nemo_fst.Tagger.from_far(far_path, key="no-such-key", cache_dir=tmp_path)


def test_tag_releases_the_gil(tagger):
    """A sibling thread must keep running while a composition does.

    Measured as a ratio against an idle baseline rather than a wall-clock
    threshold, so it means the same thing on a slow machine.
    """
    text = "On January 5th, 2021 revenue was $1,234.56, up 12.5% from Q3. " * 4

    def ticker(stop, out):
        n = 0
        while not stop.is_set():
            n += 1
        out.append(n)

    def measure(load):
        stop, out = threading.Event(), []
        thread = threading.Thread(target=ticker, args=(stop, out))
        thread.start()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 0.5:
            tagger.tag(text) if load else time.sleep(0.005)
        stop.set()
        thread.join()
        return out[0]

    idle, busy = measure(False), measure(True)
    assert busy > 0.20 * idle, (idle, busy)


def test_concurrent_tagging_is_consistent(tagger):
    """Several threads sharing one Tagger must agree with the single-threaded answer."""
    texts = ["It costs $25.50.", "Call 555-0105.", "Résumé costs €25 on 1/2/2024."]
    expected = [tagger.tag(t) for t in texts]
    errors: list = []

    def work():
        try:
            for _ in range(30):
                for i, text in enumerate(texts):
                    if tagger.tag(text) != expected[i]:
                        errors.append(("mismatch", i))
        except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[:3]
