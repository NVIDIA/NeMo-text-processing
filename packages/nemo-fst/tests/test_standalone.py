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
install it, and every assertion that needs an oracle is unavailable there. They
need nothing but the extension and the tiny grammar checked in beside them.

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


def _run(cmd):
    """Run a toolchain command, or None if it is not usable here."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True)
    except OSError:
        return None
    return out.stdout if out.returncode == 0 else None


def test_exports_only_the_module_init_symbol():
    """Two OpenFst copies in one process are safe only if ours is invisible.

    Worth checking on every platform, not just the one that was developed on:
    the export table is restricted by a version script under GNU ld and by
    -exported_symbols_list under ld64, so this is the assertion that catches a
    platform where neither took.
    """
    so = nemo_fst._nemo_fst.__file__
    if sys.platform == "darwin":
        out = _run(["nm", "-gU", so])           # global, defined
    else:
        out = _run(["nm", "-D", "--defined-only", so])
    if out is None:
        pytest.skip("nm unavailable")
    # ld64 prefixes symbols with an underscore; GNU ld does not.
    exported = {line.split()[-1].lstrip("_") for line in out.splitlines() if line.strip()}
    assert exported == {"PyInit__nemo_fst"}, sorted(exported)


def test_no_external_openfst_dependency():
    """A relocatable wheel carries its OpenFst; it does not look for one."""
    so = nemo_fst._nemo_fst.__file__
    out = _run(["otool", "-L", so]) if sys.platform == "darwin" else _run(["readelf", "-d", so])
    if out is None:
        pytest.skip("otool/readelf unavailable")
    assert "libfst" not in out, out


def test_tagged_output_is_wellformed(toy_tagger):
    """Structure, not content: no oracle needed to know this much."""
    tagged = toy_tagger.tag("abc 42 xyz")
    assert tagged.startswith("tokens {"), tagged
    assert tagged.count("{") == tagged.count("}"), tagged
    assert 'cardinal { integer: "42" }' in tagged, tagged
    assert 'name: "abc"' in tagged, tagged


def test_tagging_is_deterministic(toy_tagger):
    """The same input must give the same answer every time.

    Worth asserting rather than assuming: the lookahead FST is shared across
    calls and carries precomputed reachability, so a bug that mutated it would
    show up here and nowhere else.
    """
    text = "alpha 12 beta 345 gamma"
    first = toy_tagger.tag(text)
    assert all(toy_tagger.tag(text) == first for _ in range(20))


def test_empty_and_whitespace_input(toy_tagger):
    for text in ("", " ", "\n", "\t "):
        toy_tagger.tag(text)  # must not raise


def test_multibyte_input_round_trips(toy_tagger):
    """A byte relabelled wrongly does not raise, it silently fails to match."""
    for text in ("Résumé costs €25.", "日本語 12 items.", "Emoji 😀 and 7 things."):
        tagged = toy_tagger.tag(text)
        assert tagged.startswith("tokens {"), (text, tagged)


def test_missing_far_raises_cleanly(tmp_path):
    with pytest.raises(Exception):  # noqa: B017 -- any clean failure, not a crash
        nemo_fst.Tagger.from_far(tmp_path / "does-not-exist.far", cache_dir=tmp_path)


def test_missing_key_raises_cleanly(toy_far, tmp_path):
    with pytest.raises(RuntimeError, match="not in FAR"):
        nemo_fst.Tagger.from_far(toy_far, key="no-such-key", cache_dir=tmp_path)


def test_tag_releases_the_gil(toy_tagger):
    """A sibling thread must keep running while a composition does.

    Measured as a ratio against an idle baseline rather than a wall-clock
    threshold, so it means the same thing on a slow machine.
    """
    # Long enough that one call is tens of milliseconds, or there is nothing
    # for the sibling thread to be starved of.
    text = "alpha 12 beta 345 gamma delta 6789 " * 200

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
            toy_tagger.tag(text) if load else time.sleep(0.005)
        stop.set()
        thread.join()
        return out[0]

    idle, busy = measure(False), measure(True)
    assert busy > 0.20 * idle, (idle, busy)


def test_concurrent_tagging_is_consistent(toy_tagger):
    """Several threads sharing one Tagger must agree with the single-threaded answer."""
    texts = ["alpha 12", "beta", "gamma 345 delta"]
    expected = [toy_tagger.tag(t) for t in texts]
    errors: list = []

    def work():
        try:
            for _ in range(30):
                for i, text in enumerate(texts):
                    if toy_tagger.tag(text) != expected[i]:
                        errors.append(("mismatch", i))
        except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[:3]
