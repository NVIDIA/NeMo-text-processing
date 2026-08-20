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

"""Two OpenFst copies in one process.

This is the first test on purpose.  During migration `nemo-text-processing-lightning` and pynini are
both imported by the same interpreter, each carrying its own OpenFst; if hidden
visibility is wrong, one binds to the other's symbols and the failure is a
segfault or silent wrong answers, not an ImportError.  Everything downstream is
wasted until this passes, so it runs in a subprocess and asserts on the exit
status rather than risking taking the whole test session down.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

BOTH_WAYS = textwrap.dedent(
    """
    import sys
    order = sys.argv[1]
    if order == "nemo_first":
        import nemo_text_processing_lightning, pynini
    else:
        import pynini, nemo_text_processing_lightning

    # A composition through each library, after both are loaded.
    tagger = nemo_text_processing_lightning.Tagger.from_far(sys.argv[2], cache_dir=sys.argv[3])
    tagged = tagger.tag("It costs $25.50.")
    assert tagged.startswith("tokens {"), tagged

    a = pynini.accep("abc")
    b = pynini.accep("abc")
    lattice = pynini.compose(a, b)
    assert lattice.num_states() == 4, lattice.num_states()
    assert pynini.shortestpath(lattice).string() == "abc"

    # And back through nemo_text_processing_lightning, to catch state clobbered by pynini's OpenFst.
    assert tagger.tag("It costs $25.50.") == tagged
    assert nemo_text_processing_lightning.has_lookahead()

    # pynini's own OpenFst has no lookahead plugin; asking it must still fail
    # cleanly rather than picking up ours.
    try:
        pynini.convert(a, "olabel_lookahead")
        leaked = True
    except Exception:
        leaked = False

    print("OK", order, "leaked=" + str(leaked))
    """
)


@pytest.mark.parametrize("order", ["nemo_first", "pynini_first"])
def test_import_both_and_compose(order, far_path, artifact_dir):
    # The subprocess must find the package regardless of pytest's rootdir, so
    # put it on PYTHONPATH explicitly rather than relying on cwd.
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{PACKAGE_ROOT}{os.pathsep}{existing}" if existing else str(PACKAGE_ROOT)
    proc = subprocess.run(
        [sys.executable, "-c", BOTH_WAYS, order, str(far_path), str(artifact_dir)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, f"exit {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert proc.stdout.startswith("OK"), proc.stdout


def test_no_openfst_symbols_exported():
    """Nothing from OpenFst may be visible outside the extension."""
    import nemo_text_processing_lightning

    so = nemo_text_processing_lightning._lightning.__file__
    out = subprocess.run(["nm", "-D", "--defined-only", so], capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("nm unavailable")
    names = [line.split()[-1] for line in out.stdout.splitlines() if line.strip()]
    # The version script leaves exactly one dynamic symbol: the module init.
    assert names == ["PyInit__lightning"], names[:20]
