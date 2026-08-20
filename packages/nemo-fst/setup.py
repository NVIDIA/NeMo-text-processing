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

"""`pip install .` given an OpenFst prefix.

    OPENFST_PREFIX=/opt/openfst pip install .

`scripts/build_openfst.sh` produces such a prefix, and cibuildwheel runs it once
per container so every wheel links the same OpenFst.

Prefer the static archives when they are there: linking them leaves the wheel
with no external OpenFst dependency and no rpath, which is what makes it
relocatable. A prefix with only shared libraries still works for a local build,
with an rpath pointing back at it.

Note what is *not* here: no libfstlookahead, no lookahead plugin directory. The
extension names fst::StdOLabelLookAheadFst -- a template in <fst/matcher-fst.h>
-- rather than looking the type up in OpenFst's registry, so the registration
object that everything else has to force-link is never needed, and neither is
--whole-archive.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

PREFIX = Path(os.environ.get("OPENFST_PREFIX", "/usr/local"))
OPENFST_VERSION = os.environ.get("OPENFST_VERSION", "1.8.4")
HERE = Path(__file__).parent

if not (PREFIX / "include" / "fst" / "matcher-fst.h").exists():
    sys.exit(
        f"nemo-fst: no OpenFst headers under {PREFIX}\n"
        f"\n"
        f"This package needs an OpenFst built with --enable-lookahead-fsts, which\n"
        f"no distribution packages -- Homebrew's openfst and the usual Linux\n"
        f"packages are all built without it. Build one:\n"
        f"\n"
        f"    bash {HERE / 'scripts' / 'build_openfst.sh'} $HOME/.local/openfst\n"
        f"    export OPENFST_PREFIX=$HOME/.local/openfst\n"
        f"\n"
        f"then retry. Set OPENFST_PREFIX to an existing prefix if you already have\n"
        f"a lookahead-enabled build.\n"
    )

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

STATIC_LIBS = ["libfstfar.a", "libfst.a"]
static = all((PREFIX / "lib" / name).exists() for name in STATIC_LIBS)

PROBE_SYMBOL = "nemo_fst_probe"
PROBE_SRC = f'extern "C" int {PROBE_SYMBOL}(void) {{ return 0; }}\n'

# Each candidate is (flag template, contents of the file it points at, if any).
# The probe has to name a symbol the probe object actually defines: ld64 fails
# an export list naming something absent, which would make the probe reject a
# flag that works perfectly well on the real link.
HIDE_CANDIDATES = [
    ("-Wl,--exclude-libs,ALL", None, None),
    (
        "-Wl,--version-script,{file}",
        f"{{ global: {PROBE_SYMBOL}; local: *; }};\n",
        HERE / "src" / "nemo_fst.map",
    ),
    (
        "-Wl,-exported_symbols_list,{file}",
        f"_{PROBE_SYMBOL}\n",
        HERE / "src" / "nemo_fst.exported_symbols",
    ),
]


def linker_accepts(template: str, probe_file_contents) -> bool:
    """Does this toolchain's linker take this flag?

    Asked rather than inferred from sys.platform: these are GNU ld spellings
    that ld64 rejects and vice versa, and a wrong guess means the OpenFst
    symbols statically linked in here stay visible, which is exactly what must
    not happen when pynini brings its own OpenFst into the same process.
    """
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "probe.cc"
        src.write_text(PROBE_SRC)
        flag = template
        if probe_file_contents is not None:
            probe_file = Path(tmp) / "probe.syms"
            probe_file.write_text(probe_file_contents)
            flag = template.format(file=probe_file)
        cmd = shlex.split(os.environ.get("CXX", "c++"))
        cmd += ["-shared", "-fPIC", str(src), "-o", str(Path(tmp) / "probe.so"), flag]
        try:
            return subprocess.run(cmd, capture_output=True).returncode == 0
        except OSError:
            return False


HIDE = [
    template.format(file=real_file) if real_file else template
    for template, probe_contents, real_file in HIDE_CANDIDATES
    if linker_accepts(template, probe_contents)
]
if not HIDE:
    print("nemo-fst: WARNING no supported symbol-hiding linker flag; OpenFst symbols "
          "will be visible and may collide with pynini's copy")

if static:
    # Archives passed as objects, so nothing is left to resolve at load time.
    link_args = [str(PREFIX / "lib" / name) for name in STATIC_LIBS] + HIDE
    libraries: list[str] = []
else:
    link_args = [f"-L{PREFIX / 'lib'}", f"-Wl,-rpath,{PREFIX / 'lib'}"] + HIDE
    libraries = ["fstfar", "fst"]

class BuildExtAndVerify(build_ext):
    """Build, then check that nothing but the module init symbol is exported.

    The flags above are probed, and a probe can be wrong -- one was, and the
    result was a macOS build with every OpenFst symbol visible. Since the whole
    coexistence argument rests on those symbols being hidden, a build that
    fails to hide them should not be packaged.
    """

    def run(self):
        super().run()
        for ext in self.extensions:
            path = self.get_ext_fullpath(ext.name)
            leaked = _exported_symbols(path)
            if leaked is None:
                print(f"nemo-fst: cannot inspect {path}; skipping export-table check")
                continue
            unexpected = leaked - {"PyInit__nemo_fst"}
            if unexpected:
                raise SystemExit(
                    f"nemo-fst: {len(unexpected)} symbols are exported besides the module "
                    f"init symbol, e.g. {sorted(unexpected)[:5]}.\n"
                    f"Linked with: {HIDE or 'no symbol-hiding flag'}\n"
                    f"Those would be visible to pynini's OpenFst in the same process."
                )
            print(f"nemo-fst: export table clean ({sorted(leaked)})")


def _exported_symbols(path):
    """Defined, globally visible symbols in `path`, or None if nm cannot say."""
    cmd = ["nm", "-gU", str(path)] if sys.platform == "darwin" else \
          ["nm", "-D", "--defined-only", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True)
    except OSError:
        return None
    if out.returncode != 0:
        return None
    # ld64 prefixes with an underscore; GNU ld does not.
    return {line.split()[-1].lstrip("_") for line in out.stdout.splitlines() if line.strip()}


setup(
    cmdclass={"build_ext": BuildExtAndVerify},
    ext_modules=[
        Pybind11Extension(
            "nemo_fst._nemo_fst",
            ["src/nemo_fst.cc"],
            include_dirs=[str(PREFIX / "include")],
            libraries=libraries,
            extra_compile_args=["-O3", "-fvisibility=hidden",
                                "-fvisibility-inlines-hidden"],
            extra_link_args=link_args,
            define_macros=[("NEMO_FST_OPENFST_VERSION",
                            f'"{OPENFST_VERSION}"')],
            cxx_std=17,
        )
    ],
)
