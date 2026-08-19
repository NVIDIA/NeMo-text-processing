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
import sys
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

PREFIX = Path(os.environ.get("OPENFST_PREFIX", "/usr/local"))
OPENFST_VERSION = os.environ.get("OPENFST_VERSION", "1.8.3")

if not (PREFIX / "include" / "fst" / "matcher-fst.h").exists():
    sys.exit(f"setup.py: no OpenFst headers under {PREFIX}; set OPENFST_PREFIX")

STATIC_LIBS = ["libfstfar.a", "libfst.a"]
static = all((PREFIX / "lib" / name).exists() for name in STATIC_LIBS)

# Coexistence with pynini, which carries its own OpenFst into the same process:
# nothing of ours may be visible for it to bind to.
HIDE = [
    "-Wl,--exclude-libs,ALL",
    f"-Wl,--version-script,{Path(__file__).parent / 'src' / 'nemo_fst.map'}",
]

if static:
    # Archives passed as objects, so nothing is left to resolve at load time.
    link_args = [str(PREFIX / "lib" / name) for name in STATIC_LIBS] + HIDE
    libraries: list[str] = []
else:
    link_args = [f"-L{PREFIX / 'lib'}", f"-Wl,-rpath,{PREFIX / 'lib'}"] + HIDE
    libraries = ["fstfar", "fst"]

setup(
    cmdclass={"build_ext": build_ext},
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
