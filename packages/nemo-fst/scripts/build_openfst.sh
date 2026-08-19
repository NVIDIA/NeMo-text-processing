#!/usr/bin/env bash
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
#
# Build the OpenFst this package links against.
#
#   scripts/build_openfst.sh [prefix]
#
# cibuildwheel runs this once per container via CIBW_BEFORE_ALL, so every wheel
# is built against an OpenFst configured the same way.
#
# The flags that matter:
#   --enable-lookahead-fsts   the whole point; without it olabel_lookahead does
#                             not exist and this package has nothing to offer
#   --enable-far              FarReader, for reading grammars out of a .far
#   --enable-static           we link the archives, so the wheel carries no
#     --disable-shared        external OpenFst dependency and stays relocatable
#   --with-pic                archives get linked into a shared object

set -euo pipefail

OPENFST_VERSION="${OPENFST_VERSION:-1.8.4}"
PREFIX="${1:-${OPENFST_PREFIX:-/usr/local}}"
URL="https://www.openfst.org/twiki/pub/FST/FstDownload/openfst-${OPENFST_VERSION}.tar.gz"
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

echo "building OpenFst ${OPENFST_VERSION} into ${PREFIX}"
curl -fsSL "$URL" | tar xz -C "$BUILD_DIR" --strip-components=1
cd "$BUILD_DIR"

# Upstream bug in 1.8.3, fixed in 1.8.4, patched here for anyone pinning the
# older release: VectorHashBiTable's copy constructor initialises
# selector_ from `table.s_`, and no such member exists -- it is `selector_`.
# Nothing instantiates that constructor, so GCC never checks it, but `table`
# has the type of the current instantiation, so Clang resolves the member at
# definition time and rejects the header. Fatal on macOS, invisible on Linux.
if grep -q 'selector_(table\.s_)' src/include/fst/bi-table.h; then
    echo "patching bi-table.h: table.s_ -> table.selector_"
    sed -i.bak 's/selector_(table\.s_)/selector_(table.selector_)/' src/include/fst/bi-table.h
    rm -f src/include/fst/bi-table.h.bak
fi

./configure \
    --prefix="$PREFIX" \
    --enable-far \
    --enable-lookahead-fsts \
    --enable-static \
    --disable-shared \
    --with-pic \
    CXXFLAGS="-O2 -fPIC"

make -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
make install

test -f "$PREFIX/lib/libfst.a" || { echo "no libfst.a in $PREFIX/lib" >&2; exit 1; }
test -f "$PREFIX/include/fst/matcher-fst.h" || { echo "no headers in $PREFIX" >&2; exit 1; }
echo "OpenFst ${OPENFST_VERSION} installed to ${PREFIX}"
