#!/usr/bin/env bash
# Build the nemo_text_processing_lightning extension from a clean checkout.
#
#   OPENFST_PREFIX=/path/to/openfst ./build.sh
#
# Env:
#   OPENFST_PREFIX  OpenFst built with --enable-far and --enable-lookahead-fsts.
#                   Lookahead needs no library: the matcher types are templates
#                   in <fst/matcher-fst.h> and we name the concrete type
#                   StdOLabelLookAheadFst directly, so neither the dlopen
#                   plugins in lib/fst nor libfstlookahead are involved.
#   PYTHON          interpreter to build against (default: python3)
#   PYBIND11_DIR    pybind11 include dir (default: asked of $PYBIND11_PYTHON)
#
# Two linker settings are not optional, both for coexistence with pynini, which
# carries its own OpenFst into the same process:
#   -fvisibility=hidden          nothing but PyInit__lightning is exported
#   -Wl,--exclude-libs,ALL       no symbol from a linked archive is re-exported
#
# STATIC LINKING (not done here): this prefix was configured
# --disable-static, so we link the shared libs with -Wl,-rpath.  A static build
# must additionally wrap -Wl,--whole-archive around libfstlookahead.a *if* it
# uses the fst-type registry (fstconvert-style, by name).  This file does not:
# it instantiates the template, so the registration is unreferenced and the
# archive is not needed at all.  If a future entry point does go through the
# registry, --whole-archive becomes mandatory -- nothing references a lookahead
# symbol, the registrations are pure static-init side effects, and the linker
# drops the archive silently.  The dynamic equivalent is -Wl,--no-as-needed.

set -euo pipefail
cd "$(dirname "$0")"

OPENFST_PREFIX="${OPENFST_PREFIX:-/usr/local}"
PYTHON="${PYTHON:-python3}"
PYBIND11_PYTHON="${PYBIND11_PYTHON:-$PYTHON}"
OPENFST_VERSION="${OPENFST_VERSION:-1.8.4}"

if [ ! -f "$OPENFST_PREFIX/include/fst/matcher-fst.h" ]; then
  echo "build.sh: no OpenFst headers under $OPENFST_PREFIX" >&2
  echo "  build one with: bash scripts/build_openfst.sh \$HOME/.local/openfst" >&2
  echo "  then: export OPENFST_PREFIX=\$HOME/.local/openfst" >&2
  exit 1
fi

PYBIND11_DIR="${PYBIND11_DIR:-$("$PYBIND11_PYTHON" -c 'import pybind11;print(pybind11.get_include())')}"
PY_INCLUDE="$("$PYTHON" -c 'import sysconfig;print(sysconfig.get_paths()["include"])')"
EXT_SUFFIX="$("$PYTHON" -c 'import sysconfig;print(sysconfig.get_config_var("EXT_SUFFIX"))')"

OUT="nemo_text_processing_lightning/_lightning${EXT_SUFFIX}"

echo "building $OUT"
# Static archives when the prefix has them: the result then carries no external
# OpenFst dependency and needs no rpath, which is what makes a wheel relocatable.
if [ -f "$OPENFST_PREFIX/lib/libfst.a" ] && [ -f "$OPENFST_PREFIX/lib/libfstfar.a" ]; then
  FST_LINK=("$OPENFST_PREFIX/lib/libfstfar.a" "$OPENFST_PREFIX/lib/libfst.a")
  echo "  linking OpenFst statically"
else
  FST_LINK=(-L"$OPENFST_PREFIX/lib" -lfstfar -lfst -Wl,-rpath,"$OPENFST_PREFIX/lib")
  echo "  linking OpenFst dynamically (no static archives in $OPENFST_PREFIX/lib)"
fi

# Restricting the export table: offer every spelling the two linkers use and
# keep the ones this toolchain takes. Asked rather than inferred from `uname`,
# because a wrong guess leaves OpenFst's symbols visible to pynini's copy.
#
# The probe names a symbol the probe object actually defines: ld64 fails an
# export list naming something absent, so probing with the real list would
# reject a flag that works on the real link.
PROBE_DIR="$(mktemp -d)"
trap 'rm -rf "$PROBE_DIR"' EXIT
printf 'extern "C" int lightning_probe(void){return 0;}\n' > "$PROBE_DIR/probe.cc"
printf '{ global: lightning_probe; local: *; };\n' > "$PROBE_DIR/probe.map"
printf '_lightning_probe\n' > "$PROBE_DIR/probe.syms"

linker_accepts() {
  "${CXX:-c++}" -shared -fPIC "$PROBE_DIR/probe.cc" -o "$PROBE_DIR/probe.so" "$1" 2>/dev/null
}

HIDE=()
linker_accepts "-Wl,--exclude-libs,ALL" && HIDE+=(-Wl,--exclude-libs,ALL)
linker_accepts "-Wl,--version-script,$PROBE_DIR/probe.map" &&
  HIDE+=(-Wl,--version-script,src/lightning.map)
linker_accepts "-Wl,-exported_symbols_list,$PROBE_DIR/probe.syms" &&
  HIDE+=(-Wl,-exported_symbols_list,src/lightning.exported_symbols)
echo "  symbol hiding: ${HIDE[*]:-NONE — OpenFst symbols will be visible}"

"${CXX:-c++}" -O3 -std=c++17 -shared -fPIC -fvisibility=hidden -fvisibility-inlines-hidden \
    -DNDEBUG "-DNEMO_TPL_OPENFST_VERSION=\"$OPENFST_VERSION\"" \
    -I"$PY_INCLUDE" -I"$PYBIND11_DIR" -I"$OPENFST_PREFIX/include" \
    src/lightning.cc \
    -o "$OUT" \
    "${FST_LINK[@]}" \
    ${HIDE[@]+"${HIDE[@]}"}

echo "built $OUT"

# A probe can be wrong; the export table cannot.
if command -v nm >/dev/null 2>&1; then
  if [ "$(uname -s)" = "Darwin" ]; then EXPORTED=$(nm -gU "$OUT" 2>/dev/null | awk '{print $NF}' | sed 's/^_//')
  else EXPORTED=$(nm -D --defined-only "$OUT" 2>/dev/null | awk '{print $NF}'); fi
  UNEXPECTED=$(printf '%s\n' "$EXPORTED" | grep -v '^PyInit__lightning$' | grep -v '^$' || true)
  if [ -n "$UNEXPECTED" ]; then
    echo "build.sh: $(printf '%s\n' "$UNEXPECTED" | wc -l) symbols exported besides the module init symbol," >&2
    echo "  e.g. $(printf '%s\n' "$UNEXPECTED" | head -3 | tr '\n' ' ')" >&2
    echo "  those would be visible to pynini's OpenFst in the same process." >&2
    exit 1
  fi
  echo "  export table clean"
fi
"$PYTHON" -c "
import sys; sys.path.insert(0, '.')
import nemo_text_processing_lightning
print('has_lookahead:', nemo_text_processing_lightning.has_lookahead(), '| openfst', nemo_text_processing_lightning.__openfst_version__)
assert nemo_text_processing_lightning.has_lookahead(), 'lookahead types are not usable in this build'
"
