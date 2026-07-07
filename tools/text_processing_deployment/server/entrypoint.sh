#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
# Assembles a self-contained Sparrowhawk serving directory for each available
# direction, then starts the HTTP server.
#
# For each direction we lay out (relative paths matter — normalizer_main is run
# with cwd set to the direction dir):
#   <dir>/sparrowhawk_configuration.ascii_proto
#   <dir>/tokenizer.ascii_proto            -> classify/tokenize_and_classify.far
#   <dir>/verbalizer.ascii_proto           -> verbalize/verbalize.far
#   <dir>/sentence_boundary_exceptions.txt
#   <dir>/classify/tokenize_and_classify.far
#   <dir>/verbalize/verbalize.far
set -euo pipefail

CONFIG_SRC=${CONFIG_SRC:-/app/configs}
SERVE_ROOT=${SERVE_ROOT:-/workspace/serve}

# Where the exported .far grammars live. Baked into the image by default, but can
# be overridden at runtime with a bind mount (e.g. -e GRAMMARS_TN=/mnt/tn).
GRAMMARS_TN=${GRAMMARS_TN:-/app/grammars/tn}
GRAMMARS_ITN=${GRAMMARS_ITN:-/app/grammars/itn}

export TN_DIR=${TN_DIR:-$SERVE_ROOT/tn}
export ITN_DIR=${ITN_DIR:-$SERVE_ROOT/itn}

setup_direction () {
  local src=$1 dst=$2 name=$3
  local classify="$src/classify/tokenize_and_classify.far"
  local verbalize="$src/verbalize/verbalize.far"
  if [[ ! -f "$classify" || ! -f "$verbalize" ]]; then
    echo "[entrypoint] '$name': grammars not found under $src — direction disabled"
    return
  fi
  echo "[entrypoint] '$name': wiring grammars from $src -> $dst"
  mkdir -p "$dst/classify" "$dst/verbalize"
  cp -f "$classify" "$dst/classify/tokenize_and_classify.far"
  cp -f "$verbalize" "$dst/verbalize/verbalize.far"
  cp -f "$CONFIG_SRC"/sparrowhawk_configuration.ascii_proto \
        "$CONFIG_SRC"/tokenizer.ascii_proto \
        "$CONFIG_SRC"/verbalizer.ascii_proto \
        "$CONFIG_SRC"/sentence_boundary_exceptions.txt \
        "$dst/"
}

setup_direction "$GRAMMARS_TN" "$TN_DIR" "tn"
setup_direction "$GRAMMARS_ITN" "$ITN_DIR" "itn"

exec uvicorn app:app \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8000}" \
  --workers 1
