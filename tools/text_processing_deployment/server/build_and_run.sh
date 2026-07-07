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
# One-shot: (optionally) export grammars, ensure the Sparrowhawk base image
# exists, build the server image, and run it on :8000.
#
# Usage:
#   bash server/build_and_run.sh                       # Greek, cased, reuse existing .far
#   LANGUAGE=el INPUT_CASE=cased EXPORT=1 bash server/build_and_run.sh   # re-export first
#
# Must be run from tools/text_processing_deployment/ (the build context).
set -euo pipefail

LANGUAGE=${LANGUAGE:-el}
INPUT_CASE=${INPUT_CASE:-cased}
PORT=${PORT:-8000}
EXPORT=${EXPORT:-0}
IMAGE=${IMAGE:-nemo-sparrowhawk-server}
BASE_IMAGE=${BASE_IMAGE:-sparrowhawk:latest}

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
CTX_DIR=$(cd "$SCRIPT_DIR/.."; pwd)   # tools/text_processing_deployment
cd "$CTX_DIR"

TN_SRC="${LANGUAGE}_tn_grammars_${INPUT_CASE}"
ITN_SRC="${LANGUAGE}_itn_grammars_${INPUT_CASE}"

if [[ "$EXPORT" == "1" ]]; then
  echo "[build] exporting grammars for $LANGUAGE ($INPUT_CASE)"
  bash export_grammars.sh --GRAMMARS=tn_grammars  --LANGUAGE="$LANGUAGE" --INPUT_CASE="$INPUT_CASE" --MODE=export
  bash export_grammars.sh --GRAMMARS=itn_grammars --LANGUAGE="$LANGUAGE" --INPUT_CASE="$INPUT_CASE" --MODE=export
fi

if [[ ! -f "$TN_SRC/classify/tokenize_and_classify.far" && ! -f "$ITN_SRC/classify/tokenize_and_classify.far" ]]; then
  echo "[build] ERROR: no grammars found ($TN_SRC / $ITN_SRC). Run with EXPORT=1 or export first." >&2
  exit 1
fi

if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
  echo "[build] base image $BASE_IMAGE missing — building it (this is slow the first time)"
  bash docker/build.sh
fi

echo "[build] building $IMAGE"
docker build -f server/Dockerfile -t "$IMAGE" \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg GRAMMARS_TN_SRC="$TN_SRC" \
  --build-arg GRAMMARS_ITN_SRC="$ITN_SRC" \
  .

echo "[run] starting $IMAGE on :$PORT"
exec docker run --rm -it \
  -p "$PORT:8000" \
  --shm-size=1g \
  -e LANGUAGE="$LANGUAGE" \
  "$IMAGE"
