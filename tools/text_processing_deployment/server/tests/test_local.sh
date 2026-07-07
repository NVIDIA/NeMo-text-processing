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
# End-to-end smoke test of the HTTP server against the fake normalizer_main.
# Validates the pipe wrapper, engine pool, and all endpoints without needing the
# real Sparrowhawk image. Requires: fastapi, uvicorn, httpx (or curl).
#
#   PYTHON=/path/to/python bash server/tests/test_local.sh
set -euo pipefail

PYTHON=${PYTHON:-python3}
SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
SERVER_DIR=$(cd "$SCRIPT_DIR/.."; pwd)
WORK=$(mktemp -d)
trap 'kill "${SERVER_PID:-}" 2>/dev/null || true; rm -rf "$WORK"' EXIT

# Two serving dirs, each with a config marker (so the server enables the
# direction) and a MODE file the fake reads to pick its transform.
mkdir -p "$WORK/tn" "$WORK/itn"
: > "$WORK/tn/sparrowhawk_configuration.ascii_proto"
: > "$WORK/itn/sparrowhawk_configuration.ascii_proto"
echo upper > "$WORK/tn/MODE"
echo lower > "$WORK/itn/MODE"

chmod +x "$SCRIPT_DIR/fake_normalizer_main.py"

PORT=${PORT:-8137}
export SPARROWHAWK_BIN="$SCRIPT_DIR/fake_normalizer_main.py"
export STDBUF_PREFIX=""            # macOS has no stdbuf; the fake flushes itself
export TN_DIR="$WORK/tn"
export ITN_DIR="$WORK/itn"
export WORKERS_PER_DIRECTION=2
export IO_TIMEOUT=5
export PORT

echo "[test] starting server on :$PORT"
( cd "$SERVER_DIR" && "$PYTHON" -m uvicorn app:app --host 127.0.0.1 --port "$PORT" --log-level warning ) &
SERVER_PID=$!

# Wait for readiness.
for _ in $(seq 1 50); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
  sleep 0.2
done

fail=0
check () {
  local label=$1 got=$2 want=$3
  if [[ "$got" == "$want" ]]; then
    echo "  PASS $label -> $got"
  else
    echo "  FAIL $label -> got [$got] want [$want]"; fail=1
  fi
}

echo "[test] /health"; curl -fsS "http://127.0.0.1:$PORT/health"; echo

TN=$(curl -fsS -X POST "http://127.0.0.1:$PORT/tn" -H 'content-type: application/json' \
      -d '{"text":"δεκα μετρα"}' | "$PYTHON" -c 'import sys,json;print(json.load(sys.stdin)["results"][0])')
check "tn single" "$TN" "ΔΕΚΑ ΜΕΤΡΑ"

ITN=$(curl -fsS -X POST "http://127.0.0.1:$PORT/itn" -H 'content-type: application/json' \
      -d '{"text":"ΔΕΚΑΤΕΣΣΕΡΙΣ"}' | "$PYTHON" -c 'import sys,json;print(json.load(sys.stdin)["results"][0])')
check "itn single" "$ITN" "δεκατεσσερις"

BATCH=$(curl -fsS -X POST "http://127.0.0.1:$PORT/normalize" -H 'content-type: application/json' \
      -d '{"direction":"tn","texts":["ena","dyo tria"]}' | "$PYTHON" -c 'import sys,json;print("|".join(json.load(sys.stdin)["results"]))')
check "tn batch" "$BATCH" "ENA|DYO TRIA"

# many requests to exercise the pool + checkout/return under load
OK=1
for i in $(seq 1 30); do
  R=$(curl -fsS -X POST "http://127.0.0.1:$PORT/tn" -H 'content-type: application/json' -d "{\"text\":\"x$i\"}" \
        | "$PYTHON" -c 'import sys,json;print(json.load(sys.stdin)["results"][0])')
  [[ "$R" == "X$i" ]] || OK=0
done
check "pool load (30 reqs)" "$OK" "1"

if [[ $fail -eq 0 ]]; then echo "[test] ALL PASSED"; else echo "[test] FAILURES"; exit 1; fi
