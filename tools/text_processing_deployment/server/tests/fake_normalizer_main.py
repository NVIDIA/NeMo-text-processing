#!/usr/bin/env python3
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
# Stand-in for Sparrowhawk's `normalizer_main` used only for local testing of
# the server wrapper (the real binary needs the Linux x86_64 Sparrowhawk image).
#
# It mimics the real contract: reads one utterance per line from stdin, writes
# exactly one normalized line to stdout, and emits diagnostic noise on stderr —
# so it exercises the wrapper's "stdout = result, stderr = drained" split.
#
# Direction is read from a `MODE` file in the current working directory
# (`upper` or `lower`), mirroring how the real setup selects grammars by cwd.
import os
import sys

mode = "upper"
try:
    with open(os.path.join(os.getcwd(), "MODE")) as fh:
        mode = fh.read().strip() or "upper"
except FileNotFoundError:
    pass

# Simulate the one-time grammar load, logged to stderr like the real binary.
print(f"[fake normalizer] loaded grammars (mode={mode})", file=sys.stderr, flush=True)

for line in sys.stdin:
    text = line.rstrip("\n")
    print(f"[fake normalizer] normalizing: {text!r}", file=sys.stderr, flush=True)
    result = text.upper() if mode == "upper" else text.lower()
    sys.stdout.write(result + "\n")
    sys.stdout.flush()
