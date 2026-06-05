#!/bin/bash
# Diagnostic: why does sparrowhawk's ./configure fail with "fst/fst.h not found"?
# The header exists at /opt/conda/include/fst/fst.h, so we test whether it
# actually COMPILES (autoconf reports "not found" when the test-compile fails).

set -x
docker run --rm continuumio/miniconda3 bash -c '
  conda install -c conda-forge thrax=1.3.4 -y >/dev/null 2>&1
  apt-get update >/dev/null 2>&1
  apt-get install -y g++ >/dev/null 2>&1
  printf "#include <fst/fst.h>\nint main(){return 0;}\n" > /tmp/t.cpp
  echo "=== g++ version ==="
  g++ --version | head -1
  echo "=== PLAIN (default std) ==="
  g++ -I/opt/conda/include /tmp/t.cpp -o /tmp/t 2>&1 | head -15
  echo "=== CPP14 ==="
  g++ -std=c++14 -I/opt/conda/include /tmp/t.cpp -o /tmp/t 2>&1 | head -15
  echo "=== CPP17 ==="
  g++ -std=c++17 -I/opt/conda/include /tmp/t.cpp -o /tmp/t 2>&1 | head -15
  echo "=== DONE ==="
'
