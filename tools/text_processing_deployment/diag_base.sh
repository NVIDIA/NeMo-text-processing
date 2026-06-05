#!/bin/bash
# Test whether base miniconda3:4.6.14 (Debian stretch / g++6) compiles openfst
# 1.7.9's fst/fst.h. Stretch is EOL, so archive.debian.org has an expired Release
# file -> apt needs Check-Valid-Until=false + allow-unauthenticated to install g++.

echo "########## Base: continuumio/miniconda3:4.6.14 (Debian stretch / g++6) ##########"
docker run --rm continuumio/miniconda3:4.6.14 bash -c '
  echo "deb http://archive.debian.org/debian stretch main contrib non-free" > /etc/apt/sources.list
  echo "--- apt update ---"
  apt-get -o Acquire::Check-Valid-Until=false update 2>&1 | tail -3
  echo "--- install build-essential ---"
  apt-get install -y --allow-unauthenticated build-essential 2>&1 | tail -3
  echo "--- installing thrax (conda) ---"
  conda install -c conda-forge thrax=1.3.4 -y >/dev/null 2>&1
  echo "--- g++ version ---"
  g++ --version | head -1
  echo "--- header present? ---"
  ls -la /opt/conda/include/fst/fst.h 2>&1
  printf "#include <fst/fst.h>\nint main(){return 0;}\n" > /tmp/t.cpp
  echo "--- compile test ---"
  if g++ -I/opt/conda/include /tmp/t.cpp -o /tmp/t 2>/tmp/err; then
    echo "RESULT: COMPILE_OK  (base 4.6.14 fixes the build)"
  else
    echo "RESULT: COMPILE_FAIL"
    head -10 /tmp/err
  fi
'
