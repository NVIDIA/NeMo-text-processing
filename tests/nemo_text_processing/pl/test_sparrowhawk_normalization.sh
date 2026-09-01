#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

GRAMMARS_DIR=${1:-"/workspace/sparrowhawk/documentation/grammars"}
PROJECT_DIR=${2:-"/workspace/tests/en"}

runtest() {
  input=$1
  echo "INPUT is $input"
  cd "${GRAMMARS_DIR}" || return 1

  while read -r testcase; do
    IFS='~' read -r written spoken <<< "$testcase"
    denorm_pred=$(echo "$written" | normalizer_main --config=sparrowhawk_configuration.ascii_proto 2>&1 | tail -n 1)

    spoken="$(echo -e "${spoken}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    denorm_pred="$(echo -e "${denorm_pred}" | sed -e 's/ / /g' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"

    assertEquals "$written" "$spoken" "$denorm_pred"
  done < "$input"
}

testTNCardinal() {
  runtest "$PROJECT_DIR/pl/data_text_normalization/test_cases_cardinal.txt"
}

testTNDate() {
  runtest "$PROJECT_DIR/pl/data_text_normalization/test_cases_date.txt"
}

testTNMeasure() {
  runtest "$PROJECT_DIR/pl/data_text_normalization/test_cases_measure.txt"
}

testTNOrdinal() {
  runtest "$PROJECT_DIR/pl/data_text_normalization/test_cases_ordinal.txt"
}

testTNRoman() {
  runtest "$PROJECT_DIR/pl/data_text_normalization/test_cases_roman.txt"
}

testTNTime() {
  runtest "$PROJECT_DIR/pl/data_text_normalization/test_cases_time.txt"
}

testTNWhitelist() {
  runtest "$PROJECT_DIR/pl/data_text_normalization/test_cases_whitelist.txt"
}

. "$PROJECT_DIR/../shunit2/shunit2"
