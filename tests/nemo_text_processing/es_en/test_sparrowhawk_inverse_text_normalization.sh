#! /bin/sh
# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

PROJECT_DIR=/workspace/tests

runtest () {
  input=$1
  cd /workspace/sparrowhawk/documentation/grammars

  # read test file
  while read testcase; do
    IFS='~' read spoken written <<< $testcase
    denorm_pred=$(echo $written | normalizer_main --config=sparrowhawk_configuration.ascii_proto 2>&1 | tail -n 1 | sed 's/\xC2\xA0/ /g')

    # trim white space
    written="$(echo -e "${written}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    denorm_pred="$(echo -e "${denorm_pred}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"

    # input expected actual
    assertEquals "$spoken" "$written" "$denorm_pred"
  done < "$input"
}

testITNDecimal() {
  input=$PROJECT_DIR/es_en/data_inverse_text_normalization/test_cases_decimal.txt
  runtest $input
}

testITNCardinal() {
  input=$PROJECT_DIR/es_en/data_inverse_text_normalization/test_cases_cardinal.txt
  runtest $input
}

testITNDate() {
  input=$PROJECT_DIR/es_en/data_inverse_text_normalization/test_cases_date.txt
  runtest $input
}



testITNOrdinal() {
  input=$PROJECT_DIR/es_en/data_inverse_text_normalization/test_cases_ordinal.txt
  runtest $input
}

testITNTime() {
  input=$PROJECT_DIR/es_en/data_inverse_text_normalization/test_cases_time.txt
  runtest $input
}

testITNMeasure() {
  input=$PROJECT_DIR/es_en/data_inverse_text_normalization/test_cases_measure.txt
  runtest $input
}

testITNMoney() {
  input=$PROJECT_DIR/es_en/data_inverse_text_normalization/test_cases_money.txt
  runtest $input
}

testITNWhitelist() {
  input=$PROJECT_DIR/es_en/data_inverse_text_normalization/test_cases_whitelist.txt
  runtest $input
}

# Load shUnit2
. $PROJECT_DIR/../shunit2/shunit2
