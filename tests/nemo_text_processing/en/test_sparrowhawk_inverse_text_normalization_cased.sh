#! /bin/sh

PROJECT_DIR=/workspace/tests

runtest () {
  input=$1
  cd /workspace/sparrowhawk/documentation/grammars

  # read test file
  while read testcase; do
    IFS='~' read spoken written <<< $testcase
    denorm_pred=$(echo $spoken | normalizer_main --config=sparrowhawk_configuration.ascii_proto 2>&1 | tail -n 1)

    # trim white space
    written="$(echo -e "${written}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    denorm_pred="$(echo -e "${denorm_pred}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"

    # input expected actual
    assertEquals "$spoken" "$written" "$denorm_pred"
  done < "$input"
}

testITNCardinal() {
  input=$PROJECT_DIR/en/data_inverse_text_normalization/test_cases_cardinal
  runtest $input.txt
  runtest $input_cased.txt
}

testITNDate() {
  input=$PROJECT_DIR/en/data_inverse_text_normalization/test_cases_date
  runtest $input.txt
  runtest $input_cased.txt
}

testITNDecimal() {
  input=$PROJECT_DIR/en/data_inverse_text_normalization/test_cases_decimal
  runtest $input.txt
  runtest $input_cased.txt
}

testITNElectronic() {
  input=$PROJECT_DIR/en/data_inverse_text_normalization/test_cases_electronic
  runtest $input.txt
  runtest $input_cased.txt
}

testITNOrdinal() {
  input=$PROJECT_DIR/en/data_inverse_text_normalization/test_cases_ordinal
  runtest $input.txt
  runtest $input_cased.txt
}

testITNTime() {
  input=$PROJECT_DIR/en/data_inverse_text_normalization/test_cases_time
  runtest $input.txt
  runtest $input_cased.txt
}

testITNMeasure() {
  input=$PROJECT_DIR/en/data_inverse_text_normalization/test_cases_measure
  runtest $input.txt
  runtest $input_cased.txt
}

testITNMoney() {
  input=$PROJECT_DIR/en/data_inverse_text_normalization/test_cases_money
  runtest $input.txt
  runtest $input_cased.txt
}

testITNWhitelist() {
  input=$PROJECT_DIR/en/data_inverse_text_normalization/test_cases_whitelist
  runtest $input.txt
  runtest $input_cased.txt
}

testITNTelephone() {
  input=$PROJECT_DIR/en/data_inverse_text_normalization/test_cases_telephone
  runtest $input.txt
  runtest $input_cased.txt
}

testITNWord() {
  input=$PROJECT_DIR/en/data_inverse_text_normalization/test_cases_word
  runtest $input.txt
  runtest $input_cased.txt
}

# Load shUnit2
. $PROJECT_DIR/../shunit2/shunit2
