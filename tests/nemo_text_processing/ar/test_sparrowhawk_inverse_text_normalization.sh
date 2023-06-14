#! /bin/sh

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

testITNCardinal() {
  input=$PROJECT_DIR/ar/data_inverse_text_normalization/test_cases_cardinal.txt
  runtest $input
}

testITNDecimal() {
  input=$PROJECT_DIR/ar/data_inverse_text_normalization/test_cases_decimal.txt
  runtest $input
}

testITNMeasure() {
  input=$PROJECT_DIR/ar/data_inverse_text_normalization/test_cases_measure.txt
  runtest $input
}

testITNMoney() {
  input=$PROJECT_DIR/ar/data_inverse_text_normalization/test_cases_money.txt
  runtest $input
}

testITNWhitelist() {
  input=$PROJECT_DIR/ar/data_inverse_text_normalization/test_cases_whitelist.txt
  runtest $input
}

# Load shUnit2
. $PROJECT_DIR/../shunit2/shunit2
