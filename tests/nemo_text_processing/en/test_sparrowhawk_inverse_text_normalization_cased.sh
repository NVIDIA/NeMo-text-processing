#! /bin/sh

GRAMMARS_DIR=${1:-"/workspace/sparrowhawk/documentation/grammars"}
TEST_DIR=${2:-"/workspace/tests/en"}

runtest () {
  input=$1
  echo "INPUT is $input"
  cd ${GRAMMARS_DIR}

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
  runtest $TEST_DIR/data_inverse_text_normalization/test_cases_cardinal.txt
  runtest $TEST_DIR/data_inverse_text_normalization/test_cases_cardinal_cased.txt
}

testITNDate() {
  runtest $TEST_DIR/data_inverse_text_normalization/test_cases_date.txt
  runtest $TEST_DIR/data_inverse_text_normalization/test_cases_date_cased.txt
}

testITNDecimal() {
  runtest $TEST_DIR/data_inverse_text_normalization/test_cases_decimal.txt
  runtest $TEST_DIR/data_inverse_text_normalization/test_cases_decimal_cased.txt
}

testITNElectronic() {
  runtest $TEST_DIR/data_inverse_text_normalization/test_cases_electronic.txt
  runtest $TEST_DIR/data_inverse_text_normalization/test_cases_electronic_cased.txt
}

testITNOrdinal() {
  runtest $TEST_DIR/data_inverse_text_normalization/test_cases_ordinal.txt
  runtest $TEST_DIR/data_inverse_text_normalization/test_cases_ordinal_cased.txt
}

testITNTime() {
  runtest $TEST_DIR/data_inverse_text_normalization/test_cases_time.txt
  runtest $TEST_DIR/data_inverse_text_normalization/test_cases_time_cased.txt
}

testITNMeasure() {
  runtest $TEST_DIR/data_inverse_text_normalization/test_cases_measure.txt
  runtest $TEST_DIR/data_inverse_text_normalization/test_cases_measure_cased.txt
}

testITNMoney() {
  runtest $TEST_DIR/data_inverse_text_normalization/test_cases_money.txt
  runtest $TEST_DIR/data_inverse_text_normalization/test_cases_money_cased.txt
}

testITNWhitelist() {
  runtest $TEST_DIR/data_inverse_text_normalization/test_cases_whitelist.txt
  runtest $TEST_DIR/data_inverse_text_normalization/test_cases_whitelist_cased.txt
}

testITNTelephone() {
  runtest $TEST_DIR/data_inverse_text_normalization/test_cases_telephone.txt
  runtest $TEST_DIR/data_inverse_text_normalization/test_cases_telephone_cased.txt
}

testITNWord() {
  runtest $TEST_DIR/data_inverse_text_normalization/test_cases_word.txt
  runtest $TEST_DIR/data_inverse_text_normalization/test_cases_word_cased.txt
}


# Remove all command-line arguments
shift $#

# Load shUnit2
. /workspace/shunit2/shunit2
