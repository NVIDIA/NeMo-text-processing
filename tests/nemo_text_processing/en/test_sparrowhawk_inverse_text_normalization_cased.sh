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
  runtest $PROJECT_DIR/en/data_inverse_text_normalization/test_cases_cardinal.txt
  runtest $PROJECT_DIR/en/data_inverse_text_normalization/test_cases_cardinal_cased.txt
}

testITNDate() {
  runtest $PROJECT_DIR/en/data_inverse_text_normalization/test_cases_date.txt
  runtest $PROJECT_DIR/en/data_inverse_text_normalization/test_cases_date_cased.txt
}

testITNDecimal() {
  runtest $PROJECT_DIR/en/data_inverse_text_normalization/test_cases_decimal.txt
  runtest $PROJECT_DIR/en/data_inverse_text_normalization/test_cases_decimal_cased.txt
}

testITNElectronic() {
  runtest $PROJECT_DIR/en/data_inverse_text_normalization/test_cases_electronic.txt
  runtest $PROJECT_DIR/en/data_inverse_text_normalization/test_cases_electronic_cased.txt
}

testITNOrdinal() {
  runtest $PROJECT_DIR/en/data_inverse_text_normalization/test_cases_ordinal.txt
  runtest $PROJECT_DIR/en/data_inverse_text_normalization/test_cases_ordinal_cased.txt
}

testITNTime() {
  runtest $PROJECT_DIR/en/data_inverse_text_normalization/test_cases_time.txt
  runtest $PROJECT_DIR/en/data_inverse_text_normalization/test_cases_time_cased.txt
}

testITNMeasure() {
  runtest $PROJECT_DIR/en/data_inverse_text_normalization/test_cases_measure.txt
  runtest $PROJECT_DIR/en/data_inverse_text_normalization/test_cases_measure_cased.txt
}

testITNMoney() {
  runtest $PROJECT_DIR/en/data_inverse_text_normalization/test_cases_money.txt
  runtest $PROJECT_DIR/en/data_inverse_text_normalization/test_cases_money_cased.txt
}

testITNWhitelist() {
  runtest $PROJECT_DIR/en/data_inverse_text_normalization/test_cases_whitelist.txt
  runtest $PROJECT_DIR/en/data_inverse_text_normalization/test_cases_whitelist_cased.txt
}

testITNTelephone() {
  runtest $PROJECT_DIR/en/data_inverse_text_normalization/test_cases_telephone.txt
  runtest $PROJECT_DIR/en/data_inverse_text_normalization/test_cases_telephone_cased.txt
}

testITNWord() {
  runtest $PROJECT_DIR/en/data_inverse_text_normalization/test_cases_word.txt
  runtest $PROJECT_DIR/en/data_inverse_text_normalization/test_cases_word_cased.txt
}

# Load shUnit2
. $PROJECT_DIR/../shunit2/shunit2
