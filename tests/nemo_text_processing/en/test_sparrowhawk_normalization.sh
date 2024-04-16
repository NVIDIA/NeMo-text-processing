#! /bin/sh
GRAMMARS_DIR=${1:-"/workspace/sparrowhawk/documentation/grammars"}
TEST_DIR=${2:-"/workspace/tests/en"}

runtest () {
  input=$1
  echo "INPUT is $input"
  cd ${GRAMMARS_DIR}

  # read test file
  while read testcase; do
    IFS='~' read written spoken <<< $testcase
    # replace non breaking space with breaking space
    # Use below if postprocessor is not used. Comment if it is used
    denorm_pred=$(echo $written | normalizer_main --config=sparrowhawk_configuration.ascii_proto 2>&1 | tail -n 1 | sed 's/\xC2\xA0/ /g')
    # Use below if postprocessor is  used. Comment if it is not used
    #denorm_pred=$(echo $written | normalizer_main --config=sparrowhawk_configuration_pp.ascii_proto 2>&1 | tail -n 1 | sed 's/\xC2\xA0/ /g')

    # trim white space
    spoken="$(echo -e "${spoken}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    denorm_pred="$(echo -e "${denorm_pred}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"

    # input expected actual
    assertEquals "$written" "$spoken" "$denorm_pred"
  done < "$input"
}

testTNSpecialText() {
  input=$TEST_DIR/data_text_normalization/test_cases_special_text.txt
  runtest $input
}

testTNCardinal() {
  input=$TEST_DIR/data_text_normalization/test_cases_cardinal.txt
  runtest $input
}

testTNDate() {
  input=$TEST_DIR/data_text_normalization/test_cases_date.txt
  runtest $input
}

testTNDecimal() {
  input=$TEST_DIR/data_text_normalization/test_cases_decimal.txt
  runtest $input
}

testTNRange() {
  input=$TEST_DIR/data_text_normalization/test_cases_range.txt
  runtest $input
}

testTNSerial() {
  input=$TEST_DIR/data_text_normalization/test_cases_serial.txt
  runtest $input
}

#testTNRoman() {
#  input=$TEST_DIR/data_text_normalization/test_cases_roman.txt
#  runtest $input
#}

testTNElectronic() {
  input=$TEST_DIR/data_text_normalization/test_cases_electronic.txt
  runtest $input
}

testTNFraction() {
  input=$TEST_DIR/data_text_normalization/test_cases_fraction.txt
  runtest $input
}

testTNMoney() {
  input=$TEST_DIR/data_text_normalization/test_cases_money.txt
  runtest $input
}

testTNOrdinal() {
  input=$TEST_DIR/data_text_normalization/test_cases_ordinal.txt
  runtest $input
}

testTNTelephone() {
  input=$TEST_DIR/data_text_normalization/test_cases_telephone.txt
  runtest $input
}

testTNTime() {
  input=$TEST_DIR/data_text_normalization/test_cases_time.txt
  runtest $input
}

testTNMeasure() {
  input=$TEST_DIR/data_text_normalization/test_cases_measure.txt
  runtest $input
}

testTNWhitelist() {
  input=$TEST_DIR/data_text_normalization/test_cases_whitelist.txt
  runtest $input
}

testTNWord() {
  input=$TEST_DIR/data_text_normalization/test_cases_word.txt
  runtest $input
}

testTNAddress() {
  input=$TEST_DIR/data_text_normalization/test_cases_address.txt
  runtest $input
}

testTNMath() {
  input=$TEST_DIR/data_text_normalization/test_cases_math.txt
  runtest $input
}

# Remove all command-line arguments
shift $#

# Load shUnit2
. /workspace/shunit2/shunit2
