#! /bin/sh

GRAMMARS_DIR=${1:-"/workspace/sparrowhawk/documentation/grammars"}
TEST_DIR=${2:-"/workspace/tests/ko"}

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
  input=$TEST_DIR/data_inverse_text_normalization/test_cases_cardinal.txt
  runtest $input
}

testITNOrdinal() {
  input=$TEST_DIR/data_inverse_text_normalization/test_cases_ordinal.txt
  runtest $input
}

testITNDecimal() {
  input=$TEST_DIR/data_inverse_text_normalization/test_cases_decimal.txt
  runtest $input
}

testITNFraction() {
  input=$TEST_DIR/data_inverse_text_normalization/test_cases_fraction.txt
  runtest $input
}

testITNTime() {
  input=$TEST_DIR/data_inverse_text_normalization/test_cases_time.txt
  runtest $input
}

# Remove all command-line arguments
shift $#

# Load shUnit2
. /workspace/shunit2/shunit2