#! /bin/sh

GRAMMARS_DIR=${1:-"/workspace/sparrowhawk/documentation/grammars"}
TEST_DIR=${2:-"/workspace/tests"}

runtest () {
  input=$1
  echo "INPUT is $input"
  cd ${GRAMMARS_DIR}

  # read test file
  while read testcase; do
    IFS='~' read spoken written <<< $testcase
    denorm_pred=$(echo $spoken | normalizer_main --config=sparrowhawk_configuration_pp.ascii_proto 2>&1 | tail -n 1 | sed 's/\xC2\xA0/ /g')

    # trim white space
    written="$(echo -e "${written}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    denorm_pred="$(echo -e "${denorm_pred}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"

    # input expected actual
    assertEquals "$spoken" "$written" "$denorm_pred"
  done < "$input"
}
testITNFractionText() {
  input=$TEST_DIR/ja/data_inverse_text_normalization/test_cases_fraction.txt
  runtest $input
}
testITNCardinalText() {
  input=$TEST_DIR/ja/data_inverse_text_normalization/test_cases_cardinal.txt
  runtest $input
}
testITNOrdinalText() {
  input=$TEST_DIR/ja/data_inverse_text_normalization/test_cases_ordinal.txt
  runtest $input
}
testITNDateText() {
  input=$TEST_DIR/ja/data_inverse_text_normalization/test_cases_date.txt
  runtest $input
}
testITNDecimalText() {
  input=$TEST_DIR/ja/data_inverse_text_normalization/test_cases_decimal.txt
  runtest $input
}
testITNTimeText() {
  input=$TEST_DIR/ja/data_inverse_text_normalization/test_cases_time.txt
  runtest $input
}


# Remove all command-line arguments
shift $#

# Load shUnit2
#. $PROJECT_DIR/../shunit2/shunit2
. /workspace/shunit2/shunit2