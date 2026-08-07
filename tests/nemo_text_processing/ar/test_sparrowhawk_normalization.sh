#! /bin/sh
GRAMMARS_DIR=${1:-"/workspace/sparrowhawk/documentation/grammars"}
TEST_DIR=${2:-"/workspace/tests/ar"}

runtest () {
  input=$1
  echo "INPUT is $input"
  cd ${GRAMMARS_DIR}

  while IFS= read -r testcase; do
    IFS='~' read -r written spoken <<< "$testcase"

    escaped_written=$(printf '%s' "$written" | sed 's/\\/\\\\/g')
    denorm_pred=$(echo "$escaped_written" | normalizer_main --config=sparrowhawk_configuration.ascii_proto 2>&1 | tail -n 1 | sed 's/\xC2\xA0/ /g')

    spoken="$(echo -e "${spoken}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    denorm_pred="$(echo -e "${denorm_pred}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"

    assertEquals "$written" "$spoken" "$denorm_pred"
  done < "$input"
}

# For test files stored as expected~input (spoken~written).
runtest_swapped () {
  input=$1
  echo "INPUT is $input"
  cd ${GRAMMARS_DIR}

  while IFS= read -r testcase; do
    IFS='~' read -r spoken written <<< "$testcase"

    escaped_written=$(printf '%s' "$written" | sed 's/\\/\\\\/g')
    denorm_pred=$(echo "$escaped_written" | normalizer_main --config=sparrowhawk_configuration.ascii_proto 2>&1 | tail -n 1 | sed 's/\xC2\xA0/ /g')

    spoken="$(echo -e "${spoken}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    denorm_pred="$(echo -e "${denorm_pred}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"

    assertEquals "$written" "$spoken" "$denorm_pred"
  done < "$input"
}

testTNCardinal() {
  input=$TEST_DIR/data_text_normalization/test_cases_cardinal.txt
  runtest $input
}

testTNDecimal() {
  input=$TEST_DIR/data_text_normalization/test_cases_decimal.txt
  runtest $input
}

testTNFraction() {
  input=$TEST_DIR/data_text_normalization/test_cases_fraction.txt
  runtest_swapped $input
}

testTNMeasure() {
  input=$TEST_DIR/data_text_normalization/test_cases_measure.txt
  runtest_swapped $input
}

testTNMoney() {
  input=$TEST_DIR/data_text_normalization/test_cases_money.txt
  runtest $input
}

# Remove all command-line arguments
shift $#

# Load shUnit2
. /workspace/shunit2/shunit2
