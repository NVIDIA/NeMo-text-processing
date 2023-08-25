#! /bin/sh

PROJECT_DIR=/workspace/tests

runtest () {
  input=$1
  cd /workspace/sparrowhawk/documentation/grammars

  # read test file
  while read testcase; do
    IFS='~' read written spoken <<< $testcase
    denorm_pred=$(echo $written | normalizer_main --config=sparrowhawk_configuration.ascii_proto 2>&1 | tail -n 1)

    # trim white space
    spoken="$(echo -e "${spoken}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    denorm_pred="$(echo -e "${denorm_pred}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"

    # input expected actual
    assertEquals "$written" "$spoken" "$denorm_pred"
  done < "$input"
}

testTNCardinal() {
  input=$PROJECT_DIR/fr/data_text_normalization/test_cases_cardinal.txt
  runtest $input
}

testTNDecimal() {
  input=$PROJECT_DIR/fr/data_text_normalization/test_cases_decimal.txt
  runtest $input
}

testTNFraction() {
  input=$PROJECT_DIR/fr/data_text_normalization/test_cases_fraction.txt
  runtest $input
}

testTNOrdinal() {
  input=$PROJECT_DIR/fr/data_text_normalization/test_cases_ordinal.txt
  runtest $input
}

# Load shUnit2
. $PROJECT_DIR/../shunit2/shunit2