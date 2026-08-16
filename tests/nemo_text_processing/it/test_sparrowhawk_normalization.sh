#! /bin/sh

GRAMMARS_DIR=${1:-"/workspace/sparrowhawk/documentation/grammars"}
PROJECT_DIR=${2:-"/workspace/tests/"}

runtest () {
  input=$1
  echo "INPUT is $input"
  cd ${GRAMMARS_DIR}

  # read test file
  while read testcase; do
    IFS='~' read -a testcase_tokenized <<< $testcase
    written=${testcase_tokenized[0]}
    # only tests against first possible option when there are multiple shortest paths
    spoken=${testcase_tokenized[1]}

    # replace non breaking space with breaking space
    denorm_pred=$(echo $written | normalizer_main --config=sparrowhawk_configuration.ascii_proto 2>&1 | tail -n 1 | sed 's/\xC2\xA0/ /g')
    
    # trim white space
    spoken="$(echo -e "${spoken}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    denorm_pred="$(echo -e "${denorm_pred}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"

    # input expected actual
    assertEquals "$written" "$spoken" "$denorm_pred"
  done < "$input"
}

testTNTime() {
  input=$PROJECT_DIR/it/data_text_normalization/test_cases_time.txt
  runtest $input
}

testTNCardinal() {
  input=$PROJECT_DIR/it/data_text_normalization/test_cases_cardinal.txt
  runtest $input
}

testTNDecimal() {
  input=$PROJECT_DIR/it/data_text_normalization/test_cases_decimal.txt
  runtest $input
}

testTNElectronic() {
  input=$PROJECT_DIR/it/data_text_normalization/test_cases_electronic.txt
  runtest $input
}

testTNWhitelist() {
  input=$PROJECT_DIR/it/data_text_normalization/test_cases_whitelist.txt
  runtest $input
}

testTNMeasure() {
  input=$PROJECT_DIR/it/data_text_normalization/test_cases_measure.txt
  runtest $input
}

testTNMoney() {
  input=$PROJECT_DIR/it/data_text_normalization/test_cases_money.txt
  runtest $input
}

# Load shUnit2
. $PROJECT_DIR/../shunit2/shunit2