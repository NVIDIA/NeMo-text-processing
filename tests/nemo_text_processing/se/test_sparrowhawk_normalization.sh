#! /bin/bash

GRAMMARS_DIR=${1:-"/workspace/sparrowhawk/documentation/grammars"}
PROJECT_DIR=${2:-"/workspace/tests"}

runtest () {
  input=$1
  echo "INPUT is $input"
  cd "${GRAMMARS_DIR}" || return 1

  while read -r testcase; do
    IFS='~' read -r written spoken <<< "$testcase"
    norm_pred=$(echo "$written" | normalizer_main --config=sparrowhawk_configuration.ascii_proto 2>&1 | tail -n 1)

    spoken="$(echo -e "${spoken}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    norm_pred="$(echo -e "${norm_pred}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"

    assertEquals "$written" "$spoken" "$norm_pred"
  done < "$input"
}

testTNAbbreviation() {
  runtest "$PROJECT_DIR/se/data_text_normalization/test_cases_abbreviation.txt"
}

testTNCardinal() {
  runtest "$PROJECT_DIR/se/data_text_normalization/test_cases_cardinal.txt"
}

testTNDate() {
  runtest "$PROJECT_DIR/se/data_text_normalization/test_cases_date.txt"
}

testTNElectronic() {
  runtest "$PROJECT_DIR/se/data_text_normalization/test_cases_electronic.txt"
}

testTNMeasure() {
  runtest "$PROJECT_DIR/se/data_text_normalization/test_cases_measure.txt"
}

testTNMoney() {
  runtest "$PROJECT_DIR/se/data_text_normalization/test_cases_money.txt"
}

testTNOrdinal() {
  runtest "$PROJECT_DIR/se/data_text_normalization/test_cases_ordinal.txt"
}

testTNTime() {
  runtest "$PROJECT_DIR/se/data_text_normalization/test_cases_time.txt"
}

testTNWhitelist() {
  runtest "$PROJECT_DIR/se/data_text_normalization/test_cases_whitelist.txt"
}

testTNWord() {
  runtest "$PROJECT_DIR/se/data_text_normalization/test_cases_word.txt"
}

. "$PROJECT_DIR/shunit2/shunit2"
