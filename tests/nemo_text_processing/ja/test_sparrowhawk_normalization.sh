#! /bin/sh

GRAMMARS_DIR=${1:-"/workspace/sparrowhawk/documentation/grammars"}
PROJECT_DIR=${2:-"/workspace/tests"}

runtest () {
  input=$1
  echo "INPUT is $input"
  cd ${GRAMMARS_DIR}

  # read test file
  while read testcase; do
    IFS='~' read written spoken <<< $testcase
    # replace non breaking space with breaking space
    denorm_pred=$(echo $written | normalizer_main --config=sparrowhawk_configuration_pp.ascii_proto 2>&1 | tail -n 1 | sed 's/\xC2\xA0/ /g')

    # # trim white space
    spoken="$(echo -e "${spoken}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    denorm_pred="$(echo -e "${denorm_pred}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"

    # input expected actual
    assertEquals "$written" "$spoken" "$denorm_pred"
  done < "$input"
}
testTNFractionText() {
  input=$PROJECT_DIR/ja/data_text_normalization/test_cases_fraction.txt
  runtest $input
}
testTNTimeText() {
  input=$PROJECT_DIR/ja/data_text_normalization/test_cases_time.txt
  runtest $input
}
testTNCardinalText() {
  input=$PROJECT_DIR/ja/data_text_normalization/test_cases_cardinal.txt
  runtest $input
}
testTNOrdinalText() {
  input=$PROJECT_DIR/ja/data_text_normalization/test_cases_ordinal.txt
  runtest $input
}
testTNDecimalalText() {
 input=$PROJECT_DIR/ja/data_text_normalization/test_cases_decimal.txt
  runtest $input
}
testTNDateText() {
  input=$PROJECT_DIR/ja/data_text_normalization/test_cases_date.txt
  runtest $input
}


# Load shUnit2
#. $PROJECT_DIR/../shunit2/shunit2
. /workspace/shunit2/shunit2
