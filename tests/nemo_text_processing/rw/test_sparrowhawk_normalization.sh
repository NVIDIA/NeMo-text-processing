#! /bin/sh
GRAMMARS_DIR=${1:-"/workspace/sparrowhawk/documentation/grammars"}
TEST_DIR=${2:-"/workspace/tests/rw"}

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



testTNCardinal() {
  input=$TEST_DIR/data_text_normalization/test_cases_cardinal.txt
  runtest $input
}


testTNTime() {
  input=$TEST_DIR/data_text_normalization/test_cases_time.txt
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





# Remove all command-line arguments
shift $#

# Load shUnit2
. /workspace/shunit2/shunit2
