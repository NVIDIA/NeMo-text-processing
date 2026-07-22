#! /bin/sh

PROJECT_DIR=/workspace/tests

runtest () {
  input=$1
  cd /workspace/sparrowhawk/documentation/grammars

  # read test file
  while IFS= read -r testcase; do
    IFS='~' read -r written spoken <<< "$testcase"
    
    # Escape backslashes and replace non breaking space with breaking space
    escaped_written=$(printf '%s' "$written" | sed 's/\\/\\\\/g')
    denorm_pred=$(echo "$escaped_written" | normalizer_main --config=sparrowhawk_configuration.ascii_proto 2>&1 | tail -n 1 | sed 's/\xC2\xA0/ /g')

    # trim white space
    # spoken="$(echo -e "${spoken}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    # denorm_pred="$(echo -e "${denorm_pred}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"

    # trim white space and remove space before punctuation
    spoken="$(echo -e "${spoken}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/ \([!?.]\)/\1/g')"
    denorm_pred="$(echo -e "${denorm_pred}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/ \([!?.]\)/\1/g')"

    # input expected actual
    assertEquals "$written" "$spoken" "$denorm_pred"
  done < "$input"
}

#testTNSpecialText() {
#  input=$PROJECT_DIR/kn/data_text_normalization/test_cases_special_text.txt
#  runtest $input
#}

testTNCardinal() {
  input=$PROJECT_DIR/kn/data_text_normalization/test_cases_cardinal.txt
  runtest $input
}

# Load shUnit2
. $PROJECT_DIR/../shunit2/shunit2