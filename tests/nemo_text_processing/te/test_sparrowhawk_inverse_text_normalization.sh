#! /bin/sh

PROJECT_DIR=/workspace/tests

runtest () {
  input=$1
  cd /workspace/sparrowhawk/documentation/grammars

  while read testcase; do
    IFS='~' read spoken written <<< $testcase
    denorm_pred=$(echo $spoken | normalizer_main --config=sparrowhawk_configuration.ascii_proto 2>&1 | tail -n 1)

    written="$(echo -e "${written}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    denorm_pred="$(echo -e "${denorm_pred}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"

    assertEquals "$spoken" "$written" "$denorm_pred"
  done < "$input"
}

testITNCardinal() {
  input=$PROJECT_DIR/te/data_inverse_text_normalization/test_cases_cardinal.txt
  runtest $input
}

. $PROJECT_DIR/../shunit2/shunit2
