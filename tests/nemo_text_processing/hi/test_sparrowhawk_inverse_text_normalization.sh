#! /bin/sh

PROJECT_DIR=/workspace/tests

runtest () {
  input=$1
  cd /workspace/sparrowhawk/documentation/grammars

  # per-case timeout (seconds): bounds any input that causes a lattice blow-up
  # so the suite can never hang forever. Override with CASE_TIMEOUT env var.
  : ${CASE_TIMEOUT:=20}
  total=$(wc -l < "$input")
  n=0

  # read test file
  while read testcase; do
    n=$((n+1))
    IFS='~' read spoken written <<< $testcase

    # run with a timeout; if it times out, mark the prediction so it fails loudly
    denorm_pred=$(echo $spoken | timeout ${CASE_TIMEOUT} normalizer_main --config=sparrowhawk_configuration.ascii_proto 2>&1 | tail -n 1)
    if [ $? -eq 124 ]; then
      denorm_pred="<<TIMEOUT after ${CASE_TIMEOUT}s>>"
    fi

    # trim white space
    written="$(echo -e "${written}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    denorm_pred="$(echo -e "${denorm_pred}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"

    # progress to stderr so you can see it moving (not part of test output)
    echo "[$n/$total] '$spoken' -> '$denorm_pred'" >&2

    # input expected actual
    assertEquals "$spoken" "$written" "$denorm_pred"
  done < "$input"
}

testITNCardinal() {
  input=$PROJECT_DIR/hi/data_inverse_text_normalization/test_cases_cardinal.txt
  runtest $input
}


testITNDecimal() {
  input=$PROJECT_DIR/hi/data_inverse_text_normalization/test_cases_decimal.txt
  runtest $input
}

testITNOrdinal() {
  input=$PROJECT_DIR/hi/data_inverse_text_normalization/test_cases_ordinal.txt
  runtest $input
}


testITNFraction() {
  input=$PROJECT_DIR/hi/data_inverse_text_normalization/test_cases_fraction.txt
  runtest $input
}


testITNDate() {
  input=$PROJECT_DIR/hi/data_inverse_text_normalization/test_cases_date.txt
  runtest $input
}

testITNTime() {
  input=$PROJECT_DIR/hi/data_inverse_text_normalization/test_cases_time.txt
  runtest $input
}

testITNMeasure() {
  input=$PROJECT_DIR/hi/data_inverse_text_normalization/test_cases_measure.txt
  runtest $input
}

testITNAddress() {
  input=$PROJECT_DIR/hi/data_inverse_text_normalization/test_cases_address.txt
  runtest $input
}

testITNMoney() {
  input=$PROJECT_DIR/hi/data_inverse_text_normalization/test_cases_money.txt
  runtest $input
}

testITNTelephone() {
  input=$PROJECT_DIR/hi/data_inverse_text_normalization/test_cases_telephone.txt
  runtest $input
}

testITNWord() {
  input=$PROJECT_DIR/hi/data_inverse_text_normalization/test_cases_word.txt
  runtest $input
}

testITNWhiteList() {
  input=$PROJECT_DIR/hi/data_inverse_text_normalization/test_cases_whitelist.txt
  runtest $input
}

testITNElectronic() {
  input=$PROJECT_DIR/hi/data_inverse_text_normalization/test_cases_electronic.txt
  runtest $input
}


# Load shUnit2
. $PROJECT_DIR/../shunit2/shunit2
