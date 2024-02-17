#! /bin/sh

PROJECT_DIR=/workspace/tests

runtest () {
  input=$1
  cd /workspace/sparrowhawk/documentation/grammars

  # read test file
  while read testcase; do
    IFS='~' read spoken written <<< $testcase
    denorm_pred=$(echo $spoken | normalizer_main --config=sparrowhawk_configuration.ascii_proto 2>&1 | tail -n 1)

    # trim white space
    written="$(echo -e "${written}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    denorm_pred="$(echo -e "${denorm_pred}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"

    # input expected actual
    assertEquals "$spoken" "$written" "$denorm_pred"
  done < "$input"
}

testITNCardinal() {
<<<<<<< HEAD
  input=$PROJECT_DIR/fr/data_inverse_text_normalization/test_cases_cardinal.txt
=======
  input=$PROJECT_DIR/zh/data_inverse_text_normalization/test_cases_cardinal.txt
>>>>>>> 42c0071bbeb3141ba013d3965693bb100c06a8e6
  runtest $input
}

testITNDate() {
<<<<<<< HEAD
  input=$PROJECT_DIR/fr/data_inverse_text_normalization/test_cases_date.txt
=======
  input=$PROJECT_DIR/zh/data_inverse_text_normalization/test_cases_date.txt
>>>>>>> 42c0071bbeb3141ba013d3965693bb100c06a8e6
  runtest $input
}

testITNDecimal() {
<<<<<<< HEAD
  input=$PROJECT_DIR/fr/data_inverse_text_normalization/test_cases_decimal.txt
=======
  input=$PROJECT_DIR/zh/data_inverse_text_normalization/test_cases_decimal.txt
>>>>>>> 42c0071bbeb3141ba013d3965693bb100c06a8e6
  runtest $input
}

testITNOrdinal() {
<<<<<<< HEAD
  input=$PROJECT_DIR/fr/data_inverse_text_normalization/test_cases_ordinal.txt
=======
  input=$PROJECT_DIR/zh/data_inverse_text_normalization/test_cases_ordinal.txt
>>>>>>> 42c0071bbeb3141ba013d3965693bb100c06a8e6
  runtest $input
}

testITNFraction() {
<<<<<<< HEAD
  input=$PROJECT_DIR/fr/data_inverse_text_normalization/test_cases_fraction.txt
=======
  input=$PROJECT_DIR/zh/data_inverse_text_normalization/test_cases_fraction.txt
>>>>>>> 42c0071bbeb3141ba013d3965693bb100c06a8e6
  runtest $input
}

testITNTime() {
<<<<<<< HEAD
  input=$PROJECT_DIR/fr/data_inverse_text_normalization/test_cases_time.txt
  runtest $input
}

testITNMeasure() {
  input=$PROJECT_DIR/fr/data_inverse_text_normalization/test_cases_measure.txt
  runtest $input
}

testITNMoney() {
  input=$PROJECT_DIR/fr/data_inverse_text_normalization/test_cases_money.txt
=======
  input=$PROJECT_DIR/zh/data_inverse_text_normalization/test_cases_time.txt
  runtest $input
}

#testITNMeasure() {
#  input=$PROJECT_DIR/fr/data_inverse_text_normalization/test_cases_measure.txt
#  runtest $input
#}

testITNMoney() {
  input=$PROJECT_DIR/zh/data_inverse_text_normalization/test_cases_money.txt
>>>>>>> 42c0071bbeb3141ba013d3965693bb100c06a8e6
  runtest $input
}

testITNWhitelist() {
<<<<<<< HEAD
  input=$PROJECT_DIR/fr/data_inverse_text_normalization/test_cases_whitelist.txt
  runtest $input
}

testITNTelephone() {
  input=$PROJECT_DIR/fr/data_inverse_text_normalization/test_cases_telephone.txt
  runtest $input
}

testITNElectronic() {
  input=$PROJECT_DIR/fr/data_inverse_text_normalization/test_cases_electronic.txt
  runtest $input
}

testITNWord() {
  input=$PROJECT_DIR/fr/data_inverse_text_normalization/test_cases_word.txt
=======
  input=$PROJECT_DIR/zh/data_inverse_text_normalization/test_cases_whitelist.txt
  runtest $input
}

#testITNTelephone() {
#  input=$PROJECT_DIR/zh/data_inverse_text_normalization/test_cases_telephone.txt
#  runtest $input
#}

#testITNElectronic() {
#  input=$PROJECT_DIR/fr/data_inverse_text_normalization/test_cases_electronic.txt
#  runtest $input
#}

testITNWord() {
  input=$PROJECT_DIR/zh/data_inverse_text_normalization/test_cases_word.txt
>>>>>>> 42c0071bbeb3141ba013d3965693bb100c06a8e6
  runtest $input
}

# Load shUnit2
. $PROJECT_DIR/../shunit2/shunit2
