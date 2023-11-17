#!/bin/bash

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script compiles and exports WFST-grammars from nemo_text_processing, builds C++ production backend Sparrowhawk (https://github.com/google/sparrowhawk) in docker,
# plugs grammars into Sparrowhawk and returns prompt inside docker.
# For inverse text normalization run:
#       bash export_grammars.sh --GRAMMARS=itn_grammars --LANGUAGE=en
#       echo "two dollars fifty" | ../../src/bin/normalizer_main --config=sparrowhawk_configuration.ascii_proto
# For text normalization run:
#       bash export_grammars.sh --GRAMMARS=tn_grammars --LANGUAGE=en
#       echo "\$2.5" | ../../src/bin/normalizer_main --config=sparrowhawk_configuration.ascii_proto
#
# To test TN grammars, run:
#       bash export_grammars.sh --GRAMMARS=tn_grammars --LANGUAGE=en --MODE=test
#
# To test ITN grammars, run:
#       bash export_grammars.sh --GRAMMARS=itn_grammars --LANGUAGE=en --MODE=test

GRAMMARS="itn_grammars" # tn_grammars
INPUT_CASE="lower_cased" # cased
LANGUAGE="en" # language, {'en', 'es', 'de','zh'} supports both TN and ITN, {'pt', 'ru', 'fr', 'vi'} supports ITN only
MODE="export"
OVERWRITE_CACHE="False" # Set to False to re-use .far files
WHITELIST=None # Path to a whitelist file, if None the default will be used
FAR_PATH=$(pwd) # Path where the grammars should be written

for ARG in "$@"
do
    key=$(echo $ARG | cut -f1 -d=)
    value=$(echo $ARG | cut -f2 -d=)

    if [[ $key == *"--"* ]]; then
        v="${key/--/}"
        declare $v="${value}"
    fi
done


CACHE_DIR=${FAR_PATH}/${LANGUAGE}
echo "GRAMMARS = $GRAMMARS"
echo "LANGUAGE = $LANGUAGE"
echo "INPUT_CASE = $INPUT_CASE"
echo "CACHE_DIR = $CACHE_DIR"
echo "OVERWRITE_CACHE = $OVERWRITE_CACHE"
echo "FORCE_REBUILD = $FORCE_REBUILD"
echo "WHITELIST = $WHITELIST"

bash export_grammars.sh --MODE="export" --GRAMMARS=$GRAMMARS --LANGUAGE=$LANGUAGE --INPUT_CASE=$INPUT_CASE \
      --FAR_PATH=$FAR_PATH  --CACHE_DIR=$CACHE_DIR --OVERWRITE_CACHE=$OVERWRITE_CACHE --FORCE_REBUILD=$FORCE_REBUILD \
      --WHITELIST=$WHITELIST

CLASSIFY_FAR=${CACHE_DIR}"/classify/tokenize_and_classify.far"
VERBALIZE_FAR=${CACHE_DIR}"/verbalize/verbalize.far"

cp $CLASSIFY_FAR /workspace/sparrowhawk/documentation/grammars/en_toy/classify/
cp $VERBALIZE_FAR /workspace/sparrowhawk/documentation/grammars/en_toy/verbalize/
WORK_DIR="tests/${LANGUAGE}"

if [[ $MODE == "test_tn_grammars" ]]; then
  CMD="cd ${WORK_DIR} && bash test_sparrowhawk_normalization.sh"
elif [[ $MODE == "test_itn_grammars" ]]; then
  CMD="cd ${WORK_DIR} && bash test_sparrowhawk_inverse_text_normalization${INPUT_CASE}"
fi
