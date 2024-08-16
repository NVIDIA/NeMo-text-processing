#!/bin/bash

# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

# This script runs the CI/CD tests for Sparrowhawk. It calls export_grammars.sh to create the grammars.


GRAMMARS="itn_grammars" # tn_grammars
INPUT_CASE="lower_cased" # cased
LANGUAGE="en" # language, {'en', 'es', 'de','zh'} supports both TN and ITN, {'pt', 'ru', 'fr', 'vi'} supports ITN only
OVERWRITE_CACHE="False" # Set to False to re-use .far files
WHITELIST="" # Path to a whitelist file, if None the default will be used
FAR_PATH=$(pwd) # Path where the grammars should be written
MODE="test_itn_grammars"

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

if [[ ${WHITELIST} != "" ]] && [[ -f $WHITELIST ]]; then
  WHITELIST="--whitelist=${WHITELIST} "
  echo "[I] Whitelist file wasn't provided or doesn't exist, using default"
else
  WHITELIST=""
fi

bash export_grammars.sh --MODE="export" --GRAMMARS=$GRAMMARS --LANGUAGE=$LANGUAGE --INPUT_CASE=$INPUT_CASE \
      --FAR_PATH=$FAR_PATH  --CACHE_DIR=$CACHE_DIR --OVERWRITE_CACHE=$OVERWRITE_CACHE \
      --FORCE_REBUILD=$FORCE_REBUILD $WHITELIST

CLASSIFY_FAR=${CACHE_DIR}_${GRAMMARS}_${INPUT_CASE}/classify/tokenize_and_classify.far
VERBALIZE_FAR=${CACHE_DIR}_${GRAMMARS}_${INPUT_CASE}/verbalize/verbalize.far

CONFIG=${LANGUAGE}_${GRAMMARS}_${INPUT_CASE}

cp $CLASSIFY_FAR /workspace/sparrowhawk/documentation/grammars_${CONFIG}/en_toy/classify/
cp $VERBALIZE_FAR /workspace/sparrowhawk/documentation/grammars_${CONFIG}/en_toy/verbalize/