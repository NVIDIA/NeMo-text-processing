#!/bin/bash

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

# this script runs Sparrowhawk tests in a docker container "locally" (not in CI/CD pipeline)

MODE=${1:-"interactive"}
LANGUAGE=${2:-"en"}
INPUT_CASE=${3:-"lower_cased"}
GRAMMARS=${4:-"tn_grammars"} # tn_grammars or itn_grammars
SCRIPT_DIR=$(cd $(dirname $0); pwd)
GRAMMAR_DIR=${5:-${SCRIPT_DIR}"/.."}
CONFIG=${LANGUAGE}_${GRAMMARS}_${INPUT_CASE}

: ${CLASSIFY_DIR:="$GRAMMAR_DIR/${CONFIG}/classify"}
: ${VERBALIZE_DIR:="$GRAMMAR_DIR/${CONFIG}/verbalize"}
: ${CMD:=${6:-"/bin/bash"}}

MOUNTS=""
MOUNTS+=" -v $CLASSIFY_DIR:/workspace/sparrowhawk/documentation/grammars/en_toy/classify"
MOUNTS+=" -v $VERBALIZE_DIR:/workspace/sparrowhawk/documentation/grammars/en_toy/verbalize"

WORK_DIR="/workspace/sparrowhawk/documentation/grammars"

# update test case script based on input case (for ITN English)
if [[ $INPUT_CASE == "lower_cased" ]]; then
  INPUT_CASE=".sh"
else
  INPUT_CASE="_cased.sh"
fi

if [[ $MODE == "test_tn_grammars" ]]; then
  CMD="bash test_sparrowhawk_normalization.sh"
  WORK_DIR="/workspace/tests/${LANGUAGE}"
elif [[ $MODE == "test_itn_grammars" ]]; then
  CMD="bash test_sparrowhawk_inverse_text_normalization${INPUT_CASE}"
  WORK_DIR="/workspace/tests/${LANGUAGE}"
fi

echo $MOUNTS
docker run -it -e LANG=C.UTF-8 -e LC_ALL=C.UTF-8 --rm \
  --shm-size=4g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  $MOUNTS \
  -v $SCRIPT_DIR/../../../tests/nemo_text_processing/:/workspace/tests/ \
  -w $WORK_DIR \
  sparrowhawk:latest $CMD