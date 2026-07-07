# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

# HTTP server that serves NeMo TN/ITN grammars through Sparrowhawk.
#
# Build FROM the Sparrowhawk backend image produced by
# tools/text_processing_deployment/docker/build.sh (tag: sparrowhawk:latest),
# which already provides the `normalizer_main` binary and the FST runtime.
#
# The build context is tools/text_processing_deployment/ so the exported grammar
# dirs (el_*_grammars_cased) are visible to COPY. Build with:
#   docker build -f server/Dockerfile -t nemo-sparrowhawk-server \
#       --build-arg GRAMMARS_TN_SRC=el_tn_grammars_cased \
#       --build-arg GRAMMARS_ITN_SRC=el_itn_grammars_cased .
ARG BASE_IMAGE=sparrowhawk:latest
FROM ${BASE_IMAGE}

# Grammar source dirs, relative to the build context. Point these at whatever
# language you exported (default: Greek, cased).
ARG GRAMMARS_TN_SRC=el_tn_grammars_cased
ARG GRAMMARS_ITN_SRC=el_itn_grammars_cased

WORKDIR /app

# The base is continuumio/miniconda3, so `pip` is already on PATH.
COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Server code + Sparrowhawk ascii_proto config templates.
COPY server/app.py server/normalizer.py /app/
COPY server/configs /app/configs
COPY server/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Bake the compiled .far grammars into the image. Override at runtime by bind
# mounting over /app/grammars/{tn,itn} or setting GRAMMARS_TN / GRAMMARS_ITN.
COPY ${GRAMMARS_TN_SRC} /app/grammars/tn
COPY ${GRAMMARS_ITN_SRC} /app/grammars/itn

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    LANGUAGE=el \
    PORT=8000 \
    WORKERS_PER_DIRECTION=2 \
    STDBUF_PREFIX="stdbuf -oL -eL"

EXPOSE 8000
ENTRYPOINT ["/app/entrypoint.sh"]
