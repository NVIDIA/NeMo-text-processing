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

# Dockerfile for C++ (inverse) text normalization backend Sparrowhawk https://github.com/google/sparrowhawk

# set base image (host OS)
FROM continuumio/miniconda3 


# set the working directory in the container
WORKDIR /workspace

# install dependencies
RUN echo "deb http://archive.debian.org/debian stretch main contrib non-free" > /etc/apt/sources.list
RUN apt-get update &&  apt-get upgrade -y &&   apt-get install -y --reinstall build-essential pkg-config  git make  wget
RUN conda install conda-build -y
RUN conda install -c conda-forge thrax=1.3.4 -y
RUN git clone https://github.com/google/re2 
RUN cd re2 && git checkout tags/2022-02-01 && make && make install
RUN wget https://github.com/protocolbuffers/protobuf/releases/download/v2.5.0/protobuf-2.5.0.tar.gz
RUN tar xzvf protobuf-2.5.0.tar.gz
RUN cd protobuf-2.5.0 && ./configure && make && make install && ldconfig
RUN printf  "# Conda lib path \n/opt/conda/lib" > /etc/ld.so.conf.d/conda.so.conf
ENV CPPFLAGS="-I/opt/conda/include"
ENV LDFLAGS="-L/opt/conda/lib"
RUN git clone https://github.com/anand-nv/sparrowhawk.git && cd sparrowhawk &&  git checkout nemo_tests &&   apt-get install -y autoconf &&     bash autoreconf && ./configure && make && make install && ldconfig
RUN git clone https://github.com/kward/shunit2.git
RUN echo "DONE"
