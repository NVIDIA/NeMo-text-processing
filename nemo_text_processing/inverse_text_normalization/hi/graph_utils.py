# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2024 and onwards Google, Inc.
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


import pynini

from nemo_text_processing.inverse_text_normalization.hi.utils import get_abs_path


graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))

NEMO_HI_DIGIT = pynini.union("०", "१", "२", "३", "४", "५", "६", "७", "८", "९").optimize()

NEMO_NON_BREAKING_SPACE = u"\u00A0"


MINUS = pynini.union("ऋणात्मक", "नकारात्मक").optimize()

