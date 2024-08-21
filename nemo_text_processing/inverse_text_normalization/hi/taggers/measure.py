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
from nemo_text_processing.inverse_text_normalization.hi.graph_utils import (
    GraphFst,
    insert_space,
    delete_space,
    convert_space,
    delete_extra_space,
)
from pynini.lib import pynutil


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure
        e.g. ऋण बारह किलोग्राम -> measure { negative: "true" cardinal { integer: "१२" } units: "kg" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        input_case: accepting either "lower_cased" or "cased" input.
    """

