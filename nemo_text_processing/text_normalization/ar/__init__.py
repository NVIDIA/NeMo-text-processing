# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.en.taggers.tokenize_and_classify import ClassifyFst
from nemo_text_processing.text_normalization.en.verbalizers.verbalize import VerbalizeFst
from nemo_text_processing.text_normalization.en.verbalizers.verbalize_final import VerbalizeFinalFst
from nemo_text_processing.utils.logging import logger

try:
    import pynini

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    logger.warning(
        "`pynini` is not installed ! \n"
        "Please run the `nemo_text_processing/setup.sh` script"
        "prior to usage of this toolkit."
    )

    PYNINI_AVAILABLE = False
