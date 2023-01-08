# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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


MAJOR = 0
MINOR = 1
PATCH = 4
PRE_RELEASE = 'rc0'

# Use the following formatting: (major, minor, patch, pre-release)
VERSION = (MAJOR, MINOR, PATCH, PRE_RELEASE)

__shortversion__ = '.'.join(map(str, VERSION[:3]))
__version__ = '.'.join(map(str, VERSION[:3])) + ''.join(VERSION[3:])

__package_name__ = 'nemo_text_processing'
__contact_names__ = 'NVIDIA'
__contact_emails__ = 'nemo-toolkit@nvidia.com'
__homepage__ = 'https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/'
__repository_url__ = 'https://github.com/nvidia/nemo-text-processing'
__download_url__ = 'https://github.com/NVIDIA/NeMo-text-processing/releases'
__description__ = 'NeMo text processing for ASR and TTS'
__license__ = 'Apache2'
__keywords__ = ' NeMo, nvidia, tts, asr, text processing, text normalization, inverse text normalization, language'
