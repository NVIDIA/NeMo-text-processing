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

import pytest

from nemo_text_processing.text_normalization.utils_audio_based import get_alignment


class TestAudioBasedTNUtils:
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_default(self):
        raw = 'This is #4 ranking on G.S.K.T.'
        pred_text = 'this iss for ranking on g k p'
        norm = 'This is nubmer four ranking on GSKT'

        output = get_alignment(raw, norm, pred_text, True)
        reference = (
            ['is #4', 'G.S.K.T.'],
            ['iss for', 'g k p'],
            ['is nubmer four', 'GSKT'],
            ['This', '[SEMIOTIC_SPAN]', 'ranking', 'on', '[SEMIOTIC_SPAN]'],
            [1, 4],
        )
        assert output == reference
