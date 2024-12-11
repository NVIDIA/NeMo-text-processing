# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.fst_alignment.alignment import get_word_segments


class TestFSTAlignmentUtils:
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_get_word_segments(self):
        text1 = "a hello world    b ccc d   e"
        text2 = " a hello world    b ccc d   e"
        text3 = "a hello world    b ccc d   e "
        text4 = " a hello world    b ccc d   e "
        text5 = "  a hello world    b ccc d   e  "
        words = ["a", "hello", "world", "b", "ccc", "d", "e"]
        for text in [text1, text2, text3, text4, text5]:
            segments = get_word_segments(text)
            assert ' '.join(text[s:e] for s, e in segments) == ' '.join(words)

        empty_text = ""
        segments = get_word_segments(empty_text)
        assert segments == []
