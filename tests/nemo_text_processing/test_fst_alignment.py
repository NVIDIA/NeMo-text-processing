# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

pytest.importorskip('pynini')

from nemo_text_processing.fst_alignment.alignment import EPS, indexed_map_to_output  # noqa: E402


def test_leading_insertion_stays_in_the_first_span():
    alignment = [
        (EPS, 't'),
        ('2', 'w'),
    ]
    start, end = indexed_map_to_output(alignment, start=0, end=1, mode='itn')
    assert (start, end) == (0, 2)


def test_a_later_span_does_not_take_the_leading_insertion():
    alignment = [
        (EPS, 'X'),
        ('a', 'a'),
        ('b', 'b'),
    ]
    start, end = indexed_map_to_output(alignment, start=1, end=2, mode='itn')
    assert (start, end) == (2, 3)
