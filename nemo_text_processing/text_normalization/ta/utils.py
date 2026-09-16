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

import os

import pynini

from nemo_text_processing.text_normalization.en.utils import load_labels


def get_abs_path(rel_path):
    """
    Get absolute path

    Args:
        rel_path: relative path to this file

    Returns absolute path
    """
    return os.path.dirname(os.path.abspath(__file__)) + '/' + rel_path


def table_fst(abs_path: str, key: int = 0, value: int = 1) -> 'pynini.FstLike':
    """
    Compiles two columns of a TSV table into an optimized string map.

    Unlike ``pynini.string_file`` this tolerates a third column that is not a weight, so it is
    the loader for the tables that carry a kind or a note in their last column.

    Args:
        abs_path: absolute path of the table
        key: index of the input column
        value: index of the output column
    """
    width = max(key, value) + 1
    rows = [row for row in load_labels(abs_path) if len(row) >= width]
    return pynini.string_map([(row[key], row[value]) for row in rows]).optimize()
