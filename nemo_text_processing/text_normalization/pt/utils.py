# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Portuguese (PT) text normalization utilities.

Provides get_abs_path for resolving data paths and load_labels for reading TSV label files.
"""
import csv
import os


def get_abs_path(rel_path: str) -> str:
    """
    Resolve a path relative to this module to an absolute path.

    Args:
        rel_path: path relative to the PT text normalization data directory.

    Returns:
        Absolute path string.
    """
    return os.path.dirname(os.path.abspath(__file__)) + '/' + rel_path


def load_labels(abs_path: str):
    """
    Load a TSV file as a list of rows (list of lists).

    Args:
        abs_path: absolute path to a UTF-8 TSV file.

    Returns:
        List of rows, each row a list of fields (e.g. from csv.reader).
    """
    with open(abs_path, encoding="utf-8") as label_tsv:
        labels = list(csv.reader(label_tsv, delimiter="\t"))
    return labels
