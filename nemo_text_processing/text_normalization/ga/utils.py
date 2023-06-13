# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright (c) 2022, Jim O'Regan
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

import csv
import os


def get_abs_path(rel_path):
    """
    Get absolute path

    Args:
        rel_path: relative path to this file

    Returns absolute path
    """
    return os.path.dirname(os.path.abspath(__file__)) + '/' + rel_path


def load_labels(abs_path):
    """
    loads relative path file as dictionary

    Args:
        abs_path: absolute path

    Returns list of mappings
    """
    label_tsv = open(abs_path)
    labels = list(csv.reader(label_tsv, delimiter="\t"))
    return labels


def load_labels_dict(abs_path):
    """
    loads relative path file as dictionary

    Args:
        abs_path: absolute path

    Returns dictionary of mappings
    """
    return {x[0]: x[1] for x in load_labels(abs_path)}


def extend_list_with_mutations(input):
    out = []

    UPPER_VOWELS = "AEIOUÁÉÍÓÚ"
    LOWER_VOWELS = "aeiouáéíóú"

    for word in input:
        out.append(word)
        if word[0] in UPPER_VOWELS:
            out.append(f"h{word}")
            out.append(f"n{word}")
            out.append(f"t{word}")
        elif word[0] in LOWER_VOWELS:
            out.append(f"h{word}")
            out.append(f"n-{word}")
            out.append(f"t-{word}")

    return out
