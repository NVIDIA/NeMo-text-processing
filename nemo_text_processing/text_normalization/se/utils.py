# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright (c) 2023, Jim O'Regan for Språkbanken Tal
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
import logging
import os

CASE_KEYS = ["ess", "com_pl", "com_sg", "gen_sg", "gen_pl", "ill_pl", "ill_sg", "loc_sg", "nom_pl"]
CASE_KEYS_EXT = CASE_KEYS + ["nom_sg"]


def get_abs_path(rel_path):
    """
    Get absolute path

    Args:
        rel_path: relative path to this file
        
    Returns absolute path
    """
    abs_path = os.path.dirname(os.path.abspath(__file__)) + os.sep + rel_path

    if not os.path.exists(abs_path):
        logging.warning(f'{abs_path} does not exist')
    return abs_path


def load_labels(abs_path):
    """
    loads relative path file as list of lists

    Args:
        abs_path: absolute path

    Returns list of mappings
    """
    with open(abs_path, encoding="utf-8") as label_tsv:
        labels = list(csv.reader(label_tsv, delimiter="\t"))
        return labels


def load_case_forms(abs_path, extended=False):
    """
    loads relative path file as dictionary, keyed on case/number

    Args:
        abs_path: absolute path of file

    Returns dictionary of case forms
    """
    KEYS = CASE_KEYS
    if extended:
        KEYS = CASE_KEYS_EXT
    with open(abs_path, encoding="utf-8") as label_tsv:
        labels = list(csv.reader(label_tsv, delimiter="\t"))
        ret = {}
        for label in labels:
            if label and (label[0] in KEYS):
                ret[label[0]] = label[1]
        return ret