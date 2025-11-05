# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

####################
# HEBREW CONSTANTS #
####################
units_feminine_dict = {
    "0": "אפס",
    "1": "אחת",
    "2": "שתיים",
    "3": "שלוש",
    "4": "ארבע",
    "5": "חמש",
    "6": "שש",
    "7": "שבע",
    "8": "שמונה",
    "9": "תשע",
}

units_masculine_dict = {
    "0": "אפס",
    "1": "אחד",
    "2": "שניים",
    "3": "שלושה",
    "4": "ארבעה",
    "5": "חמישה",
    "6": "שישה",
    "7": "שבעה",
    "8": "שמונה",
    "9": "תשעה",
}

tens_dict = {
    "2": "עשרים",
    "3": "שלושים",
    "4": "ארבעים",
    "5": "חמישים",
    "6": "שישים",
    "7": "שבעים",
    "8": "שמונים",
    "9": "תשעים",
}

ten = {
    "short": "עשר",
    "long": "עשרה",
}  # double pronunciation: short is 'eser' and 'asar', long is 'esre' and 'asara'


#############
# FUNCTIONS #
#############
def get_abs_path(rel_path):
    """
    Get absolute path

    Args:
        rel_path: relative path to this file

    Returns absolute path
    """
    return os.path.dirname(os.path.abspath(__file__)) + "/" + rel_path


def augment_labels_with_punct_at_end(labels):
    """
    augments labels: if key ends on a punctuation that value does not have, add a new label
    where the value maintains the punctuation

    Args:
        labels : input labels
    Returns:
        additional labels
    """
    res = []
    for label in labels:
        if len(label) > 1:
            if label[0][-1] == "." and label[1][-1] != ".":
                res.append([label[0], label[1] + "."] + label[2:])
    return res


def digit_by_digit(num):

    dbd = [" ".join([units_feminine_dict[digit] for digit in num])]

    # generate "1" as masculine and as feminine if exists
    if units_feminine_dict["1"] in dbd[0]:
        dbd.append(dbd[0].replace(units_feminine_dict["1"], units_masculine_dict["1"]))

    return dbd


def integer_to_text(num, only_fem=False):
    if isinstance(num, int):
        num = str(num)
    # number is zero
    if num == len(num) * "0":
        return ["אפס"]
    else:
        # remove leading zeros from number
        num = num.lstrip("0")

        # units
        if len(num) == 1:
            return _less_than_10(num, only_fem)

        # tenths
        elif len(num) == 2:
            return _less_than_100(num, only_fem)

        else:
            raise Exception


def _less_than_10(num, only_fem=False):
    """
    Returns a list of all the possible names of a number in range 0-9
    """

    if only_fem:
        return [units_feminine_dict[num]]
    else:
        return [units_feminine_dict[num], units_masculine_dict[num]]


def _less_than_100(num, only_fem=False):
    """
    Returns a list of all the possible names of a number in range 0-99
    """

    # init result
    res = list()

    # split number to digits
    tens, units = num

    # number is in range 0-9
    if len(num) == 1:
        res.extend(_less_than_10(num))

    # number is in range 10-99
    elif len(num) == 2:

        if num == "10":
            if only_fem:
                res.extend([ten["short"]])
            else:
                res.extend([ten["long"], ten["short"]])

        # number is in range 11-19
        elif tens == "1":
            res.append(f'{units_feminine_dict[num[1]]} {ten["long"]}')
            if not only_fem:
                res.append(f'{units_masculine_dict[num[1]]} {ten["short"]}')

        else:

            # number is in range 20-99, a multiplication of 10
            if units == "0":
                res.append(tens_dict[num[0]])

            # number is in range 20-99, but not multiplication of 10
            else:
                res.append(f'{tens_dict[num[0]]} {"ו"}{units_feminine_dict[num[1]]}')
                if not only_fem:
                    res.append(f'{tens_dict[num[0]]} {"ו"}{units_masculine_dict[num[1]]}')

    return res
