# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright (c) 2023, Jim O'Regan.
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

    Returns dictionary of mappings
    """
    label_tsv = open(abs_path)
    labels = list(csv.reader(label_tsv, delimiter="\t"))
    return labels


def load_inflection(abs_path):
    """
    loads inflection information

    Args:
        abs_path: absolute path
    
    Returns dictionary of mappings of word endings to
    lists of case endings.
    """
    with open(abs_path) as inflection_tsv:
        items = list(csv.reader(inflection_tsv, delimiter="\t"))
        inflections = {k[0]: k[1].split(" ") for k in items}
        return inflections


def naive_inflector(abbr: str, word: str, singular_only=False):
    """
    Performs naïve inflection of a pair of words: the abbreviation,
    and its expansion. Possessive forms are omitted, due to the
    nature of the kinds of words/abbreviations being expanded

    Args:
        abbr: the abbreviated base form
        word: the base (nominative singular) form of the expansion
              of abbr
        singular_only: whether or not to add plural forms
    
    Returns a list of tuples containing the inflected abbreviation and
    its expansion.
    """
    singular = load_inflection(get_abs_path("data/inflection/endings.tsv"))
    plural = load_inflection(get_abs_path("data/inflection/plural_endings.tsv"))
    lexical = load_inflection(get_abs_path("data/inflection/word_endings.tsv"))
    keys_sorted = sorted(singular, key=lambda k: len(k), reverse=True)

    def get_kv():
        if word in lexical:
            return (word, lexical[word])
        for key in keys_sorted:
            if word.endswith(key):
                return (key, singular[key])
        return None

    forms = []
    key, ends = get_kv()
    outword = word
    for wordend in ["ny", "ly", "év", "a", "e"]:
        if outword.endswith(wordend):
            outword = outword[: -len(wordend)]

    def tweak(form: str) -> str:
        if outword == word:
            return form
        endings = ["ny", "nny", "ly", "lly", "ev", "év", "a", "á", "e", "é"]
        undouble = {
            "nny": "ny",
            "lly": "ly",
        }
        for ending in endings:
            if form.startswith(ending):
                final = ""
                if ending in undouble:
                    final = undouble[ending]
                return final + form[len(ending) :]

    for form in ends:
        forms.append((f"{abbr}-{tweak(form)}", f"{outword}{form}"))
    if not singular_only:
        for plural_form in plural[key]:
            forms.append((f"{abbr}-{tweak(plural_form)}", f"{outword}{plural_form}"))
            for form in singular[plural_form]:
                forms.append((f"{abbr}-{tweak(plural_form)}{form}", f"{outword}{plural_form}{form}"))
    return forms
