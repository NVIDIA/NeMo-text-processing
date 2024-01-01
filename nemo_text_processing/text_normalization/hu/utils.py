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
    with open(abs_path) as label_tsv:
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


def _modify_ending(outword: str, word: str, form: str) -> str:
    """
    Helper for the inflector. Modifies endings where there is a difference
    between how they are written for abbreviations, and for full words.

    Args:
        outword: the form of the word to be output
        word: the base form of the word
        form: the ending to be appended
    """
    if outword == word:
        return form
    endings = ["ny", "nny", "ly", "lly", "ev", "év", "út", "ut", "a", "á", "e", "é"]
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
    return form


def inflect_abbreviation(abbr: str, word: str, singular_only=False):
    """
    For currency symbols, the inflection can either be taken from
    the underlying final word, or from the letter itself.
    This (ab)uses naive_inflector to get the letter-based
    inflection.

    Args:
        abbr: the abbreviated base form
        word: the base (nominative singular) form of the expansion
              of abbr
        singular_only: whether or not to add plural forms

    Returns a list of tuples containing the inflected abbreviation and
    its expansion.
    """
    abbr_orig = abbr
    abbr = abbr.lower()
    if abbr[-1] in "bcdgjptvz":
        ending = "é"
    elif abbr[-1] in "aáeéiíoóöőuúüű":
        ending = abbr[-1]
    elif abbr[-1] in "flmnrs":
        ending = "e" + abbr[-1]
    elif abbr[-1] in "hk":
        ending = "á"
    else:
        return []

    word_part = naive_inflector(".", word, singular_only)
    abbr_part = naive_inflector(abbr_orig, ending, singular_only)

    word_useful = [x[1] for x in word_part]
    abbr_useful = [x[0] for x in abbr_part]
    return zip(abbr_useful, word_useful)


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
    keys_sorted = sorted(singular, key=len, reverse=True)

    def get_kv():
        if word in lexical:
            return (word, lexical[word])
        for key in keys_sorted:
            if word.endswith(key):
                return (key, singular[key])
        raise KeyError(f"Key {key} not found ({word})")

    forms = []
    key, ends = get_kv()
    outword = word
    for wordend in ["ny", "ly", "év", "út", "a", "e"]:
        if outword.endswith(wordend):
            outword = outword[: -len(wordend)]

    def tweak(form: str) -> str:
        return _modify_ending(outword, word, form)

    if "-" in abbr:
        abbr = abbr.split("-")[0]
    for form in ends:
        forms.append((f"{abbr}-{tweak(form)}", f"{outword}{form}"))
    if not singular_only:
        for plural_form in plural[key]:
            plural_key = plural_form
            if plural_form == "k":
                plural_key = key + "k"
            forms.append((f"{abbr}-{tweak(plural_form)}", f"{outword}{plural_form}"))
            for form in singular[plural_key]:
                forms.append((f"{abbr}-{tweak(plural_form)}{form}", f"{outword}{plural_form}{form}"))
    return forms
