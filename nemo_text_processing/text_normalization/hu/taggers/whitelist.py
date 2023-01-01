# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright (c) 2023, Jim O'Regan
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
import pynini
from nemo_text_processing.text_normalization.en.graph_utils import GraphFst, convert_space
from nemo_text_processing.text_normalization.hu.utils import get_abs_path, load_labels
from pynini.lib import pynutil


def naive_inflector(abbr: str, word: str):
    singular = {
        "er": "t nek rel ért ré ig ként ben en nél be re hez ből ről től",
        "ek": "et nek kel ért ké ig ként ben en nél be re hez ből ről től",
        "amm": "ot nak al ért á ig ként ban on nál ba ra hoz ból ról tól",
        "um": "ot nak mal ért má ig ként ban on nál ba ra hoz ból ról tól",
        "ok": "at nak kal ért ká ig ként ban on nál ba ra hoz ból ról tól",
        "ák": "at nak kal ért ká ig ként ban on nál ba ra hoz ból ról tól",
        "alék": "ot nak kal ért ká ig ként ban on nál ba ra hoz ból ról tól",
        "erc": "et nek cel ért cé ig ként ben en nél be re hez ből ről től",
        "óra": "át ának ával áért ává áig aként ában án ánál ába ára ához ából áról ától akor",
    }
    keys_sorted = sorted(singular, key=lambda k: len(k), reverse=True)
    plural = {
        "er": "ek",
        "erc": "ek",
        "amm": "ok",
        "um": "ok",
        "alék": "ok",
        "óra": "ák",
    }

    def get_key():
        if word == "óra":
            return "óra"
        for key in keys_sorted:
            if word.endswith(key):
                return key
        return None

    forms = []
    key = get_key()
    outword = word
    if outword[-1] in ["a", "e"]:
        outword = outword[:-1]

    def tweak(form: str) -> str:
        if outword == word:
            return form
        assert form[0] in ["a", "e", "á", "é"]
        return form[1:]

    for form in singular[key].split():
        forms.append((f"{abbr}-{tweak(form)}", f"{outword}{form}"))
    plural_form = plural[key]
    forms.append((f"{abbr}-{tweak(plural_form)}", f"{outword}{plural_form}"))
    for form in singular[plural_form].split():
        forms.append((f"{abbr}-{tweak(plural_form)}{form}", f"{outword}{plural_form}{form}"))
    return forms


def load_inflected(filename, skip_spaces = True):
    forms = []
    with open(filename) as tsv:
        for line in tsv.readlines():
            parts = line.strip().split("\t")
            forms.append((parts[0], parts[1]))
            if not (skip_spaces and " " in parts[1]):
                forms += naive_inflector(parts[0], parts[1])
    graph = pynini.string_map(forms)
    return graph


# TODO: inflected nouns/adjectives
# everything in whitelist.tsv until stb. has many inflected forms
class WhiteListFst(GraphFst):
    """
    Finite state transducer for classifying whitelist, e.g.
        "stb." -> tokens { name: "s a többi" }
    This class has highest priority among all classifier grammars. Whitelisted tokens are defined and loaded from "data/whitelist.tsv".

    Args:
        input_case: accepting either "lower_cased" or "cased" input.
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
        input_file: path to a file with whitelist replacements
    """

    def __init__(self, input_case: str, deterministic: bool = True, input_file: str = None):
        super().__init__(name="whitelist", kind="classify", deterministic=deterministic)

        def _get_whitelist_graph(input_case, file):
            whitelist = load_labels(file)
            if input_case == "lower_cased":
                whitelist = [[x[0].lower()] + x[1:] for x in whitelist]
            graph = pynini.string_map(whitelist)
            return graph

        graph = _get_whitelist_graph(input_case, get_abs_path("data/whitelist.tsv"))
        if not deterministic and input_case != "lower_cased":
            graph |= pynutil.add_weight(
                _get_whitelist_graph("lower_cased", get_abs_path("data/whitelist.tsv")), weight=0.0001
            )

        if input_file:
            whitelist_provided = _get_whitelist_graph(input_case, input_file)
            if not deterministic:
                graph |= whitelist_provided
            else:
                graph = whitelist_provided

        if not deterministic:
            units_graph = _get_whitelist_graph(input_case, file=get_abs_path("data/measures/measurements.tsv"))
            graph |= units_graph

        self.graph = graph
        self.final_graph = convert_space(self.graph).optimize()
        self.fst = (pynutil.insert("name: \"") + self.final_graph + pynutil.insert("\"")).optimize()
