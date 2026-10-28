# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.kn.graph_utils import NEMO_SIGMA, NEMO_SPACE, GraphFst
from nemo_text_processing.text_normalization.kn.utils import get_abs_path


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction
        ೨೩ ೪/೬-> fraction { integer_part: "ಇಪ್ಪತ್ತಮೂರು" numerator: "ನಾಲ್ಕು" denominator: "ಆರರಲ್ಲಿ" }
        ೪/೦-> fraction { numerator: "ನಾಲ್ಕು" denominator: "ಸೊನ್ನೆ" }


    Args:
        cardinal: cardinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="fraction", kind="classify", deterministic=deterministic)

        number = cardinal.final_graph
        words = pynini.string_file(get_abs_path("data/fraction/words.tsv"))
        scales = pynini.union(
            pynini.string_file(get_abs_path("data/numbers/scale_suffixes.tsv")),
            pynini.string_file(get_abs_path("data/fraction/scale_suffixes.tsv")),
        )
        specials = pynini.invert(pynini.string_file(get_abs_path("data/fraction/special_fractions.tsv")))
        locative_suffix = pynini.string_file(get_abs_path("data/fraction/locative_suffixes.tsv"))
        lengthen_suffix = pynini.string_file(get_abs_path("data/fraction/lengthen_suffixes.tsv"))

        spaced_slash = pynini.cdrewrite(pynini.cross("/", NEMO_SPACE + "/" + NEMO_SPACE), "", "", NEMO_SIGMA)

        def get(table, key):
            return pynini.project(pynini.accep(key) @ table, "output")

        def field(name, graph):
            return pynutil.insert(f'{name}: "') + graph + pynutil.insert('"')

        def at_end(rule):
            return pynini.cdrewrite(rule, "", "[EOS]", NEMO_SIGMA)

        half, quarter, three_quarters = (get(specials, key) for key in ("ONE_HALF", "ONE_QUARTER", "THREE_QUARTERS"))

        half, quarter, three_quarters = (
            pynini.union(lit, pynini.project(lit @ spaced_slash, "output")).optimize()
            for lit in (half, quarter, three_quarters)
        )
        ardha, kaalu, mukkaalu, vare = (get(words, key) for key in ("ARDHA", "KAALU", "MUKKAALU", "VARE"))

        scale_words, genitive, locative = [], [], []
        for key in ("thousand", "lakh", "crore"):
            word = get(scales, key)
            scale_words.append(word)
            genitive.append(pynini.cross(word, get(scales, key + "_gen")))
            locative.append(pynini.cross(word, get(scales, key + "_loc")))

        to_genitive = at_end(pynini.union(*genitive))
        to_locative = at_end(pynini.union(*locative) | locative_suffix)
        lengthen = at_end(lengthen_suffix)

        ends_with_scale = NEMO_SIGMA + pynini.union(*scale_words)
        not_scale = pynini.difference(NEMO_SIGMA, ends_with_scale).optimize()

        plain = number @ not_scale
        lengthened = plain @ lengthen
        with_scale = (number @ ends_with_scale @ to_genitive) + pynutil.insert(NEMO_SPACE)

        negative_sign = pynini.closure(pynini.cross("-", 'negative: "true" '), 0, 1)

        pairs = ((half, ardha), (quarter, kaalu), (three_quarters, mukkaalu))
        alone = pynini.union(*(pynini.cross(symbol, word) for symbol, word in pairs))

        after_number = [
            (lengthened, half, vare),
            (lengthened, quarter, kaalu),
            (plain, three_quarters, NEMO_SPACE + mukkaalu),
            *((with_scale, symbol, word) for symbol, word in pairs),
        ]
        with_integer = pynini.union(
            *(
                whole + pynini.cross(NEMO_SPACE + symbol, "") + pynutil.insert(text)
                for whole, symbol, text in after_number
            )
        )
        special = field("morphosyntactic_features", with_integer | alone)

        slash = pynini.cross(pynini.union("/", NEMO_SPACE + "/" + NEMO_SPACE), NEMO_SPACE)
        generic = (
            pynini.closure(field("integer_part", number) + NEMO_SPACE, 0, 1)
            + field("numerator", number)
            + slash
            + field("denominator", number @ to_locative)
        )

        symbols = pynini.project(alone, "input")
        with_symbol = pynini.union(symbols, NEMO_SIGMA + NEMO_SPACE + symbols).optimize()
        generic = pynini.compose(pynini.difference(NEMO_SIGMA, with_symbol), generic)

        self.fst = self.add_tokens(negative_sign + (generic | special)).optimize()
