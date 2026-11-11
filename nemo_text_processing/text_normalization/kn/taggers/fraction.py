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

from nemo_text_processing.text_normalization.kn.graph_utils import (
    NEMO_ALL_DIGIT,
    NEMO_DIGIT,
    COMMA ,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
)
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
        special_fractions = pynini.string_file(get_abs_path("data/fraction/special_fractions.tsv")) 
        scales = pynini.union(
            pynini.string_file(get_abs_path("data/numbers/scale_suffixes.tsv")),
            pynini.string_file(get_abs_path("data/fraction/scale_suffixes.tsv")),
        ) 
        locative_suffix = pynini.string_file(get_abs_path("data/fraction/locative_suffixes.tsv"))
        lengthen_suffix = pynini.string_file(get_abs_path("data/fraction/lengthen_suffixes.tsv"))

        remove_slash_spaces = pynini.cdrewrite(pynini.cross(NEMO_SPACE + "/" + NEMO_SPACE, "/"), "", "", NEMO_SIGMA)
        symbol_to_key = remove_slash_spaces @ special_fractions

        alone = symbol_to_key @ words 
        half_after_integer = symbol_to_key @ pynini.cross("ARDHA", "VARE") @ words 
        quarter_after_integer = symbol_to_key @ pynini.accep("KAALU") @ words 
        three_quarters_after_integer = symbol_to_key @ pynini.accep("MUKKAALU") @ words 

        scale_to_key = pynini.invert(scales) @ pynini.union("thousand", "lakh", "crore")
        to_genitive_words = (scale_to_key + pynutil.insert("_gen")) @ scales
        to_locative_words = (scale_to_key + pynutil.insert("_loc")) @ scales

        ends_with_scale = NEMO_SIGMA + pynini.project(scale_to_key, "input")
        not_scale = pynini.difference(NEMO_SIGMA, ends_with_scale).optimize()

        to_genitive = pynini.cdrewrite(to_genitive_words, "", "[EOS]", NEMO_SIGMA)
        to_locative = pynini.cdrewrite(to_locative_words | locative_suffix, "", "[EOS]", NEMO_SIGMA)
        lengthen = pynini.cdrewrite(lengthen_suffix, "", "[EOS]", NEMO_SIGMA)

        plain = number @ not_scale
        lengthened = plain @ lengthen
        with_scale = (number @ ends_with_scale @ to_genitive) + NEMO_SPACE

        negative_sign = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        with_integer = pynini.union(
            lengthened + pynutil.delete(NEMO_SPACE) + half_after_integer,
            lengthened + pynutil.delete(NEMO_SPACE) + quarter_after_integer,
            plain + NEMO_SPACE + three_quarters_after_integer,
            with_scale + alone,
        )
        special = pynutil.insert('morphosyntactic_features: "') + (with_integer | alone) + pynutil.insert('"')

        slash = pynini.cross(pynini.union("/", NEMO_SPACE + "/" + NEMO_SPACE), NEMO_SPACE)
        integer_part = pynutil.insert('integer_part: "') + number + pynutil.insert('"')
        numerator = pynutil.insert('numerator: "') + number + pynutil.insert('"')
        denominator = pynutil.insert('denominator: "') + (number @ to_locative) + pynutil.insert('"')
        fraction_graph = pynini.closure(integer_part + NEMO_SPACE, 0, 1) + numerator + slash + denominator

        ascii_number = pynini.closure(NEMO_DIGIT | COMMA, 1)
        kannada_number = pynini.closure(pynini.difference(NEMO_ALL_DIGIT, NEMO_DIGIT) | COMMA, 1)
        any_number = pynini.closure(NEMO_ALL_DIGIT | COMMA, 1)

        slash_in = pynini.union("/", NEMO_SPACE + "/" + NEMO_SPACE)

        ascii_pair = ascii_number + slash_in + ascii_number
        kannada_pair = kannada_number + slash_in + kannada_number
        same_script_pairs = (ascii_pair | kannada_pair).optimize()
        mixed_pair = pynini.difference(any_number + slash_in + any_number, same_script_pairs).optimize()

        same_script = pynini.union(
            pynini.closure(ascii_number + NEMO_SPACE, 0, 1) + ascii_pair,
            pynini.closure(kannada_number + NEMO_SPACE, 0, 1) + kannada_pair,
        ).optimize()

        special_symbols = pynini.project(alone, "input")
        with_special = pynini.union(special_symbols, NEMO_SIGMA + NEMO_SPACE + special_symbols).optimize()
        generic = pynini.compose(pynini.difference(same_script, with_special), fraction_graph)

        passthrough = (
            pynutil.insert('morphosyntactic_features: "')
            + pynini.closure(pynini.accep("-"), 0, 1)
            + mixed_pair
            + pynutil.insert('"')
        )

        self.fst = self.add_tokens((negative_sign + (generic | special)) | passthrough).optimize()