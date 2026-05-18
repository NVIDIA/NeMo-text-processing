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

"""
US-style postal address surface for Spanish TN (embedded in ``MeasureFst`` as
``units: "address_us_es"``).

Street numbers and ZIP are Spanish; street types, states, and ordinals (e.g. ``42nd``)
use English expansions from shared ``en/data/address/`` lexicons.
"""

import pynini
from pynini.examples import plurals
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    NEMO_UPPER,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.en.taggers.cardinal import CardinalFst as EnCardinalFst
from nemo_text_processing.text_normalization.en.taggers.ordinal import OrdinalFst as OrdinalTagger
from nemo_text_processing.text_normalization.en.taggers.whitelist import get_formats
from nemo_text_processing.text_normalization.en.utils import get_abs_path as en_get_abs_path
from nemo_text_processing.text_normalization.en.utils import load_labels
from nemo_text_processing.text_normalization.en.verbalizers.ordinal import OrdinalFst as OrdinalVerbalizer
from nemo_text_processing.text_normalization.es.graph_utils import normalize_spanish_cardinal_for_us_address_street
from nemo_text_processing.text_normalization.es.utils import get_abs_path


class AddressUSSurfaceFst(GraphFst):
    """
    Surface FST for US addresses inside Spanish sentences.

    Output is the spoken string stored in ``measure { units: "address_us_es" cardinal { integer: "..." } }``.
    Not registered in ``tokenize_and_classify``; consumed by :class:`~nemo_text_processing.text_normalization.es.taggers.measure.MeasureFst`.

    Args:
        cardinal: Spanish :class:`~nemo_text_processing.text_normalization.es.taggers.cardinal.CardinalFst`
        deterministic: passed to English ordinal/cardinal helpers
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="address_us_es_surface", kind="classify", deterministic=deterministic)

        graph_direction = pynini.string_file(get_abs_path("data/address/direction.tsv"))
        graph_zip_digit = pynini.string_file(get_abs_path("data/address/zip_digit.tsv"))
        graph_suite_designator = pynini.string_file(get_abs_path("data/address/suite_designator.tsv"))
        graph_apt_designator = pynini.string_file(get_abs_path("data/address/apt_designator.tsv"))
        graph_unit_designator = pynini.string_file(get_abs_path("data/address/unit_designator.tsv"))
        graph_po_box = pynini.string_file(get_abs_path("data/address/po_box.tsv"))
        
        en_cardinal = EnCardinalFst(deterministic=deterministic)
        g = cardinal.graph

        ordinal_en = pynini.compose(
            pynutil.insert('integer: "') + OrdinalTagger(cardinal=en_cardinal).graph + pynutil.insert('"'),
            OrdinalVerbalizer().graph,
        )

        address_num = NEMO_DIGIT ** (1, 2) @ cardinal.graph_hundreds_component_at_least_one_none_zero_digit
        address_num += insert_space + NEMO_DIGIT**2 @ (
            pynini.closure(pynini.cross("0", "cero "), 0, 1)
            + cardinal.graph_hundreds_component_at_least_one_none_zero_digit
        )
        address_num = pynini.compose(NEMO_DIGIT ** (3, 4), address_num)
        address_num = normalize_spanish_cardinal_for_us_address_street(
            plurals._priority_union(address_num, g, NEMO_SIGMA).optimize()
        )

        direction = pynini.closure(
            pynini.accep(NEMO_SPACE) + graph_direction + pynini.closure(pynutil.delete("."), 0, 1),
            0,
            1,
        )

        address_words = get_formats(en_get_abs_path("data/address/address_word.tsv"))
        street = (
            pynini.accep(NEMO_SPACE)
            + (pynini.closure(ordinal_en, 0, 1) | NEMO_UPPER + pynini.closure(NEMO_ALPHA, 1))
            + NEMO_SPACE
            + pynini.closure(NEMO_UPPER + pynini.closure(NEMO_ALPHA) + NEMO_SPACE)
            + address_words
        )

        zip_five = (
            graph_zip_digit
            + insert_space
            + graph_zip_digit
            + insert_space
            + graph_zip_digit
            + insert_space
            + graph_zip_digit
            + insert_space
            + graph_zip_digit
        ).optimize()

        city = pynini.closure(NEMO_ALPHA | pynini.accep(NEMO_SPACE), 1)
        city = pynini.closure(pynini.accep(",") + pynini.accep(NEMO_SPACE) + city, 0, 1)

        states = load_labels(en_get_abs_path("data/address/state.tsv"))
        states_extra = [(x, f"{y[0]}.{y[1:]}") for x, y in states]
        states.extend(states_extra)
        state = pynini.closure(
            pynini.accep(",") + pynini.accep(NEMO_SPACE) + pynini.invert(pynini.string_map(states)), 0, 1
        )

        zip_code = pynini.compose(NEMO_DIGIT**5, zip_five)
        zip_code = pynini.closure(
            pynini.closure(pynini.accep(","), 0, 1) + pynini.accep(NEMO_SPACE) + zip_code,
            0,
            1,
        )
        tail = pynini.closure(city + state + zip_code, 0, 1).optimize()

        suite_num = normalize_spanish_cardinal_for_us_address_street(
            (pynini.closure(NEMO_DIGIT, 1, 4) @ g).optimize()
        )
        unit_num = normalize_spanish_cardinal_for_us_address_street(
            (pynini.closure(NEMO_DIGIT, 1, 3) @ g).optimize()
        )

        comma_sp = pynini.accep(",") + pynini.accep(NEMO_SPACE)
        suite = graph_suite_designator + pynini.closure(NEMO_SPACE, 0, 1) + suite_num
        apt = graph_apt_designator + pynini.closure(NEMO_DIGIT | NEMO_UPPER, 1, 4)
        unit = graph_unit_designator + unit_num
        middle = pynini.closure(comma_sp + (suite | apt | unit), 0, 3).optimize()

        po_box = (
            graph_po_box
            + normalize_spanish_cardinal_for_us_address_street(pynini.closure(NEMO_DIGIT, 1, 4) @ g)
            + tail
        ).optimize()

        standard = address_num + direction + street + middle + tail
        hyphen = pynini.accep("-")
        alpha_chars = NEMO_ALPHA | hyphen
        standard_eos = (
            address_num
            + direction
            + street
            + middle
            + pynini.accep(".")
            + pynini.closure(NEMO_SPACE, 1, 2)
            + NEMO_UPPER
            + pynini.closure(alpha_chars)
        )
        standard |= pynutil.add_weight(standard_eos, -0.001)
        standard |= address_num + direction + street + middle + pynini.closure(pynini.cross(".", ""), 0, 1)

        self.graph = (po_box | standard.optimize()).optimize()
