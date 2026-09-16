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

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_ALPHA,
    NEMO_DIGIT,
    GraphFst,
    convert_space,
    insert_space,
)
from nemo_text_processing.text_normalization.en.utils import load_labels
from nemo_text_processing.text_normalization.ta.taggers.serial import digit_words, letter_names
from nemo_text_processing.text_normalization.ta.utils import get_abs_path

# Characters a URL path may hold; the symbols the whitelist speaks (# % &) are split off by the
# tokenizer before the text reaches the tagger, so a path stops at them.
PATH_SYMBOLS = "./-_~:+"

# Every piece of an address costs the same, so the cheapest reading is the one with the fewest
# pieces: a run of letters is read whole rather than split into shorter runs.
_PIECE_WEIGHT = 1.0


class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying electronic addresses, e.g.
        kumar@gmail.com -> tokens { name: "kumar எட் gmail டாட் காம்" }
        www.example.com/page2 -> tokens { name: "டபிள்யூ டபிள்யூ டபிள்யூ டாட் example டாட் காம் வெட்டுக்கோடு page இரண்டு" }
        192.168.1.1 -> tokens { name: "ஒன்று ஒன்பது இரண்டு டாட் ..." }
        @handle -> tokens { name: "எட் handle" }

    A Latin word is left for the voice to read; a lone letter is spelled, a digit is read on its
    own, and the symbols come from ``data/electronic/symbols.tsv``. A top-level domain reads from
    ``data/electronic/domains.tsv`` and is what makes a bare domain one (example.com); http://
    and https:// are not spoken.

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="classify", deterministic=deterministic)

        letter = letter_names()
        digit = digit_words()
        symbols = pynini.string_file(get_abs_path("data/electronic/symbols.tsv"))
        tld = pynutil.add_weight(
            pynini.string_map([(k, v) for k, v, *_ in load_labels(get_abs_path("data/electronic/domains.tsv"))]),
            -0.01,
        )
        tld_shape = pynini.project(tld, "input")
        dot = pynini.accep(".") @ symbols

        piece = pynini.union(pynini.closure(NEMO_ALPHA, 2), letter, digit, pynini.union(*PATH_SYMBOLS) @ symbols)
        piece = pynutil.add_weight(piece, _PIECE_WEIGHT)
        run = piece + pynini.closure(insert_space + piece)

        # Labels joined by dots, closing with a known top-level domain, possibly after a
        # second-level one that is also in the table (co.in).
        label_shape = pynini.closure(pynini.union(NEMO_ALPHA, NEMO_DIGIT, "-"), 1)
        domain_shape = pynini.closure(label_shape + ".", 1) + tld_shape + pynini.closure("." + tld_shape, 0, 1)
        domain = domain_shape @ (
            pynini.closure((label_shape @ run) + insert_space + dot + insert_space, 1)
            + tld
            + pynini.closure(insert_space + dot + insert_space + tld, 0, 1)
        )

        at = pynini.accep("@") @ symbols
        local = pynini.closure(pynini.union(NEMO_ALPHA, NEMO_DIGIT, *".-_+"), 1) @ run
        email = local + insert_space + at + insert_space + domain

        # "www" is spelled rather than read as a word, and outranks the plain domain reading.
        w = pynini.shortestpath(pynini.accep("W") @ letter).string()
        www = (
            pynutil.add_weight(
                pynutil.delete("www") + pynutil.insert(f"{w} {w} {w}") + insert_space + dot, -_PIECE_WEIGHT
            )
            + insert_space
        )
        protocol = pynutil.delete(pynini.union("http://", "https://"))
        path = pynini.closure(pynini.union(NEMO_ALPHA, NEMO_DIGIT, *PATH_SYMBOLS), 1) @ run
        slash = pynini.accep("/") @ symbols
        optional_path = pynini.closure(insert_space + slash + pynini.closure(insert_space + path, 0, 1), 0, 1)
        url = pynini.union(protocol + pynini.closure(www, 0, 1) + domain, www + domain, domain) + optional_path

        octet = pynini.closure(NEMO_DIGIT, 1, 3) @ (digit + pynini.closure(insert_space + digit))
        ip = octet + pynini.closure(insert_space + dot + insert_space + octet, 3, 3)

        handle = at + insert_space + (pynini.closure(pynini.union(NEMO_ALPHA, NEMO_DIGIT, "_"), 1) @ run)

        graph = pynini.union(
            pynutil.add_weight(email, 0.1),
            pynutil.add_weight(url, 0.2),
            pynutil.add_weight(ip, 0.1),
            pynutil.add_weight(handle, 0.3),
        )
        self.graph = convert_space(graph).optimize()
        self.fst = (pynutil.insert("name: \"") + self.graph + pynutil.insert("\"")).optimize()
