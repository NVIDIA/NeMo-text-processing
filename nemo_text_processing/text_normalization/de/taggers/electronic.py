# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.de.utils import get_abs_path, load_labels
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_ALPHA, NEMO_DIGIT, GraphFst, insert_space


class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying electronic: email addresses
        e.g. "abc@hotmail.com" -> electronic { username: "abc" domain: "hotmail.com" preserve_order: true }
        e.g. "www.abc.com/123" -> electronic { protocol: "www." domain: "abc.com/123" preserve_order: true }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="classify", deterministic=deterministic)

        dot = pynini.accep(".")

        symbols = [x[0] for x in load_labels(get_abs_path("data/electronic/symbols.tsv"))]
        symbols = pynini.union(*symbols)
        # all symbols
        symbols_no_period = pynini.difference(symbols, dot)  # alphabet of accepted symbols excluding the '.'
        accepted_characters = pynini.closure(
            (NEMO_ALPHA | NEMO_DIGIT | symbols_no_period), 1
        )  # alphabet of accepted chars excluding the '.'
        all_characters = pynini.closure(
            (NEMO_ALPHA | NEMO_DIGIT | symbols), 1
        )  # alphabet of accepted chars including the '.'

        # domains
        domain = dot + accepted_characters
        domain_graph = (
            pynutil.insert('domain: "')
            + (accepted_characters + pynini.closure(domain, 1))
            + dot.ques
            + pynutil.insert('"')
        )

        # email
        username = pynutil.insert('username: "') + all_characters + pynutil.insert('"') + pynini.cross("@", " ")
        email = username + domain_graph

        # social media tags
        tag = (
            pynini.cross("@", "")
            + pynutil.insert('username: "')
            + (accepted_characters | (accepted_characters + pynini.closure(domain, 1)))
            + dot.ques
            + pynutil.insert('"')
        )

        # url
        protocol_start = pynini.accep("https://") | pynini.accep("http://")
        protocol_end = pynini.accep("www.")
        protocol = protocol_start | protocol_end | (protocol_start + protocol_end)
        protocol = pynutil.insert('protocol: "') + protocol + pynutil.insert('"')
        url = protocol + insert_space + (domain_graph)

        graph = url | domain_graph | email | tag
        self.graph = graph

        final_graph = self.add_tokens(self.graph + pynutil.insert(" preserve_order: true"))
        self.fst = final_graph.optimize()
