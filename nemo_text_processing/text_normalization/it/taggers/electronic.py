# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
    NEMO_SPACE,
    GraphFst,
    at,
    colon,
    domain_string,
    double_quotes,
    double_slash,
    http,
    https,
    period,
    protocol_string,
    username_string,
    www,
)
from nemo_text_processing.text_normalization.it.utils import get_abs_path, load_labels

# common_domains = [x[0] for x in load_labels(get_abs_path("data/electronic/domain.tsv"))]


class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying electronic: email addresses
        e.g. "abc.def@studente.università.it" -> electronic { username: "abc.def" domain: "studente.università.it" preserve_order: true }
        e.g. "www.abc.com/123" -> electronic { protocol: "www." domain: "abc.com/123" preserve_order: true }
        e.g. "https://github.com/NVIDIA/electronic.py" -> electronic { protocol: "https://" domain: "github.com/NVIDIA/electronic.py" preserve_order: true }
    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="classify", deterministic=deterministic)

        dot = pynini.accep(".")

        symbols = [x[0] for x in load_labels(get_abs_path("data/electronic/symbols.tsv"))]
        symbols = pynini.union(*symbols)
        symbols_no_period = pynini.difference(symbols, dot)
        accepted_characters = pynini.closure((NEMO_ALPHA | NEMO_DIGIT | symbols_no_period), 1)
        all_characters = pynini.closure((NEMO_ALPHA | NEMO_DIGIT | symbols), 1)

        # domains
        domain = dot + accepted_characters
        domain_graph = (
            pynutil.insert(domain_string + colon + NEMO_SPACE + double_quotes)
            + (accepted_characters + pynini.closure(domain, 1))
            + pynutil.insert(double_quotes)
        )

        # email
        username = (
            pynutil.insert(username_string + colon + NEMO_SPACE + double_quotes)
            + all_characters
            + pynutil.insert(double_quotes)
            + pynini.cross(at, NEMO_SPACE)
        )
        email = username + domain_graph

        # social media tags
        tag = (
            pynutil.delete(at)
            + pynutil.insert(username_string + colon + NEMO_SPACE + double_quotes)
            + (accepted_characters | (accepted_characters + pynini.closure(domain, 1)))
            + pynutil.insert(double_quotes)
        )

        # url
        protocol_start = pynini.accep(https + colon + double_slash) | pynini.accep(http + colon + double_slash)
        protocol_end = (
            pynini.accep(www + period)
            if deterministic
            else pynini.accep(www + period) | pynini.cross((www + period), "vu vu vu.")
        )
        protocol = protocol_start | protocol_end | (protocol_start + protocol_end)
        protocol = (
            pynutil.insert(protocol_string + colon + NEMO_SPACE + double_quotes)
            + protocol
            + pynutil.insert(double_quotes)
        )

        url = protocol + pynutil.insert(NEMO_SPACE) + (domain_graph)

        graph = url | domain_graph | email | tag
        self.graph = graph

        final_graph = self.add_tokens(self.graph + pynutil.insert(" preserve_order: true"))
        self.fst = final_graph.optimize()
