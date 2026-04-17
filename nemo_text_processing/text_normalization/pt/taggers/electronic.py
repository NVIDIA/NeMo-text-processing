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

from nemo_text_processing.text_normalization.pt.graph_utils import (
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_SPACE,
    GraphFst,
)
from nemo_text_processing.text_normalization.pt.utils import get_abs_path, load_labels


class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying electronic strings in pt-BR:
        abc@hotmail.com -> electronic { username: "abc" domain: "hotmail.com" preserve_order: true }
        https://www.abc.com -> electronic { protocol: "https://www." domain: "abc.com" preserve_order: true }
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="classify", deterministic=deterministic)

        full_stop = pynini.accep(".")
        at_symbol = "@"
        protocol_string = "protocol"
        domain_string = "domain"
        username_string = "username"
        http = "http"
        https = "https"
        www = "www"

        symbols = [x[0] for x in load_labels(get_abs_path("data/electronic/symbols.tsv"))]
        symbols = pynini.union(*symbols)
        symbols_no_full_stop = pynini.difference(symbols, full_stop)
        accepted_characters = pynini.closure((NEMO_ALPHA | NEMO_DIGIT | symbols_no_full_stop), 1)
        all_characters = pynini.closure((NEMO_ALPHA | NEMO_DIGIT | symbols), 1)

        domain_component = full_stop + accepted_characters
        domain_graph = (
            pynutil.insert(domain_string + ': "')
            + (accepted_characters + pynini.closure(domain_component, 1))
            + pynutil.insert('"')
        )

        username = (
            pynutil.insert(username_string + ': "')
            + all_characters
            + pynutil.insert('"')
            + pynini.cross(at_symbol, NEMO_SPACE)
        )
        email = username + domain_graph

        social_tag = (
            pynini.cross(at_symbol, "")
            + pynutil.insert(username_string + ': "')
            + (accepted_characters | (accepted_characters + pynini.closure(domain_component, 1)))
            + pynutil.insert('"')
        )

        protocol_start = pynini.accep(https + "://") | pynini.accep(http + "://")
        protocol_end = pynini.accep(www + ".")
        if not deterministic:
            protocol_end |= pynini.cross(www + ".", "dáblio dáblio dáblio.")

        protocol = protocol_start | protocol_end | (protocol_start + protocol_end)
        protocol = pynutil.insert(protocol_string + ': "') + protocol + pynutil.insert('"')
        url = protocol + pynutil.insert(NEMO_SPACE) + domain_graph

        graph = url | domain_graph | email | social_tag
        self.graph = graph

        final_graph = self.add_tokens(self.graph + pynutil.insert(" preserve_order: true"))
        self.fst = final_graph.optimize()
