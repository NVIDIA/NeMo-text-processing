# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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

from nemo_text_processing.inverse_text_normalization.vi.graph_utils import (
    NEMO_ALPHA,
    GraphFst,
    delete_single_space,
    insert_space,
)
from nemo_text_processing.inverse_text_normalization.vi.utils import get_abs_path


class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying electronic: as URLs, email addresses, etc.
        e.g. c d f một a còng a b c dot e d u -> tokens { electronic { username: "cdf1" domain: "abc.edu" } }
    """

    def __init__(self):
        super().__init__(name="electronic", kind="classify")

        alpha_num = pynini.union(
            NEMO_ALPHA,
            pynini.string_file(get_abs_path("data/numbers/digit.tsv")),
            pynini.string_file(get_abs_path("data/numbers/zero.tsv")),
        )

        symbols = pynini.string_file(get_abs_path("data/electronic/symbols.tsv")).invert()
        url_symbols = pynini.string_file(get_abs_path("data/electronic/url_symbols.tsv")).invert()

        accepted_username = pynini.union(alpha_num, symbols)
        accepted_url_chars = pynini.union(alpha_num, url_symbols)
        process_dot = pynini.cross("chấm", ".")
        username = (
            pynutil.insert('username: "')
            + alpha_num
            + pynini.closure(delete_single_space + accepted_username)
            + pynutil.insert('"')
        )
        single_alphanum = pynini.closure(alpha_num + delete_single_space) + alpha_num
        server = pynini.union(
            single_alphanum,
            pynini.string_file(get_abs_path("data/electronic/server_name.tsv")),
            pynini.closure(NEMO_ALPHA, 2),  # At least 2 letters for server name
        )
        domain = pynini.union(
            single_alphanum,
            pynini.string_file(get_abs_path("data/electronic/domain.tsv")),
            pynini.closure(NEMO_ALPHA, 2),  # At least 2 letters for domain
        )
        multi_domain = (
            pynini.closure(process_dot + delete_single_space + domain + delete_single_space)
            + process_dot
            + delete_single_space
            + domain
        )
        domain_graph = pynutil.insert('domain: "') + server + delete_single_space + multi_domain + pynutil.insert('"')
        graph = (
            username
            + delete_single_space
            + pynutil.delete(pynini.union("a còng", "a móc", "a vòng"))
            + insert_space
            + delete_single_space
            + domain_graph
        )

        protocol_end = pynini.cross(pynini.union("w w w", "www"), "www")
        protocol_start = pynini.union(
            pynini.cross("h t t p", "http"), pynini.cross("h t t p s", "https")
        ) + pynini.cross(" hai chấm sẹc sẹc ", "://")

        # Domain part: server.domain (e.g., nvidia.com, www.nvidia.com)
        url_domain = server + delete_single_space + process_dot + delete_single_space + domain

        # Optional endings: /path or .vn or .com.vn
        url_ending = (
            delete_single_space
            + url_symbols
            + delete_single_space
            + pynini.union(domain, pynini.closure(accepted_url_chars + delete_single_space) + accepted_url_chars)
        )
        protocol = (
            pynini.closure(protocol_start, 0, 1)  # Optional http://
            + pynini.closure(
                protocol_end + delete_single_space + process_dot + delete_single_space, 0, 1
            )  # Optional www.
            + url_domain  # Required: server.domain
            + pynini.closure(url_ending, 0)  # Optional: /path or .vn
        )

        protocol = pynutil.insert('protocol: "') + protocol + pynutil.insert('"')
        graph = pynini.union(graph, protocol)

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
