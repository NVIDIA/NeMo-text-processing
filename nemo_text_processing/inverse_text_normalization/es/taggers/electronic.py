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
from nemo_text_processing.inverse_text_normalization.es.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_ALPHA, GraphFst, insert_space
from pynini.lib import pynutil


class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying 'electronic' semiotic classes, i.e.
    email address (which get converted to "username" and "domain" fields),
    and URLS (which get converted to a "protocol" field).
        e.g. c d f uno arroba a b c punto e d u -> tokens { electronic { username: "cdf1" domain: "abc.edu" } }
        e.g. doble ve doble ve doble ve a b c punto e d u -> tokens { electronic { protocol: "www.abc.edu" } }
    """

    def __init__(self):
        super().__init__(name="electronic", kind="classify")

        delete_extra_space = pynutil.delete(" ")
        alpha_num = (
            NEMO_ALPHA
            | pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
            | pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        )

        symbols = pynini.string_file(get_abs_path("data/electronic/symbols.tsv")).invert()

        accepted_username = alpha_num | symbols
        process_dot = pynini.cross("punto", ".")
        username = (
            pynutil.insert("username: \"")
            + alpha_num
            + delete_extra_space
            + pynini.closure(accepted_username + delete_extra_space)
            + alpha_num
            + pynutil.insert("\"")
        )
        single_alphanum = pynini.closure(alpha_num + delete_extra_space) + alpha_num
        server = single_alphanum | pynini.string_file(get_abs_path("data/electronic/server_name.tsv")).invert()
        domain = single_alphanum | pynini.string_file(get_abs_path("data/electronic/domain.tsv")).invert()
        domain_graph = (
            pynutil.insert("domain: \"")
            + server
            + delete_extra_space
            + process_dot
            + delete_extra_space
            + domain
            + pynutil.insert("\"")
        )
        graph = (
            username + delete_extra_space + pynutil.delete("arroba") + insert_space + delete_extra_space + domain_graph
        )

        ############# url ###
        protocol_end = pynini.cross(pynini.union("www", "w w w", "doble ve doble ve doble ve"), "www")
        protocol_start = pynini.cross(pynini.union("http", "h t t p", "hache te te pe"), "http")
        protocol_start |= pynini.cross(pynini.union("https", "h t t p s", "hache te te pe ese"), "https")
        protocol_start += pynini.cross(" dos puntos barra barra ", "://")

        # e.g. .com, .es
        ending = (
            delete_extra_space
            + symbols
            + delete_extra_space
            + (domain | pynini.closure(accepted_username + delete_extra_space,) + accepted_username)
        )

        protocol = (
            pynini.closure(protocol_start, 0, 1)
            + protocol_end
            + delete_extra_space
            + process_dot
            + delete_extra_space
            + (pynini.closure(delete_extra_space + accepted_username, 1) | server)
            + pynini.closure(ending, 1)
        )
        protocol = pynutil.insert("protocol: \"") + protocol + pynutil.insert("\"")
        graph |= protocol
        ########

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
