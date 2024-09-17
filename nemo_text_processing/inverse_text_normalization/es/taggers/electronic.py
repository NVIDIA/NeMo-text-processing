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

from nemo_text_processing.inverse_text_normalization.en.utils import get_various_formats
from nemo_text_processing.inverse_text_normalization.es.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_CASED,
    INPUT_LOWER_CASED,
    MIN_POS_WEIGHT,
    NEMO_ALPHA,
    GraphFst,
    capitalized_input_graph,
    insert_space,
)
from nemo_text_processing.text_normalization.en.utils import load_labels


class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying 'electronic' semiotic classes, i.e.
    email address (which get converted to "username" and "domain" fields),
    and URLS (which get converted to a "protocol" field).
        e.g. c d f uno arroba a b c punto e d u -> tokens { electronic { username: "cdf1" domain: "abc.edu" } }
        e.g. doble ve doble ve doble ve a b c punto e d u -> tokens { electronic { protocol: "www.abc.edu" } }

    Args:
        input_case: accepting either "lower_cased" or "cased" input.
    """

    def __init__(self, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="electronic", kind="classify")

        delete_extra_space = pynutil.delete(" ")

        num = pynini.string_file(get_abs_path("data/numbers/digit.tsv")) | pynini.string_file(
            get_abs_path("data/numbers/zero.tsv")
        )
        if input_case == INPUT_CASED:
            num = capitalized_input_graph(num)

        alpha_num = (NEMO_ALPHA | num).optimize()

        symbols = pynini.string_file(get_abs_path("data/electronic/symbols.tsv")).invert()
        if input_case == INPUT_CASED:
            symbols = capitalized_input_graph(symbols)

        accepted_username = alpha_num | symbols
        dot = pynini.accep("punto")
        if input_case == INPUT_CASED:
            dot |= pynini.accep("Punto")
        process_dot = pynini.cross(dot, ".")
        alternative_dot = (
            pynini.closure(delete_extra_space, 0, 1) + pynini.accep(".") + pynini.closure(delete_extra_space, 0, 1)
        )
        username = (
            pynutil.insert("username: \"")
            + alpha_num
            + delete_extra_space
            + pynini.closure(accepted_username + delete_extra_space)
            + alpha_num
            + pynutil.insert("\"")
        )
        single_alphanum = pynini.closure(alpha_num + delete_extra_space) + alpha_num

        server_names = pynini.string_file(get_abs_path("data/electronic/server_name.tsv")).invert()
        if input_case == INPUT_CASED:
            server_names = capitalized_input_graph(server_names)
        server = single_alphanum | server_names | pynini.closure(NEMO_ALPHA, 2)

        if input_case == INPUT_CASED:
            domain = []
            # get domain formats
            for d in load_labels(get_abs_path("data/electronic/domain.tsv")):
                domain.extend(get_various_formats(d[0]))
            domain = pynini.string_map(domain).optimize()
        else:
            domain = pynini.string_file(get_abs_path("data/electronic/domain.tsv")).invert()

        domain = pynutil.add_weight(single_alphanum, weight=-0.0001) | domain | pynini.closure(NEMO_ALPHA, 2)

        domain_graph = (
            pynutil.insert("domain: \"")
            + server
            + ((delete_extra_space + process_dot + delete_extra_space) | alternative_dot)
            + domain
            + pynutil.insert("\"")
        )

        at = pynini.accep("arroba")
        if input_case == INPUT_CASED:
            at |= pynini.accep("Arroba")

        graph = username + delete_extra_space + pynutil.delete(at) + insert_space + delete_extra_space + domain_graph

        ############# url ###
        if input_case == INPUT_CASED:
            spoken_ws = pynini.union(
                "doble ve doble ve doble ve", "Doble Ve Doble Ve Doble Ve", "Doble ve doble ve doble ve"
            )
            protocol_end = pynini.cross(pynini.union(*get_various_formats("www")) | spoken_ws, "www")

            spoken_http = pynini.union("hache te te pe", "Hache te te pe", "Hache Te Te Pe")
            spoken_https = pynini.union("hache te te pe ese", "Hache te te pe ese", "Hache Te Te Pe Ese")
            protocol_start = pynini.cross(
                pynini.union(*get_various_formats("http")) | spoken_http, "http"
            ) | pynini.cross(pynini.union(*get_various_formats("https")) | spoken_https, "https")
        else:
            protocol_end = pynutil.add_weight(
                pynini.cross(pynini.union("www", "w w w", "doble ve doble ve doble ve"), "www"), MIN_POS_WEIGHT
            )
            protocol_start = pynutil.add_weight(
                pynini.cross(pynini.union("http", "h t t p", "hache te te pe"), "http"), MIN_POS_WEIGHT
            )
            protocol_start |= pynutil.add_weight(
                pynini.cross(pynini.union("https", "h t t p s", "hache te te pe ese"), "https"), MIN_POS_WEIGHT
            )

        protocol_start += pynini.cross(" dos puntos barra barra ", "://")

        # e.g. .com, .es
        ending = (
            delete_extra_space
            + symbols
            + delete_extra_space
            + (domain | pynini.closure(accepted_username + delete_extra_space,) + accepted_username)
        )

        protocol_default = (
            (
                (pynini.closure(delete_extra_space + accepted_username, 1) | server)
                | pynutil.add_weight(pynini.closure(NEMO_ALPHA, 1), weight=0.001)
            )
            + pynini.closure(ending, 1)
        ).optimize()

        protocol = (
            pynini.closure(protocol_start, 0, 1)
            + protocol_end
            + delete_extra_space
            + process_dot
            + delete_extra_space
            + protocol_default
        ).optimize()

        if input_case == INPUT_CASED:
            protocol |= (
                pynini.closure(protocol_start, 0, 1) + protocol_end + alternative_dot + protocol_default
            ).optimize()

        protocol |= pynini.closure(protocol_end + delete_extra_space + process_dot, 0, 1) + protocol_default

        protocol = pynutil.insert("protocol: \"") + protocol + pynutil.insert("\"")
        graph |= protocol

        if input_case == INPUT_CASED:
            graph = capitalized_input_graph(graph, capitalized_graph_weight=MIN_POS_WEIGHT)

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
