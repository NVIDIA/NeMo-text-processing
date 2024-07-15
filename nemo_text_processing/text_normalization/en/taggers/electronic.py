# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
    MIN_NEG_WEIGHT,
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_NOT_SPACE,
    NEMO_SIGMA,
    NEMO_UPPER,
    TO_UPPER,
    GraphFst,
    get_abs_path,
    insert_space,
    insert_username,
    insert_double_quotes,
    insert_domain,
    insert_protocol,
    accept_slash,
    accept_colon_double_slash,
    accept_colon_triple_slash,
    accept_file,
    accept_period,
    accept_at,
)


class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying electronic: as URLs, email addresses, etc.
        e.g. cdf1@abc.edu -> tokens { electronic { username: "cdf1" domain: "abc.edu" } }

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(
            name="electronic", kind="classify", deterministic=deterministic
        )

        # String literal insertions
        insert_username = pynutil.insert('username: "')
        insert_double_quotes = pynutil.insert('"')
        insert_domain = pynutil.insert('domain: "')
        insert_protocol = pynutil.insert('protocol: "')

        # String literal acceptors
        accept_slash = pynini.accep("/")
        accept_colon_double_slash = pynini.accep("://")
        accept_colon_triple_slash = pynini.accep(":///")
        accept_file = pynini.accep("file")
        accept_period = pynini.accep(".")
        accept_at = pynini.accep("@")

        if deterministic:
            numbers = NEMO_DIGIT
        else:
            numbers = insert_space + cardinal.long_numbers + insert_space

        cc_cues = pynutil.add_weight(
            pynini.string_file(get_abs_path("data/electronic/cc_cues.tsv")),
            MIN_NEG_WEIGHT,
        )

        accepted_symbols = pynini.project(
            pynini.string_file(get_abs_path("data/electronic/symbol.tsv")), "input"
        )
        accepted_common_domains = pynini.project(
            pynini.string_file(get_abs_path("data/electronic/domain.tsv")), "input"
        )

        dict_words = pynutil.add_weight(
            pynini.string_file(get_abs_path("data/electronic/words.tsv")),
            MIN_NEG_WEIGHT,
        )

        dict_words_without_delimiter = dict_words + pynini.closure(
            pynutil.add_weight(insert_space + dict_words, MIN_NEG_WEIGHT), 1
        )
        dict_words_graph = dict_words_without_delimiter | dict_words

        all_accepted_symbols_start = (
            dict_words_graph
            | pynini.closure(TO_UPPER)
            | pynini.closure(NEMO_UPPER)
            | accepted_symbols
        ).optimize()

        all_accepted_symbols_end = (
            dict_words_graph
            | numbers
            | pynini.closure(TO_UPPER)
            | pynini.closure(NEMO_UPPER)
            | accepted_symbols
        ).optimize()

        graph_symbols = pynini.string_file(
            get_abs_path("data/electronic/symbol.tsv")
        ).optimize()
        username = (NEMO_ALPHA | dict_words_graph) + pynini.closure(
            NEMO_ALPHA | numbers | accepted_symbols | dict_words_graph
        )

        username = (
            insert_username + username + insert_double_quotes + pynini.cross("@", " ")
        )

        domain_graph = all_accepted_symbols_start + pynini.closure(
            all_accepted_symbols_end
            | pynutil.add_weight(accepted_common_domains, MIN_NEG_WEIGHT)
        )

        protocol_symbols = pynini.closure(
            (graph_symbols | pynini.cross(":", "colon")) + insert_space
        )
        protocol_start = (
            pynini.cross("https", "HTTPS ") | pynini.cross("http", "HTTP ")
        ) + (accept_colon_double_slash @ protocol_symbols)
        protocol_file_start = (
            accept_file + insert_space + (accept_colon_triple_slash @ protocol_symbols)
        )

        protocol_end = pynutil.add_weight(
            pynini.cross("www", "WWW ") + accept_period @ protocol_symbols, -1000
        )
        protocol = (
            protocol_file_start
            | protocol_start
            | protocol_end
            | (protocol_start + protocol_end)
        )

        domain_graph_with_class_tags = (
            insert_domain
            + pynini.compose(
                NEMO_ALPHA
                + pynini.closure(NEMO_NOT_SPACE)
                + (NEMO_ALPHA | NEMO_DIGIT | accept_slash),
                domain_graph,
            ).optimize()
            + insert_double_quotes
        )

        protocol = (
            insert_protocol
            + pynutil.add_weight(protocol, MIN_NEG_WEIGHT)
            + insert_double_quotes
        )
        # email
        graph = pynini.compose(
            NEMO_SIGMA + accept_at + NEMO_SIGMA + accept_period + NEMO_SIGMA,
            username + domain_graph_with_class_tags,
        )

        # abc.com, abc.com/123-sm
        # when only domain, make sure it starts and end with NEMO_ALPHA
        graph |= (
            insert_domain
            + pynini.compose(
                NEMO_ALPHA
                + pynini.closure(NEMO_NOT_SPACE)
                + accepted_common_domains
                + pynini.closure(pynini.difference(NEMO_NOT_SPACE, accept_period)),
                domain_graph,
            ).optimize()
            + insert_double_quotes
        )
        # www.abc.com/sdafsdf, or https://www.abc.com/asdfad or www.abc.abc/asdfad
        graph |= protocol + insert_space + domain_graph_with_class_tags

        if deterministic:
            # credit card cues
            numbers = pynini.closure(NEMO_DIGIT, 4, 16)
            cc_phrases = (
                insert_protocol
                + cc_cues
                + insert_domain
                + numbers
                + insert_double_quotes
            )
            graph |= cc_phrases

        final_graph = self.add_tokens(graph)

        self.fst = final_graph.optimize()
