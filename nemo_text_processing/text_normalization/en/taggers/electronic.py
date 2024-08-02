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

from nemo_text_processing.text_normalization.en.graph_utils import (  # common string literals
    MIN_NEG_WEIGHT,
    MIN_POS_WEIGHT,
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_NOT_SPACE,
    NEMO_SIGMA,
    NEMO_UPPER,
    TO_UPPER,
    insert_space,
    GraphFst,
    get_abs_path,
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

        if deterministic:
            numbers = NEMO_DIGIT
        else:
            numbers = pynutil.insert(" ") + cardinal.long_numbers + pynutil.insert(" ")

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
            pynutil.add_weight(pynutil.insert(" ") + dict_words, MIN_NEG_WEIGHT),
            1,
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
            pynutil.insert('username: "')
            + username
            + pynutil.insert('"')
            + pynini.cross("@", " ")
        )

        # dot = pynini.accep(".")

        # symbols = pynini.project(
        #     pynini.string_file(get_abs_path("data/electronic/symbol.tsv")), "input"
        # )
        # # symbols = pynini.union(*symbols)
        # # all symbols
        # symbols_no_period = pynini.difference(
        #     symbols, dot
        # )  # alphabet of accepted symbols excluding the '.'
        # accepted_characters = pynini.closure(
        #     (NEMO_ALPHA | NEMO_DIGIT | symbols_no_period), 1
        # )  # alphabet of accepted chars excluding the '.'

        # # domains
        # domain = dot + accepted_characters
        # domain_graph_with_class_tags = (
        #     pynutil.insert('domain: "')
        #     + (accepted_characters + pynini.closure(domain, 1))
        #     + pynutil.insert('"')
        # )

        domain_graph = all_accepted_symbols_start + pynini.closure(
            all_accepted_symbols_end
            | pynutil.add_weight(accepted_common_domains, MIN_NEG_WEIGHT)
        )

        # remove_dot = pynini.cdrewrite(pynini.cross(".", " ."), "", "[EOS]", NEMO_SIGMA)
        # domain_graph = domain_graph @ remove_dot
        # domain_graph.optimize()

        protocol_symbols = pynini.closure(
            (graph_symbols | pynini.cross(":", "colon")) + pynutil.insert(" ")
        )
        protocol_start = (
            pynini.cross("https", "HTTPS ") | pynini.cross("http", "HTTP ")
        ) + (pynini.accep("://") @ protocol_symbols)

        protocol_file_start = (
            pynini.accep("file")
            + insert_space
            + (pynini.accep(":///") @ protocol_symbols)
        )

        protocol_end = pynutil.add_weight(
            pynini.cross("www", "WWW ") + pynini.accep(".") @ protocol_symbols, -1000
        )

        protocol = (
            protocol_file_start
            | protocol_start
            | protocol_end
            | (protocol_start + protocol_end)
        )

        domain_graph_with_class_tags = (
            pynutil.insert('domain: "')
            + pynini.compose(
                NEMO_ALPHA
                + pynini.closure(NEMO_NOT_SPACE)
                + (NEMO_ALPHA | NEMO_DIGIT | pynini.accep("/")),
                domain_graph,
            ).optimize()
            + pynutil.insert('"')
        )

        protocol = (
            pynutil.insert('protocol: "')
            + pynutil.add_weight(protocol, MIN_NEG_WEIGHT)
            + pynutil.insert('"')
        )

        # email
        graph = pynini.compose(
            NEMO_SIGMA
            + pynini.accep("@")
            + NEMO_SIGMA
            + pynini.accep(".")
            + NEMO_SIGMA,
            username + domain_graph_with_class_tags,
        )

        """
        # abc.com, abc.com/123-sm
        # when only domain, make sure it starts and ends with NEMO_ALPHA
        graph |= (
            pynutil.insert('domain: "')
            + pynini.compose(
                NEMO_ALPHA
                + pynini.closure(NEMO_NOT_SPACE)
                + accepted_common_domains
                + pynini.closure(NEMO_NOT_SPACE),
                domain_graph,
                # domain_graph_with_class_tags,
            ).optimize()
            + pynutil.insert('"')
        )
        """
        dot = pynini.accep(".")
        # Include for the correct transduction of the money graph
        dollar = pynini.accep("$")
        exclude = dot | dollar
        symbols_filtered = pynini.difference(accepted_symbols, exclude)
        accepted_characters = pynini.closure(
            (NEMO_ALPHA | NEMO_DIGIT | symbols_filtered), 2
        )
        domain_component = dot + accepted_characters
        graph_domain = (
            pynutil.insert('domain: "')
            + (accepted_characters + pynini.closure(domain_component, 1))
            + pynutil.insert('"')
        ).optimize()

        graph |= pynutil.add_weight(graph_domain, MIN_POS_WEIGHT)

        # www.abc.com/sdafsdf, or https://www.abc.com/asdfad or www.abc.abc/asdfad
        graph |= protocol + pynutil.insert(" ") + domain_graph_with_class_tags

        # graph |= pynini.closure(domain_graph_with_class_tags).optimize()

        # mail.nasa.gov, or abc.some.university.edu
        # period = pynini.accep(".")
        # domain_graph_no_period = domain_graph - period
        # graph |= domain_graph + pynini.closure(period + domain_graph_no_period)

        if deterministic:
            # credit card cues
            numbers = pynini.closure(NEMO_DIGIT, 4, 16)
            cc_phrases = (
                pynutil.insert('protocol: "')
                + cc_cues
                + pynutil.insert('" domain: "')
                + numbers
                + pynutil.insert('"')
            )
            graph |= cc_phrases

        final_graph = self.add_tokens(graph)

        self.fst = final_graph.optimize()
