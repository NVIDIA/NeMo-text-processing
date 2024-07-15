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
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_NOT_SPACE,
    NEMO_SIGMA,
    NEMO_SPACE,
    NEMO_UPPER,
    TO_UPPER,
    GraphFst,
    at,
    colon,
    domain_string,
    double_quotes,
    double_slash,
    file,
    get_abs_path,
    http,
    https,
    period,
    protocol_string,
    slash,
    triple_slash,
    username_string,
    www,
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
        super().__init__(name="electronic", kind="classify", deterministic=deterministic)

        if deterministic:
            numbers = NEMO_DIGIT
        else:
            numbers = pynutil.insert(NEMO_SPACE) + cardinal.long_numbers + pynutil.insert(NEMO_SPACE)

        cc_cues = pynutil.add_weight(pynini.string_file(get_abs_path("data/electronic/cc_cues.tsv")), MIN_NEG_WEIGHT,)

        accepted_symbols = pynini.project(pynini.string_file(get_abs_path("data/electronic/symbol.tsv")), "input")
        accepted_common_domains = pynini.project(
            pynini.string_file(get_abs_path("data/electronic/domain.tsv")), "input"
        )

        dict_words = pynutil.add_weight(pynini.string_file(get_abs_path("data/electronic/words.tsv")), MIN_NEG_WEIGHT,)

        dict_words_without_delimiter = dict_words + pynini.closure(
            pynutil.add_weight(pynutil.insert(NEMO_SPACE) + dict_words, MIN_NEG_WEIGHT), 1,
        )
        dict_words_graph = dict_words_without_delimiter | dict_words

        all_accepted_symbols_start = (
            dict_words_graph | pynini.closure(TO_UPPER) | pynini.closure(NEMO_UPPER) | accepted_symbols
        ).optimize()

        all_accepted_symbols_end = (
            dict_words_graph | numbers | pynini.closure(TO_UPPER) | pynini.closure(NEMO_UPPER) | accepted_symbols
        ).optimize()

        graph_symbols = pynini.string_file(get_abs_path("data/electronic/symbol.tsv")).optimize()
        username = (NEMO_ALPHA | dict_words_graph) + pynini.closure(
            NEMO_ALPHA | numbers | accepted_symbols | dict_words_graph
        )

        username = (
            pynutil.insert(username_string + colon + NEMO_SPACE + double_quotes)
            + username
            + pynutil.insert(double_quotes)
            + pynini.cross(at, NEMO_SPACE)
        )

        domain_graph = all_accepted_symbols_start + pynini.closure(
            all_accepted_symbols_end | pynutil.add_weight(accepted_common_domains, MIN_NEG_WEIGHT)
        )

        protocol_symbols = pynini.closure((graph_symbols | pynini.cross(colon, "colon")) + pynutil.insert(NEMO_SPACE))
        protocol_start = (
            pynini.cross(https, (https.upper() + NEMO_SPACE)) | pynini.cross(http, (http.upper() + NEMO_SPACE))
        ) + (pynini.accep(colon + double_slash) @ protocol_symbols)
        protocol_file_start = (
            pynini.accep(file) + pynutil.insert(NEMO_SPACE) + (pynini.accep(colon + triple_slash) @ protocol_symbols)
        )

        protocol_end = pynutil.add_weight(
            pynini.cross(www, (www.upper() + NEMO_SPACE)) + pynini.accep(period) @ protocol_symbols, -1000,
        )
        protocol = protocol_file_start | protocol_start | protocol_end | (protocol_start + protocol_end)

        domain_graph_with_class_tags = (
            pynutil.insert(domain_string + colon + NEMO_SPACE + double_quotes)
            + pynini.compose(
                NEMO_ALPHA + pynini.closure(NEMO_NOT_SPACE) + (NEMO_ALPHA | NEMO_DIGIT | pynini.accep(slash)),
                domain_graph,
            ).optimize()
            + pynutil.insert(double_quotes)
        )

        protocol = (
            pynutil.insert(protocol_string + colon + NEMO_SPACE + double_quotes)
            + pynutil.add_weight(protocol, MIN_NEG_WEIGHT)
            + pynutil.insert(double_quotes)
        )
        # email
        graph = pynini.compose(
            NEMO_SIGMA + pynini.accep(at) + NEMO_SIGMA + pynini.accep(period) + NEMO_SIGMA,
            username + domain_graph_with_class_tags,
        )

        # abc.com, abc.com/123-sm
        # when only domain, make sure it starts and end with NEMO_ALPHA
        graph |= (
            pynutil.insert(domain_string + colon + NEMO_SPACE + double_quotes)
            + pynini.compose(
                NEMO_ALPHA
                + pynini.closure(NEMO_NOT_SPACE)
                + accepted_common_domains
                + pynini.closure(pynini.difference(NEMO_NOT_SPACE, pynini.accep(period))),
                domain_graph,
            ).optimize()
            + pynutil.insert(double_quotes)
        )
        # www.abc.com/sdafsdf, or https://www.abc.com/asdfad or www.abc.abc/asdfad
        graph |= protocol + pynutil.insert(NEMO_SPACE) + domain_graph_with_class_tags

        if deterministic:
            # credit card cues
            numbers = pynini.closure(NEMO_DIGIT, 4, 16)
            cc_phrases = (
                pynutil.insert(protocol_string + colon + NEMO_SPACE + double_quotes)
                + cc_cues
                + pynutil.insert(double_quotes + NEMO_SPACE + domain_string + colon + NEMO_SPACE + double_quotes)
                + numbers
                + pynutil.insert(double_quotes)
            )
            graph |= cc_phrases

        final_graph = self.add_tokens(graph)

        self.fst = final_graph.optimize()
