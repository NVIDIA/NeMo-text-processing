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
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_SIGMA,
    GraphFst,
    get_abs_path,
    insert_space,
)
from pynini.lib import pynutil
from nemo_text_processing.text_normalization.en.utils import load_labels


class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying electronic: as URLs, email addresses, etc.
        e.g. cdf1@abc.edu -> tokens { electronic { username: "cdf1" domain: "abc.edu" } }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="classify", deterministic=deterministic)

        accepted_symbols = pynini.project(pynini.string_file(get_abs_path("data/electronic/symbol.tsv")), "input")
        symbols_graph = pynini.string_file(get_abs_path("data/electronic/symbol.tsv"))
        accepted_common_domains = pynini.project(
            pynini.string_file(get_abs_path("data/electronic/domain.tsv")), "input"
        )
        dict_words = [x[0] for x in load_labels(get_abs_path("data/electronic/words.tsv"))]
        dict_words = pynini.union(*dict_words).optimize()

        # X"-services" -> "dash services"
        dict_words_with_delimiter = (accepted_symbols + dict_words).optimize()
        # X"services" -> " services"
        dict_words_without_delimiter = (pynutil.insert(" ") + dict_words).optimize()

        all_accepted_symbols_end = (NEMO_ALPHA | NEMO_DIGIT | accepted_symbols | (dict_words_with_delimiter | dict_words_without_delimiter).optimize()).optimize()
        all_accepted_symbols_graph = (NEMO_ALPHA | dict_words).optimize() + pynini.closure(all_accepted_symbols_end)

        # from pynini.lib.rewrite import top_rewrite
        # import pdb; pdb.set_trace()
        graph_symbols = pynini.string_file(get_abs_path("data/electronic/symbol.tsv")).optimize()

        username = (
            pynutil.insert("username: \"") + all_accepted_symbols_graph + pynutil.insert("\"") + pynini.cross('@', ' ')
        )

        domain_graph = all_accepted_symbols_graph + pynini.accep('.') + (all_accepted_symbols_graph + NEMO_ALPHA | dict_words.optimize()) + pynini.closure(all_accepted_symbols_end)

        protocol_symbols = pynini.closure((graph_symbols | pynini.cross(":", "colon")) + pynutil.insert(" "))
        protocol_start = (pynini.cross("https", "HTTPS ") | pynini.cross("http", "HTTP ")) + (
            pynini.accep("://") @ protocol_symbols
        )
        protocol_file_start = pynini.accep("file") + insert_space + (pynini.accep(":///") @ protocol_symbols)

        protocol_end = pynini.cross("www", "WWW ") + pynini.accep(".") @ protocol_symbols
        protocol = protocol_file_start | protocol_start | protocol_end | (protocol_start + protocol_end)

        # domain_graph = (
        #     pynutil.insert("domain: \"")
        #     + pynini.difference(domain_graph, pynini.project(protocol, "input") + NEMO_SIGMA)
        #     + pynini.closure((pynini.cross("-", " dash ") | pynutil.insert(" ")) + dict_words)
        #     + pynutil.insert("\"")
        # )
        # domain_common_graph = (
        #     pynutil.insert("domain: \"")
        #     + pynini.difference(
        #         all_accepted_symbols
        #         + accepted_common_domains
        #         + pynini.closure(accepted_symbols + pynini.closure(NEMO_ALPHA | NEMO_DIGIT | accepted_symbols), 0, 1),
        #         pynini.project(protocol, "input") + NEMO_SIGMA,
        #     )
        #     + pynutil.insert("\"")
        # )

        from pynini.lib.rewrite import top_rewrite
        import pdb; pdb.set_trace()

        domain_graph = (
            pynutil.insert("domain: \"")
            + domain_graph
            + pynini.closure((pynini.cross("-", " dash ") | pynutil.insert(" ")) + dict_words)
            + pynutil.insert("\"")
        )
        # domain_common_graph = (
        #     pynutil.insert("domain: \"")
        #     + pynini.difference(
        #         all_accepted_symbols
        #         + accepted_common_domains
        #         + pynini.closure(accepted_symbols + pynini.closure(NEMO_ALPHA | NEMO_DIGIT | accepted_symbols), 0, 1),
        #         pynini.project(protocol, "input") + NEMO_SIGMA,
        #     )
        #     + pynutil.insert("\"")
        # )

        protocol = pynutil.insert("protocol: \"") + protocol + pynutil.insert("\"")
        # email
        graph = username + domain_graph
        # abc.com, abc.com/123-sm
        graph |= domain_graph
        # www.abc.com/sdafsdf, or https://www.abc.com/asdfad or www.abc.abc/asdfad
        graph |= protocol + pynutil.insert(" ") + domain_graph

        final_graph = self.add_tokens(graph)

        self.fst = final_graph.optimize()
