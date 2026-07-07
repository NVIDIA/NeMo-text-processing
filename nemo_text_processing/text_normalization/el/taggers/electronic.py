import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_space,
    insert_space,
)


class ElectronicFst(GraphFst):
    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="classify", deterministic=deterministic)

        accepted_symbols = pynini.union("-", "_", "+", "~", ".")

        username = (
            pynutil.insert("username: \"")
            + (NEMO_ALPHA | NEMO_DIGIT | accepted_symbols)
            + pynini.closure(NEMO_ALPHA | NEMO_DIGIT | accepted_symbols)
            + pynutil.insert("\"")
            + pynutil.delete("@")
        )

        domain_graph = (
            (NEMO_ALPHA | NEMO_DIGIT)
            + pynini.closure(NEMO_ALPHA | NEMO_DIGIT | pynini.accep("-") | pynini.accep("."))
            + (NEMO_ALPHA | NEMO_DIGIT)
        )
        domain_graph = pynutil.insert("domain: \"") + domain_graph + pynutil.insert("\"")
        tagger_graph = (username + domain_graph).optimize()

        user_name = (
            pynutil.delete("username:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(
                pynutil.add_weight(NEMO_NOT_QUOTE + insert_space, 1.1)
            )
            + pynutil.delete("\"")
        )

        domain_default = (
            pynini.closure(NEMO_NOT_QUOTE + insert_space)
            + pynini.cross(".", "τελεία ")
            + NEMO_NOT_QUOTE
            + pynini.closure(insert_space + NEMO_NOT_QUOTE)
        )

        server_default = (
            pynini.closure((NEMO_ALPHA | NEMO_DIGIT) + insert_space, 1)
        )

        domain = (
            pynutil.delete("domain:")
            + delete_space
            + pynutil.delete("\"")
            + (pynutil.add_weight(server_default, 1.1))
            + (pynutil.add_weight(domain_default, 1.1))
            + delete_space
            + pynutil.delete("\"")
        )

        graph = user_name + delete_space + pynutil.insert("στο ") + delete_space + domain + delete_space
        verbalizer_graph = graph.optimize()

        self.final_graph = (tagger_graph @ verbalizer_graph).optimize()
        self.fst = pynutil.insert("username: \"") + self.final_graph + pynutil.insert("\"")
        self.fst = self.add_tokens(self.fst).optimize()
