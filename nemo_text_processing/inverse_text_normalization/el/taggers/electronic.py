import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.el.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_LOWER_CASED,
    NEMO_ALPHA,
    NEMO_DIGIT,
    GraphFst,
    delete_space,
)


class ElectronicFst(GraphFst):
    def __init__(self, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="electronic", kind="classify")

        digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")) | pynini.string_file(
            get_abs_path("data/numbers/zero.tsv")
        )

        single_alphanum = NEMO_ALPHA | digit
        sequence = single_alphanum + pynini.closure(delete_space + single_alphanum)

        username = pynutil.insert("username: \"") + sequence + pynutil.insert("\"")

        server = sequence

        domain = pynini.string_file(get_abs_path("data/electronic/domain.tsv"))

        domain_graph = (
            pynutil.insert("domain: \"")
            + server
            + delete_space
            + pynini.cross("dot", ".")
            + delete_space
            + domain
            + pynutil.insert("\"")
        )

        graph = username + delete_space + pynutil.delete("at") + delete_space + domain_graph

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
