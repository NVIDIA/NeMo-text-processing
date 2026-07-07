import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space


class OrdinalFst(GraphFst):
    def __init__(self):
        super().__init__(name="ordinal", kind="verbalize")
        graph = (
            pynutil.delete("integer:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
