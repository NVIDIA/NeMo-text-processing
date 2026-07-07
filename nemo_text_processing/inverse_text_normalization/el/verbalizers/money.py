import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_CHAR, GraphFst, delete_space


class MoneyFst(GraphFst):
    def __init__(self, decimal: GraphFst):
        super().__init__(name="money", kind="verbalize")
        unit = (
            pynutil.delete("currency:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_CHAR - " ", 1)
            + pynutil.delete("\"")
        )
        graph = unit + delete_space + decimal.numbers
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
