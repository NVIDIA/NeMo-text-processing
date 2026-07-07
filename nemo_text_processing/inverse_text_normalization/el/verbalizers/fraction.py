import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space


class FractionFst(GraphFst):
    def __init__(self):
        super().__init__(name="fraction", kind="verbalize")

        numerator = pynutil.delete("numerator: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        denominator = (
            pynutil.delete("denominator: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        )

        graph = numerator + delete_space + pynutil.insert("/") + denominator
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
