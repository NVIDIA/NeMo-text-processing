import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space


class MoneyFst(GraphFst):
    def __init__(self, deterministic: bool = True):
        super().__init__(name="money", kind="verbalize", deterministic=deterministic)

        integer = (
            pynutil.delete("integer_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        fractional = (
            pynutil.delete("fractional_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        graph = integer + pynini.closure(delete_space + pynutil.insert(" ") + fractional, 0, 1)
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
