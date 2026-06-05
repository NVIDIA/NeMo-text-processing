import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.ta.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space


class CardinalFst(GraphFst):
    """
    Verbalizes the digits, e.g.  cardinal { integer: "5" }  ->  5
    """

    def __init__(self):
        super().__init__(name="cardinal", kind="verbalize")

        # TODO 3: keep the digits between the quotes (1 or more non-quote chars).
        graph = (
            pynutil.delete("integer:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
