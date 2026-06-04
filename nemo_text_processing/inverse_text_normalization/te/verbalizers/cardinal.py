import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.te.graph_utils import (
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_space,
)


class CardinalFst(GraphFst):
    """
    Verbalizes Telugu digits,
    e.g. cardinal { integer: "5" } -> 5
    """

    def __init__(self):
        super().__init__(name="cardinal", kind="verbalize")

        # Keep digits between quotes (1+ non-quote chars)
        graph = (
            pynutil.delete("integer:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
