import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.te.graph_utils import (
    NEMO_NOT_QUOTE,
    GraphFst,
)


class WordFst(GraphFst):
    def __init__(self):
        super().__init__(name="word", kind="verbalize")

        word = (
            pynutil.delete('name: "')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        self.fst = word.optimize()