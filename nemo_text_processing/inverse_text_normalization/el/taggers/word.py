import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_SPACE, GraphFst


class WordFst(GraphFst):
    def __init__(self):
        super().__init__(name="word", kind="classify")
        word = pynutil.insert("name: \"") + pynini.closure(NEMO_NOT_SPACE, 1) + pynutil.insert("\"")
        self.fst = word.optimize()
