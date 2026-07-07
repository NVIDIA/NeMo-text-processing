import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_CHAR, NEMO_SIGMA, GraphFst, delete_space


class WordFst(GraphFst):
    def __init__(self):
        super().__init__(name="word", kind="verbalize")
        chars = pynini.closure(NEMO_CHAR - " ", 1)
        char = pynutil.delete("name:") + delete_space + pynutil.delete("\"") + chars + pynutil.delete("\"")
        graph = char @ pynini.cdrewrite(pynini.cross(u"\u00a0", " "), "", "", NEMO_SIGMA)

        self.fst = graph.optimize()
