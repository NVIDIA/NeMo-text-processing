import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_CHAR, NEMO_SIGMA, GraphFst, delete_space


class WhiteListFst(GraphFst):
    def __init__(self):
        super().__init__(name="whitelist", kind="verbalize")
        graph = (
            pynutil.delete("name:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_CHAR - " ", 1)
            + pynutil.delete("\"")
        )
        graph = graph @ pynini.cdrewrite(pynini.cross(u"\u00a0", " "), "", "", NEMO_SIGMA)
        self.fst = graph.optimize()
