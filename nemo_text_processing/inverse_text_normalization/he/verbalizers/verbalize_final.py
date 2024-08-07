import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.he.verbalizers.word import WordFst
from nemo_text_processing.inverse_text_normalization.he.verbalizers.verbalize import VerbalizeFst
from nemo_text_processing.inverse_text_normalization.he.graph_utils import GraphFst, delete_extra_space, delete_space


class VerbalizeFinalFst(GraphFst):
    """
    Finite state transducer that verbalizes an entire sentence in Hebrew
    """

    def __init__(self):
        super().__init__(name="verbalize_final", kind="verbalize")
        verbalize = VerbalizeFst().fst
        word = WordFst().fst
        types = verbalize | word
        graph = (
            pynutil.delete("tokens")
            + delete_space
            + pynutil.delete("{")
            + delete_space
            + types
            + delete_space
            + pynutil.delete("}")
        )
        graph = delete_space + pynini.closure(graph + delete_extra_space) + graph + delete_space
        self.fst = graph
