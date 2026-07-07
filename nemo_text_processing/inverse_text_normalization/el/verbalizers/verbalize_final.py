import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.el.verbalizers.verbalize import VerbalizeFst
from nemo_text_processing.inverse_text_normalization.el.verbalizers.word import WordFst
from nemo_text_processing.text_normalization.en.graph_utils import GraphFst, delete_extra_space, delete_space


class VerbalizeFinalFst(GraphFst):
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
