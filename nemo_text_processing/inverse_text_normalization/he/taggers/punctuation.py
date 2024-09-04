import pynini
from pynini.lib import pynutil
from nemo_text_processing.inverse_text_normalization.he.graph_utils import GraphFst


class PunctuationFst(GraphFst):
    """
    Finite state transducer for classifying punctuation
        e.g. a, -> tokens { name: "a" } tokens { name: "," }
    """

    def __init__(self):
        super().__init__(name="punctuation", kind="classify")

        s = "!#$%&\'()*+,-./:;<=>?@^_`{|}~"
        punct = pynini.union(*s)

        graph = pynutil.insert("name: \"") + punct + pynutil.insert("\"")

        self.fst = graph.optimize()
