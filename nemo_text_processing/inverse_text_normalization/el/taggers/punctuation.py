import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import GraphFst


class PunctuationFst(GraphFst):
    def __init__(self):
        super().__init__(name="punctuation", kind="classify")
        s = "!#$%&\'()*+,-./:;<=>?@^_`{|}~"
        greek = "·«»…–—΄"
        punct = pynini.union(*s) | pynini.union(*greek)
        graph = pynutil.insert("name: \"") + punct + pynutil.insert("\"")
        self.fst = graph.optimize()
