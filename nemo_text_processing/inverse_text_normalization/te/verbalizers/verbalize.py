from nemo_text_processing.inverse_text_normalization.te.graph_utils import GraphFst
from nemo_text_processing.inverse_text_normalization.te.verbalizers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.te.verbalizers.word import WordFst


class VerbalizeFst(GraphFst):
    """
    Composes the Telugu cardinal verbalizer grammar.
    """

    def __init__(self):
        super().__init__(name="verbalize", kind="verbalize")

        cardinal_graph = CardinalFst().fst
        word_graph = WordFst().fst
        self.fst = (cardinal_graph | word_graph).optimize()