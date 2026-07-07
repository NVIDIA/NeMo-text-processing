from nemo_text_processing.inverse_text_normalization.te.graph_utils import GraphFst
from nemo_text_processing.inverse_text_normalization.te.verbalizers.cardinal import CardinalFst


class VerbalizeFst(GraphFst):
    """
    Composes the Telugu cardinal verbalizer grammar.
    """

    def __init__(self):
        super().__init__(name="verbalize", kind="verbalize")

        cardinal_graph = CardinalFst().fst
        self.fst = cardinal_graph.optimize()