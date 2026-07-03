from nemo_text_processing.inverse_text_normalization.te.graph_utils import GraphFst
from nemo_text_processing.inverse_text_normalization.te.verbalizers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.te.verbalizers.fraction import FractionFst
from nemo_text_processing.inverse_text_normalization.te.verbalizers.ordinal import OrdinalFst
from nemo_text_processing.inverse_text_normalization.te.verbalizers.word import WordFst
from nemo_text_processing.inverse_text_normalization.te.verbalizers.decimal import DecimalFst


class VerbalizeFst(GraphFst):
    """
    Composes Telugu verbalizer grammars.
    Supports Fraction, Ordinal, Cardinal and Word semiotic classes.
    """

    def __init__(self):
        super().__init__(name="verbalize", kind="verbalize")

        fraction_graph = FractionFst().fst
        decimal_graph = DecimalFst().fst
        cardinal_graph = CardinalFst().fst
        ordinal_graph = OrdinalFst().fst
        word_graph = WordFst().fst

        graph = (
            decimal_graph
            | fraction_graph
            | ordinal_graph
            | cardinal_graph
            | word_graph
        )

        self.fst = graph.optimize()