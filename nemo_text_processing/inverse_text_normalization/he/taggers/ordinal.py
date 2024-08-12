import pynini
from pynini.lib import pynutil
from nemo_text_processing.inverse_text_normalization.he.utils import get_abs_path
from nemo_text_processing.inverse_text_normalization.he.graph_utils import (NEMO_SIGMA, GraphFst)


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinal in Hebrew
        e.g. ראשון -> ordinal { integer: "1" }

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="ordinal", kind="classify")

        cardinal_graph = cardinal.graph_no_exception
        graph_digit = pynini.string_file(get_abs_path("data/ordinals/digit.tsv"))
        graph = NEMO_SIGMA + graph_digit

        self.graph = graph @ cardinal_graph

        final_graph = pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()


if __name__ == '__main__':

    from nemo_text_processing.inverse_text_normalization.he.graph_utils import apply_fst
    from nemo_text_processing.inverse_text_normalization.he.taggers.cardinal import CardinalFst

    cardinal = CardinalFst()
    g = OrdinalFst(cardinal).fst

    # To test this FST, remove comment out and change the input text
    # apply_fst('טקסט לבדיקה כאן', g)
