import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.ta.graph_utils import GraphFst
from nemo_text_processing.inverse_text_normalization.ta.utils import get_abs_path


class CardinalFst(GraphFst):
    """
    Classifies spoken numbers back to digits, e.g.  <word>  ->  cardinal { integer: "5" }
    """

    def __init__(self):
        super().__init__(name="cardinal", kind="classify")

        # The SAME data files (number -> word). For ITN we read them BACKWARDS
        # (word -> number) using .invert().
        # TODO 1: add .invert() to each of the three lines below.
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert()
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert()
        graph_teens_and_ties = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv")).invert()

        # TODO 2: Combine them with the union operator  |
        graph = graph_digit | graph_zero | graph_teens_and_ties
        graph = graph.optimize()

        final_graph = pynutil.insert('integer: "') + graph + pynutil.insert('"')
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
