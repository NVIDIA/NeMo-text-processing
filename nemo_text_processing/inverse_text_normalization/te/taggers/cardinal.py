import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.te.graph_utils import GraphFst
from nemo_text_processing.inverse_text_normalization.te.utils import get_abs_path


class CardinalFst(GraphFst):
    """
    Classifies spoken Telugu numbers back to digits,
    e.g. 'ఒకటి' -> cardinal { integer: "1" }
    """

    def __init__(self):
        super().__init__(name="cardinal", kind="classify")

        # Load TSV files and invert them (word -> digit)
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert()
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert()
        graph_teens_and_ties = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv")).invert()

        # Combine all graphs
        graph = graph_digit | graph_zero | graph_teens_and_ties
        graph = graph.optimize()

        # Wrap with token labels
        final_graph = pynutil.insert('integer: "') + graph + pynutil.insert('"')
        final_graph = self.add_tokens(final_graph)

        self.fst = final_graph.optimize()
