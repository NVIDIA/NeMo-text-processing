import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_LOWER_CASED,
    GraphFst,
    delete_extra_space,
)


class FractionFst(GraphFst):
    def __init__(self, cardinal: GraphFst, ordinal: GraphFst, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="fraction", kind="classify")

        cardinal_graph = cardinal.graph_no_exception
        ordinal_graph = ordinal.graph

        graph_numerator = pynutil.insert("numerator: \"") + cardinal_graph + pynutil.insert("\"")
        graph_denominator = pynutil.insert("denominator: \"") + ordinal_graph + pynutil.insert("\"")

        graph = graph_numerator + delete_extra_space + graph_denominator
        graph = self.add_tokens(graph)
        self.fst = graph.optimize()
