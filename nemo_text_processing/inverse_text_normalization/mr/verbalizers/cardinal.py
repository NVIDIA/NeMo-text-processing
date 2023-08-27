import pynini
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space
from pynini.lib import pynutil

class CardinalFst(GraphFst):
    """
    Finite state transducer for verbalizing cardinal
        e.g.
    """

    def __init__(self):
        super().__init__(name="cardinal", kind="verbalize")
        graph = (
                pynutil.delete("integer:")
                + delete_space
                + pynutil.delete("\"")
                + pynini.closure(NEMO_DIGIT, 1)  # Accepts at least one digit
                + pynutil.delete("\"")
        )

        delete_tokens = self.delete_tokens(graph)  # removes semiotic class tag
        self.fst = delete_tokens.optimize()
