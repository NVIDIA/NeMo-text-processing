import pynini
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, NEMO_DIGIT, GraphFst, delete_space
from pynini.lib import pynutil

class CardinalFst(GraphFst):
    """
    Finite state transducer for verbalizing cardinal
        e.g.
    """
    def __init__(self):
        super().__init__(name="cardinal", kind="verbalize")

        optional_sign = pynini.closure(
            pynutil.delete("negative:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.accep("-")
            + pynutil.delete("\"")
            + delete_space,
            0,
            1,
        )
        graph = (
                pynutil.delete("integer:")
                + delete_space
                + pynutil.delete("\"")
                + pynini.closure(NEMO_DIGIT,
                                 1)  # Accepts at least one digit change nemo digit to whatever is relevant
                + pynutil.delete("\"")
                + delete_space
        )
        # graph = optional_sign + graph # concatenates two properties
        graph = optional_sign + graph
        delete_tokens = self.delete_tokens(graph)  # removes semiotic class tag

        self.fst = delete_tokens.optimize()