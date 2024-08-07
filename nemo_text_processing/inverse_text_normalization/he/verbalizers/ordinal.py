import pynini
from pynini.lib import pynutil
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, GraphFst, delete_space


class OrdinalFst(GraphFst):
    """
    Finite state transducer for verbalizing ordinal in Hebrew
        e.g. ordinal { integer: "10" } -> 10
    """

    def __init__(self):
        super().__init__(name="ordinal", kind="verbalize")
        graph = (
            pynutil.delete("integer:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_DIGIT, 1)
            + pynutil.delete("\"")
        )
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()


if __name__ == "__main__":

    from nemo_text_processing.inverse_text_normalization.he.graph_utils import apply_fst

    cardinal = OrdinalFst().fst

    # To test this FST, remove comment out and change the input text
    # apply_fst("טקסט לבדיקה", g)
