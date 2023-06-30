import pynini
from pynini.lib import pynutil
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space


class ProfaneFst(GraphFst):
    """
    Finite state transducer for verbalizing profane words
        e.g. bitch -> profane { filtered: "b****" } -> b****
    """

    def __init__(self):
        super().__init__(name="profane", kind="verbalize")
        graph = (
            pynutil.delete("filtered:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
            + delete_space
        )

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
