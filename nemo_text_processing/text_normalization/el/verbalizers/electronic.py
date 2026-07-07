import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space


class ElectronicFst(GraphFst):
    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="verbalize", deterministic=deterministic)

        graph = (
            pynutil.delete("username:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
