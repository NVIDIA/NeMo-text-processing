import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space


class ElectronicFst(GraphFst):
    def __init__(self):
        super().__init__(name="electronic", kind="verbalize")
        user_name = (
            pynutil.delete("username:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        domain = (
            pynutil.delete("domain:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        graph = user_name + delete_space + pynutil.insert("@") + domain
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
