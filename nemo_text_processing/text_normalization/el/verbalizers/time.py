import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space


class TimeFst(GraphFst):
    def __init__(self, deterministic: bool = True):
        super().__init__(name="time", kind="verbalize", deterministic=deterministic)

        hour = (
            pynutil.delete("hours:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        minutes = (
            pynutil.delete("minutes:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        preserve_order = pynutil.delete("preserve_order:") + delete_space + pynutil.delete("true")

        graph = hour + delete_space + pynutil.insert(" ") + minutes + delete_space + preserve_order
        graph |= hour
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
