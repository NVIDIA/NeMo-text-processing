import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space


class DateFst(GraphFst):
    def __init__(self, deterministic: bool = True):
        super().__init__(name="date", kind="verbalize", deterministic=deterministic)

        day = (
            pynutil.delete("day:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        month = (
            pynutil.delete("month:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        year = (
            pynutil.delete("year:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        graph_dmy = (
            day
            + delete_space
            + pynutil.insert(" ")
            + month
            + pynini.closure(delete_space + pynutil.insert(" ") + year, 0, 1)
        )
        graph = graph_dmy | year
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
