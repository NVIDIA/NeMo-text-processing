import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_extra_space,
    delete_space,
)


class DateFst(GraphFst):
    def __init__(self):
        super().__init__(name="date", kind="verbalize")
        month = (
            pynutil.delete("month:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        day = (
            pynutil.delete("day:")
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
        optional_preserve_order = pynini.closure(
            pynutil.delete("preserve_order:") + delete_space + pynutil.delete("true") + delete_space
        )

        graph_mdy = month + delete_extra_space + day + pynini.closure(delete_extra_space + year, 0, 1)
        graph_dmy = day + delete_extra_space + month + pynini.closure(delete_extra_space + year, 0, 1)

        final_graph = (graph_mdy | graph_dmy) + delete_space + optional_preserve_order
        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()
