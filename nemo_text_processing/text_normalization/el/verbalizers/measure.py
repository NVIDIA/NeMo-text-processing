import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space


class MeasureFst(GraphFst):
    def __init__(self, deterministic: bool = True):
        super().__init__(name="measure", kind="verbalize", deterministic=deterministic)

        integer = (
            pynutil.delete("integer:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        graph = integer + delete_space

        preserve_order = pynutil.delete("preserve_order:") + delete_space + pynutil.delete("true") + delete_space

        graph_cardinal = (
            pynutil.delete("cardinal {")
            + delete_space
            + pynutil.delete("integer: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
            + delete_space
            + pynutil.delete("}")
        )

        units = pynutil.delete(pynini.union("units: \"address\" ", "units: \"math\" "))

        graph |= (
            units
            + graph_cardinal
            + delete_space
            + pynini.closure(preserve_order)
        )

        address_flat = (
            pynutil.delete("units: \"address\" ")
            + delete_space
            + integer
            + delete_space
            + pynini.closure(preserve_order)
        )
        graph |= address_flat

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
