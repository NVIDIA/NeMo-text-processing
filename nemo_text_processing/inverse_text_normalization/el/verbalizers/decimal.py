import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space


class DecimalFst(GraphFst):
    def __init__(self):
        super().__init__(name="decimal", kind="verbalize")
        optionl_sign = pynini.closure(pynini.cross("negative: \"true\"", "-") + delete_space, 0, 1)
        integer = (
            pynutil.delete("integer_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        optional_integer = pynini.closure(integer + delete_space, 0, 1)
        fractional = (
            pynutil.insert(",")
            + pynutil.delete("fractional_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        optional_fractional = pynini.closure(fractional + delete_space, 0, 1)
        quantity = (
            pynutil.delete("quantity:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        optional_quantity = pynini.closure(pynutil.insert(" ") + quantity + delete_space, 0, 1)
        graph = optional_integer + optional_fractional + optional_quantity
        self.numbers = graph
        graph = optionl_sign + graph
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
