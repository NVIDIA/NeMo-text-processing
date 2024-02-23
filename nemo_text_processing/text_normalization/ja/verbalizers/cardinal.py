import pynini
from nemo_text_processing.text_normalization.jp.graph_utils import (
    NEMO_DIGIT,
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_space,
)
from pynini.lib import pynutil

class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g.
    cardinal { integer: "一" } -> 一
    cardinal { negative: "-" integer: "二十三" } -> マイナス二十三
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="verbalize", deterministic=deterministic)

        optional_sign = (
            pynutil.delete("negative:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE)
            + pynutil.delete("\"")
            + delete_space
        )

        graph = (
            pynutil.delete("integer:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE)
            + pynutil.delete("\"")
        )

        final_graph = pynini.closure(optional_sign, 0, 1) + graph

        final_graph = self.delete_tokens(final_graph)
        self.fst = final_graph.optimize()