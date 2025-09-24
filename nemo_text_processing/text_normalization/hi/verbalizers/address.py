import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.hi.graph_utils import GraphFst, NEMO_NOT_QUOTE

class AddressFst(GraphFst):
    """
    Finite state transducer for verbalizing address, e.g.
    address { number_part: "एक दो तीन" } -> एक दो तीन
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="address", kind="verbalize", deterministic=deterministic)

        number_part = (
            pynutil.delete('number_part: "')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        graph = number_part
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()