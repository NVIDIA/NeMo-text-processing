import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.el.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_CASED,
    INPUT_LOWER_CASED,
    GraphFst,
    capitalized_input_graph,
    delete_space,
)


class WhiteListFst(GraphFst):
    def __init__(self, input_case: str = INPUT_LOWER_CASED, deterministic: bool = True, input_file: str = None):
        super().__init__(name="whitelist", kind="classify", deterministic=deterministic)
        graph = pynutil.insert("name: \"")
        if input_file:
            whitelist = pynini.string_file(input_file).invert()
        else:
            whitelist = pynini.string_file(get_abs_path("data/whitelist.tsv")).invert()
        if input_case == INPUT_CASED:
            whitelist |= capitalized_input_graph(whitelist)
        graph += whitelist
        graph += pynutil.insert("\"")
        self.fst = graph.optimize()
