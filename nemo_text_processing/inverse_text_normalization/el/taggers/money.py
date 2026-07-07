import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.el.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_LOWER_CASED,
    NEMO_SIGMA,
    GraphFst,
    convert_space,
    delete_extra_space,
    delete_space,
    get_singulars,
)


class MoneyFst(GraphFst):
    def __init__(self, cardinal: GraphFst, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="money", kind="classify")

        cardinal_graph = cardinal.graph_no_exception

        unit = pynini.string_file(get_abs_path("data/currency.tsv"))
        unit_singular = pynini.invert(unit)
        unit_plural = get_singulars(unit_singular)

        graph_unit_singular = pynutil.insert("currency: \"") + convert_space(unit_singular) + pynutil.insert("\"")
        graph_unit_plural = pynutil.insert("currency: \"") + convert_space(unit_plural | unit_singular) + pynutil.insert("\"")

        graph_integer = (
            pynutil.insert("integer_part: \"")
            + cardinal_graph
            + pynutil.insert("\"")
            + delete_extra_space
            + graph_unit_plural
        )

        final_graph = graph_integer
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
