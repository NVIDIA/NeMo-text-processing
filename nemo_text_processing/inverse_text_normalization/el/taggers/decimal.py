import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.el.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_LOWER_CASED,
    GraphFst,
    delete_extra_space,
    delete_space,
)


class DecimalFst(GraphFst):
    def __init__(self, cardinal: GraphFst, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="decimal", kind="classify")

        cardinal_graph = cardinal.graph_no_exception

        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_digit |= pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        graph_digit_seq = pynini.closure(graph_digit + delete_space) + graph_digit
        self.graph = graph_digit

        graph_fractional = (
            pynutil.insert("fractional_part: \"") + (cardinal_graph | graph_digit_seq) + pynutil.insert("\"")
        )
        graph_integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")

        point = pynutil.delete("κόμμα") | pynutil.delete("κομμα")

        final_graph_wo_sign = (
            pynini.closure(graph_integer + delete_extra_space, 0, 1) + point + delete_extra_space + graph_fractional
        )
        self.final_graph_wo_negative = final_graph_wo_sign

        graph = graph_integer + delete_extra_space + point + delete_extra_space + graph_fractional
        graph = self.add_tokens(graph)
        self.fst = graph.optimize()
