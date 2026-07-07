import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.el.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_CASED,
    INPUT_LOWER_CASED,
    NEMO_DIGIT,
    GraphFst,
    capitalized_input_graph,
    delete_space,
)


class OrdinalFst(GraphFst):
    def __init__(self, cardinal: GraphFst, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="ordinal", kind="classify")

        graph_digit = pynini.string_file(get_abs_path("data/ordinals/digit.tsv"))
        graph_teens = pynini.string_file(get_abs_path("data/ordinals/teen.tsv"))
        graph_tens = pynini.string_file(get_abs_path("data/ordinals/tens.tsv"))

        simple = graph_digit | graph_teens

        tens_digit = pynini.compose(graph_tens, NEMO_DIGIT)
        digit_digit = pynini.compose(graph_digit, NEMO_DIGIT)
        tens_standalone = graph_tens + pynutil.insert("0")
        compound_tens_digit = tens_digit + pynutil.delete(" ") + digit_digit

        compound = tens_standalone | compound_tens_digit

        self.graph = simple | compound

        if input_case == INPUT_CASED:
            self.graph = capitalized_input_graph(self.graph)

        final_graph = pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
