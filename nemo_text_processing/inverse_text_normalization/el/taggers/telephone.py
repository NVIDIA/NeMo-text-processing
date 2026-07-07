import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.el.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_LOWER_CASED,
    NEMO_DIGIT,
    GraphFst,
    delete_space,
)


class TelephoneFst(GraphFst):
    def __init__(self, cardinal: GraphFst, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="telephone", kind="classify")

        digit_to_str = pynini.invert(
            pynini.string_file(get_abs_path("data/numbers/digit.tsv")).optimize()
        )
        zero = pynini.invert(pynini.string_file(get_abs_path("data/numbers/zero.tsv")).optimize())
        digit_to_str = digit_to_str | zero
        str_to_digit = pynini.invert(digit_to_str)

        number_part = pynini.compose(
            str_to_digit + pynini.closure(delete_space + str_to_digit),
            NEMO_DIGIT ** 10 + pynini.closure(NEMO_DIGIT)
        )
        number_part = pynutil.insert("number_part: \"") + number_part + pynutil.insert("\"")

        final_graph = self.add_tokens(number_part)
        self.fst = final_graph.optimize()
