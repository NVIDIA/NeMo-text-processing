import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.el.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_extra_space,
    delete_space,
    insert_space,
)


class DateFst(GraphFst):
    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="date", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.graph_no_tokens
        month_map = pynini.string_file(get_abs_path("data/dates/months.tsv")).optimize()

        delete_sep = pynini.cross(pynini.union("/", "-", "."), " ")

        zero = pynutil.add_weight(pynutil.delete("0"), -0.1)
        digit_to_cardinal = pynini.compose(NEMO_DIGIT, cardinal_graph)

        day = zero + digit_to_cardinal
        day |= pynini.compose(
            (pynini.union("1", "2", "3") + NEMO_DIGIT) | NEMO_DIGIT,
            cardinal_graph,
        )
        day = pynutil.insert("day: \"") + day + pynutil.insert("\"")

        month_lookup = (pynini.accep("0") + NEMO_DIGIT) | ("1" + NEMO_DIGIT)
        month_name = pynini.compose(month_lookup, month_map)
        month_num = zero + digit_to_cardinal
        month_num |= pynini.compose(pynini.accep("1") + NEMO_DIGIT, cardinal_graph)
        # Prefer the spelled-out month name; fall back to a numeric reading only for
        # out-of-range values (13-19) that have no entry in months.tsv. The weight must
        # beat the -0.1 leading-zero bonus inside month_num so 01/03/04/07 don't tie.
        month = pynutil.insert("month: \"") + (month_name | pynutil.add_weight(month_num, 1.0)) + pynutil.insert("\"")

        year = pynini.compose(NEMO_DIGIT ** 4, cardinal_graph).optimize()
        year = pynutil.insert("year: \"") + year + pynutil.insert("\"")
        year_optional = pynini.closure(delete_sep + year, 0, 1)

        graph = (day + delete_sep + month + year_optional) | year
        self.fst = self.add_tokens(graph).optimize()
