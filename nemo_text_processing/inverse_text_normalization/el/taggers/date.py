import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.el.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_LOWER_CASED,
    NEMO_DIGIT,
    GraphFst,
    delete_extra_space,
    delete_space,
)


class DateFst(GraphFst):
    def __init__(self, cardinal: GraphFst, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="date", kind="classify")

        cardinal_graph = cardinal.graph_no_exception

        month_graph = pynini.string_file(get_abs_path("data/months.tsv"))
        month_graph = pynutil.insert("month: \"") + month_graph + pynutil.insert("\"")

        day_graph = pynutil.insert("day: \"") + cardinal_graph + pynutil.insert("\"")

        year_graph = pynutil.insert("year: \"") + cardinal_graph + pynutil.insert("\"")

        graph_dmy = (
            day_graph + delete_extra_space + month_graph + pynini.closure(delete_extra_space + year_graph, 0, 1)
        )
        graph_mdy = (
            month_graph + delete_extra_space + day_graph + pynini.closure(delete_extra_space + year_graph, 0, 1)
        )

        final_graph = graph_dmy | graph_mdy
        final_graph += pynutil.insert(" preserve_order: true")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
