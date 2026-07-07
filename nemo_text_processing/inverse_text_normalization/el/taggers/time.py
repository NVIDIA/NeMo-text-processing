import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import INPUT_LOWER_CASED, GraphFst, delete_extra_space


class TimeFst(GraphFst):
    def __init__(self, cardinal: GraphFst, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="time", kind="classify")

        cardinal_graph = cardinal.graph_no_exception

        hour_restriction = pynini.union(*[str(i) for i in range(24)])

        graph_hour_limited = pynini.compose(cardinal_graph, hour_restriction)
        graph_hour = pynutil.insert("hours: \"") + graph_hour_limited + pynutil.insert("\"")

        minute_restriction = pynini.union(*[str(i) for i in range(60)])

        graph_minutes_limited = pynini.compose(cardinal_graph, minute_restriction)
        graph_minutes = pynutil.insert("minutes: \"") + graph_minutes_limited + pynutil.insert("\"")

        graph = graph_hour + delete_extra_space + graph_minutes
        graph = self.add_tokens(graph)
        self.fst = graph.optimize()
