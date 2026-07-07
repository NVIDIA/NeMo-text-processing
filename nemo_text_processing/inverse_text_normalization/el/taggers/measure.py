import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.el.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_LOWER_CASED,
    GraphFst,
    convert_space,
    delete_extra_space,
)


class MeasureFst(GraphFst):
    def __init__(self, cardinal: GraphFst, decimal: GraphFst, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="measure", kind="classify")

        cardinal_graph = cardinal.graph_no_exception

        # measurements.tsv maps symbol -> Greek word, listing both singular and plural word forms
        # explicitly, so inverting it gives a complete word -> symbol map. (Unlike English there is
        # no need for get_singulars: Greek plurals are not formed by appending an "s".)
        graph_unit = pynini.invert(pynini.string_file(get_abs_path("data/measurements.tsv")))
        unit = convert_space(graph_unit)
        unit_wrapped = pynutil.insert("units: \"") + unit + pynutil.insert("\"")

        subgraph_decimal = (
            pynutil.insert("decimal { ")
            + decimal.final_graph_wo_negative
            + pynutil.insert(" }")
            + delete_extra_space
            + unit_wrapped
        )
        subgraph_cardinal = (
            pynutil.insert("cardinal { ")
            + pynutil.insert("integer: \"")
            + cardinal_graph
            + pynutil.insert("\"")
            + pynutil.insert(" }")
            + delete_extra_space
            + unit_wrapped
        )

        final_graph = subgraph_decimal | subgraph_cardinal
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
