import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.el.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_CASED,
    INPUT_LOWER_CASED,
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    capitalized_input_graph,
    delete_space,
)


class CardinalFst(GraphFst):
    def __init__(self, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="cardinal", kind="classify")

        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))
        graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv"))

        self.graph_two_digit = graph_teen | (
            (graph_ties | pynutil.insert("0")) + delete_space + (graph_digit | pynutil.insert("0"))
        )

        graph_hundred = pynini.string_file(get_abs_path("data/numbers/hundred.tsv"))
        graph_hundred_digit = pynini.union(graph_hundred, pynutil.insert("0"))
        graph_hundred_digit += delete_space
        graph_hundred_digit += self.graph_two_digit

        def no_all_zeros(fst):
            return fst @ (pynini.closure(NEMO_DIGIT) + (NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT))

        graph_hundred_digit_nonzero = no_all_zeros(graph_hundred_digit)

        graph_thousands = pynini.union(
            pynini.cross("χίλια", "001"),
            pynini.union(
                graph_hundred_digit_nonzero + delete_space + pynutil.delete("χιλιάδες"),
                pynutil.insert("000", weight=0.1),
            ),
        )

        graph_million = pynini.union(
            graph_hundred_digit_nonzero + delete_space + pynutil.delete("εκατομμύριο"),
            graph_hundred_digit_nonzero + delete_space + pynutil.delete("εκατομμύρια"),
            pynutil.insert("000", weight=0.1),
        )

        graph_billion = pynini.union(
            graph_hundred_digit_nonzero + delete_space + pynutil.delete("δισεκατομμύριο"),
            graph_hundred_digit_nonzero + delete_space + pynutil.delete("δισεκατομμύρια"),
            pynutil.insert("000", weight=0.1),
        )

        graph_int = graph_billion + delete_space + graph_million + delete_space + graph_thousands
        graph = (graph_int + delete_space + graph_hundred_digit) | graph_zero
        graph = graph @ pynini.union(
            pynutil.delete(pynini.closure("0")) + pynini.difference(NEMO_DIGIT, "0") + pynini.closure(NEMO_DIGIT),
            "0",
        )

        self.graph_no_exception = graph.optimize()

        labels_exception = [
            "μηδέν",
            "ένα",
            "μία",
            "μια",
            "ένας",
            "δύο",
            "δυο",
            "τρία",
            "τρεις",
            "τέσσερα",
            "τέσσερις",
            "πέντε",
            "έξι",
            "εφτά",
            "επτά",
            "οχτώ",
            "οκτώ",
            "εννιά",
            "εννέα",
            "δέκα",
            "έντεκα",
            "ένδεκα",
            "δώδεκα",
        ]
        if input_case == INPUT_CASED:
            labels_exception += [x.capitalize() for x in labels_exception]
        graph_exception = pynini.union(*labels_exception).optimize()

        self.graph = (pynini.project(graph, "input") - graph_exception.arcsort()) @ graph
        self.graph = self.graph.optimize()

        if input_case == INPUT_CASED:
            self.graph = capitalized_input_graph(self.graph)

        final_graph = pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
