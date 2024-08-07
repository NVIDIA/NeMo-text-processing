import pynini
from pynini.lib import pynutil
from nemo_text_processing.inverse_text_normalization.he.utils import get_abs_path
from nemo_text_processing.inverse_text_normalization.he.graph_utils import (
    GraphFst,
    delete_extra_space,
    delete_space,
    insert_space,
)


def _get_month_name_graph():
    """
    Transducer for month, e.g. march -> march
    """
    month_graph = pynini.string_file(get_abs_path("data/months.tsv"))
    return month_graph


def _get_year_graph(graph_two_digits, graph_thousands):
    """
    Transducer for year, e.g. twenty twenty -> 2020
    """
    graph_bc = pynini.string_file(get_abs_path("data/year_suffix.tsv")).invert()
    optional_graph_bc = pynini.closure(delete_space + insert_space + graph_bc, 0, 1)
    year_graph = pynini.union(
        (graph_two_digits + delete_space + graph_two_digits),  # 20 19, 40 12, 20 20
        graph_thousands)  # 2012 - assuming no limit on the year
    year_graph += optional_graph_bc  # optional graph for BC
    year_graph.optimize()
    return year_graph


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date in Hebrew,
        e.g. אחד במאי אלף תשע מאות שמונים ושלוש -> date { day: "1" month_prefix: "ב" month: "5" year: "1983" }
        e.g. הראשון ביוני אלפיים ושתיים עשרה -> date { day_prefix: "ה" day: "1" month_prefix: "ב" month: "6" year: "2012" }
        e.g. העשירי ביוני -> date { day_prefix: "ה" day: "10" month_prefix: "ב" month: "6" }
        e.g. מרץ אלף תשע מאות שמונים ותשע -> date { month: "מרץ" year: "1989" }
        e.g. בינואר עשרים עשרים -> date { month_prefix: "ב" month: "ינואר" year: "2020" }

    Args:
        cardinal: CardinalFst
        ordinal: OrdinalFst
    """

    def __init__(self, cardinal: GraphFst, ordinal: GraphFst):
        super().__init__(name="date", kind="classify")

        prefix_graph = pynini.string_file(get_abs_path("data/prefix.tsv"))

        ordinal_graph = ordinal.graph
        two_digits_graph = cardinal.graph_two_digit

        day_graph = pynutil.add_weight(two_digits_graph | ordinal_graph, -0.7)
        day_graph = pynutil.insert("day: \"") + day_graph + pynutil.insert("\"")
        optional_day_prefix_graph = pynini.closure(
            pynutil.insert("day_prefix: \"") + prefix_graph + pynutil.insert("\"") + insert_space, 0, 1
        )

        month_names = _get_month_name_graph()
        all_month_graph = pynini.string_file(get_abs_path("data/months_all.tsv"))
        all_month_graph = pynini.invert(all_month_graph)
        all_month_graph = pynutil.insert("month: \"") + all_month_graph + pynutil.insert("\"")
        month_names_graph = pynutil.insert("month: \"") + month_names + pynutil.insert("\"")
        month_prefix_graph = pynutil.insert("month_prefix: \"") + prefix_graph + pynutil.insert("\"") + insert_space
        optional_month_prefix_graph = pynini.closure(month_prefix_graph, 0, 1)

        year_graph = _get_year_graph(two_digits_graph, cardinal.graph_thousands)
        year_graph = pynutil.add_weight(year_graph, 0.001)

        graph_year = (
            delete_extra_space
            + pynutil.insert("year: \"")
            + pynutil.add_weight(year_graph, -0.001)
            + pynutil.insert("\"")
        )

        optional_graph_year = pynini.closure(graph_year, 0, 1,)

        graph_dmy = (
            optional_day_prefix_graph
            + day_graph
            + insert_space
            + delete_space
            + month_prefix_graph
            + all_month_graph
            + optional_graph_year
        )

        graph_my = (
            optional_month_prefix_graph
            + month_names_graph
            + graph_year
        )

        final_graph = graph_dmy | graph_my
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()


if __name__ == '__main__':
    from nemo_text_processing.inverse_text_normalization.he.graph_utils import apply_fst
    from nemo_text_processing.inverse_text_normalization.he.taggers.cardinal import CardinalFst
    from nemo_text_processing.inverse_text_normalization.he.taggers.ordinal import OrdinalFst
    c = CardinalFst()
    g = DateFst(c, OrdinalFst(c)).fst
    # apply_fst("אחד במאי אלף תשע מאות שמונים ושלוש", g)
    # apply_fst("הראשון ביוני אלפיים ושתיים עשרה", g)
    # apply_fst("העשירי ביוני", g)
    # apply_fst("מרץ אלף תשע מאות שמונים ותשע", g)
    # apply_fst("שלושים למאי תשעים ותשע", g)
    # apply_fst("שבעים לפני הספירה", g)
    # apply_fst("עשרים עשרים ושתיים לפני הספירה", g)
    # apply_fst("השלוש עשרה בינואר עשרים עשרים", g)
    # apply_fst("בשני לפברואר עשרים עשרים ואחת", g)
    # apply_fst("במרץ אלפיים עשרים ושתיים", g)
    # apply_fst("בשלישי לשלישי אלפיים עשרים ושתיים", g)
    # apply_fst("בינואר עשרים עשרים", g)
    # apply_fst("בשנת תשעים ושמונה", g)
