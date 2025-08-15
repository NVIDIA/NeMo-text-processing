import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, GraphFst
from nemo_text_processing.text_normalization.fr.utils import get_abs_path

# TODO: add articles? 'le...'

month_numbers = pynini.string_file(get_abs_path("data/dates/months.tsv"))
eras = pynini.string_file(get_abs_path("data/dates/eras.tsv"))
delete_leading_zero = (
    pynutil.delete("0") | (NEMO_DIGIT - "0")
) + NEMO_DIGIT  # reminder, NEMO_DIGIT = filter on digits


class DateFst(GraphFst):
    '''Finite state transducer for classyfing dates, e.g.:
    '02.03.2003' -> date {day: 'deux' month: 'mai' year: 'deux mille trois' preserve order: true}
    '''

    def __init__(
        self,
        cardinal: GraphFst,
        deterministic: bool = True,
        project_input: bool = False
    ):
        super().__init__(name="dates", kind="classify", project_input=project_input)

        cardinal_graph = cardinal.all_nums_no_tokens

        # 'le' -> 'le', 'les' -> 'les'
        le_determiner = pynini.accep("le ") | pynini.accep("les ")
        self.optional_le = pynini.closure(le_determiner, 0, 1)

        # '01' -> 'un'
        optional_leading_zero = delete_leading_zero | NEMO_DIGIT
        valid_day_number = pynini.union(*[str(x) for x in range(1, 32)])
        premier = pynini.string_map([("1", "premier")])
        day_number_to_word = premier | cardinal_graph

        digit_to_day = self.optional_le + optional_leading_zero @ valid_day_number @ day_number_to_word
        self.day_graph = pynutil.insert("day: \"") + digit_to_day + pynutil.insert("\"")

        # '03' -> 'mars'
        normalize_month_number = optional_leading_zero @ pynini.union(*[str(x) for x in range(1, 13)])
        number_to_month = month_numbers.optimize()
        month_graph = normalize_month_number @ number_to_month
        self.month_graph = pynutil.insert("month: \"") + month_graph + pynutil.insert("\"")

        # 2025 -> deux mille vingt cinq
        accept_year_digits = (NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT, 1, 3)
        digits_to_year = accept_year_digits @ cardinal_graph
        self.year_graph = pynutil.insert("year: \"") + digits_to_year + pynutil.insert("\"")

        # Putting it all together
        self.fst = pynini.accep("")

        for separator in ["/", ".", "-"]:
            self.fst |= (
                pynutil.insert("date { ")
                + self.day_graph
                + pynutil.delete(separator)
                + pynutil.insert(" ")
                + self.month_graph
                + pynini.closure(pynutil.delete(separator) + pynutil.insert(" ") + self.year_graph, 0, 1)
                + pynutil.insert(" preserve_order: true }")
            )

        # Accepts "janvier", "février", etc
        month_name_graph = pynutil.insert("month: \"") + month_numbers.project("output") + pynutil.insert("\"")

        self.fst |= (
            pynutil.insert("date { ")
            + self.day_graph
            + pynini.accep(" ")
            + month_name_graph
            + pynini.closure(pynini.accep(" ") + self.year_graph, 0, 1)
            + pynutil.insert(" preserve_order: true}")
        )

        # Accepts "70s", "80s", etc
        self.fst |= pynutil.insert("date { year: \"") + eras + pynutil.insert("\" preserve_order: true }")

        # Accepts date ranges, "17-18-19 juin"  -> date { day: "17" day: "18": day: "19"}
        for separator in ["-", "/"]:
            day_range_graph = (
                pynutil.insert("day: \"")
                + pynini.closure(digit_to_day + pynutil.delete(separator) + pynutil.insert(" "), 1)
                + digit_to_day
                + pynutil.insert("\"")
            )

            self.fst |= (
                pynutil.insert("date { ")
                + day_range_graph
                + pynini.accep(" ")
                + month_name_graph
                + pynini.closure(pynini.accep(" ") + self.year_graph, 0, 1)
                + pynutil.insert(" preserve_order: true }")
            )

        self.fst = self.fst.optimize()
