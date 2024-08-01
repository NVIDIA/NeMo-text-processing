import pynini
from pynini.lib import pynutil
from nemo_text_processing.inverse_text_normalization.he.utils import get_abs_path
from nemo_text_processing.inverse_text_normalization.he.graph_utils import (
    MINUS,
    NEMO_DIGIT,
    GraphFst,
    delete_extra_space,
    delete_space,
    delete_and,
    delete_zero_or_one_space
)


def get_quantity(decimal: 'pynini.FstLike', cardinal_up_to_hundred: 'pynini.FstLike') -> 'pynini.FstLike':
    """
    Returns FST that transforms either a cardinal or decimal followed by a quantity into a numeral in Hebrew,
    e.g. one million -> integer_part: "1" quantity: "million"
    e.g. one point five million -> integer_part: "1" fractional_part: "5" quantity: "million"

    Args: 
        decimal: decimal FST
        cardinal_up_to_hundred: cardinal FST
    """
    numbers = cardinal_up_to_hundred @ (
        pynutil.delete(pynini.closure("0")) + pynini.difference(NEMO_DIGIT, "0") + pynini.closure(NEMO_DIGIT)
    )

    suffix_labels = ["אלף", "מיליון", "מיליארד"]
    suffix_labels = [x for x in suffix_labels if x != "אלף"]
    suffix = pynini.union(*suffix_labels).optimize()

    res = (
        pynutil.insert("integer_part: \"")
        + numbers
        + pynutil.insert("\"")
        + delete_extra_space
        + pynutil.insert("quantity: \"")
        + suffix
        + pynutil.insert("\"")
    )
    res |= decimal + delete_extra_space + pynutil.insert("quantity: \"") + (suffix | "אלף") + pynutil.insert("\"")

    return res


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal in Hebrew
        e.g. עשרים ושלוש וחצי -> decimal { integer_part: "23" fractional_part: "5" }
        e.g. אחד נקודה שלוש -> decimal { integer_part: "1"  fractional_part: "3" }
        e.g. ארבע נקודה חמש מיליון -> decimal { integer_part: "4"  fractional_part: "5" quantity: "מיליון" }
        e.g. מינוס ארבע מאות נקודה שלוש שתיים שלוש -> decimal { negative: "true" integer_part: "400"  fractional_part: "323" }
        e.g. אפס נקודה שלושים ושלוש -> decimal { integer_part: "0"  fractional_part: "33" }
    Args:
        cardinal: CardinalFst

    TODO: add and a half, and a quarter only for negative numbers and numbers with quantity
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="decimal", kind="classify")

        cardinal_graph = cardinal.graph_no_exception

        fractions = delete_zero_or_one_space + delete_and + pynini.string_file(get_abs_path("data/numbers/decimal_fractions.tsv"))

        graph_decimal = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_decimal |= pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        graph_decimal |= cardinal.graph_two_digit

        graph_decimal = pynini.closure(graph_decimal + delete_space) + graph_decimal
        self.graph = graph_decimal

        point = pynutil.delete("נקודה")

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross(MINUS, "\"true\"") + delete_extra_space, 0, 1,
        )

        graph_integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        graph_fractional = pynutil.insert("fractional_part: \"") + graph_decimal + pynutil.insert("\"")

        fractions_graph = pynutil.insert("fractional_part: \"") + fractions + pynutil.insert("\"")
        graph_wo_point = graph_integer + delete_extra_space + fractions_graph
        graph_w_point = (
            pynini.closure(graph_integer + delete_extra_space, 0, 1) + point + delete_extra_space + graph_fractional
        )

        self.final_graph_wo_sign = graph_w_point | graph_wo_point
        final_graph = optional_graph_negative + self.final_graph_wo_sign

        self.final_graph_wo_negative = self.final_graph_wo_sign | get_quantity(self.final_graph_wo_sign, cardinal.graph_hundred)

        quantity_graph = get_quantity(self.final_graph_wo_sign, cardinal.graph_hundred)
        final_graph |= optional_graph_negative + quantity_graph

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()


if __name__ == '__main__':
    from nemo_text_processing.inverse_text_normalization.he.graph_utils import apply_fst
    from nemo_text_processing.inverse_text_normalization.he.taggers.cardinal import CardinalFst
    cardinal = CardinalFst()
    graph = DecimalFst(cardinal).fst
    # apply_fst("עשרים ושלוש וחצי", graph)
    # apply_fst("אחד נקודה שלוש", graph)
    # apply_fst("ארבע נקודה חמש מיליון", graph)
    # apply_fst("מינוס ארבע מאות נקודה שלוש שתיים שלוש", graph)
    # apply_fst("אפס נקודה שלושים ושלוש", graph)
