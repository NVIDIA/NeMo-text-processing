import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.he.graph_utils import (
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_space,
    NEMO_CHAR,
)


class MeasureFst(GraphFst):
    """
    Finite state transducer for verbalizing measure, in Hebrew.
    Some measures are concatenated to the numbers and other are don't (two measure lists)
        e.g. measure { cardinal { integer: "3" } spaced_units: "מ״ג" } -> 3 מ״ג
        e.g. measure { cardinal { integer: "1000" } units: "%" } -> 1,000%
        e.g. measure { units: "%" cardinal { integer: "1" } } -> 1%
        e.g. measure { spaced_units: "ס״מ" cardinal { integer: "1" } } -> 1 ס״מ
        e.g. measure { prefix: "ל" cardinal { integer: "4" } spaced_units: "ס״מ" } -> ל-4 ס״מ

    Args:
        decimal: DecimalFst
        cardinal: CardinalFst
    """

    def __init__(self, decimal: GraphFst, cardinal: GraphFst):
        super().__init__(name="measure", kind="verbalize")

        optional_prefix = pynini.closure(
            pynutil.delete("prefix:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.insert('-')
            + pynutil.delete("\"")
            + delete_space,
            0,
            1
        )

        # Removes the negative attribute and leaves the sign if occurs
        optional_sign = pynini.closure(
            pynutil.delete("negative:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.accep("-")
            + pynutil.delete("\"")
            + delete_space,
            0,
            1,
        )

        graph_decimal = (
            pynutil.delete("decimal {")
            + delete_space
            + decimal.numbers
            + delete_space
            + pynutil.delete("}")
        )

        graph_cardinal = (
            pynutil.delete("cardinal {")
            + delete_space
            + cardinal.numbers
            + delete_space
            + pynutil.delete("}")
        )

        unit = (
                pynutil.delete("units:")
                + delete_space
                + pynutil.delete("\"")
                + pynini.closure(NEMO_CHAR - " ", 1)
                + pynutil.delete("\"")
                + delete_space
        )

        spaced_unit = (
                pynutil.delete("spaced_units:")
                + delete_space
                + pynutil.delete("\"")
                + pynini.closure(NEMO_CHAR - " ", 1)
                + pynutil.delete("\"")
                + delete_space
        )

        numbers_units = delete_space + (unit | pynutil.insert(" ") + spaced_unit)
        numbers_graph = (graph_cardinal | graph_decimal) + numbers_units

        one_units = unit | (pynutil.insert(" ") + spaced_unit)
        one_graph = (
            delete_space
            + pynutil.insert("1")
            + one_units
            + pynutil.delete("cardinal { integer: \"1\" }")
        )

        graph = optional_prefix + optional_sign + (numbers_graph | one_graph)
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()


if __name__ == "__main__":

    from nemo_text_processing.inverse_text_normalization.he.graph_utils import apply_fst
    from nemo_text_processing.inverse_text_normalization.he.verbalizers.cardinal import CardinalFst
    from nemo_text_processing.inverse_text_normalization.he.verbalizers.decimal_he import DecimalFst

    cardinal = CardinalFst()
    decimal = DecimalFst()
    g = MeasureFst(decimal, cardinal).fst

    # To test this FST, remove comment out and change the input text
    # apply_fst("טקסט לבדיקה", g)
