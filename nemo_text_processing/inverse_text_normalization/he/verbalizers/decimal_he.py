import pynini
from pynini.lib import pynutil
from nemo_text_processing.inverse_text_normalization.he.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space, NEMO_ALPHA, NEMO_DIGIT


class DecimalFst(GraphFst):
    """
    Finite state transducer for verbalizing decimal,
    e.g. decimal { integer_part: "0"  fractional_part: "33" } -> 0.33
    e.g. decimal { negative: "true" integer_part: "400"  fractional_part: "323" } -> -400.323
    e.g. decimal { integer_part: "4"  fractional_part: "5" quantity: "מיליון" } -> 4.5 מיליון

    """

    def __init__(self):
        super().__init__(name="decimal", kind="verbalize")
        optionl_sign = pynini.closure(pynini.cross("negative: \"true\"", "-") + delete_space, 0, 1)

        # Need parser to group digits by threes
        exactly_three_digits = NEMO_DIGIT ** 3
        at_most_three_digits = pynini.closure(NEMO_DIGIT, 1, 3)

        # Thousands separator
        group_by_threes = (
            at_most_three_digits +
            (pynutil.insert(",") + exactly_three_digits).closure()
        )

        integer = (
            pynutil.delete("integer_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        integer = integer @ group_by_threes

        optional_integer = pynini.closure(integer + delete_space, 0, 1)

        fractional = (
            pynutil.insert(".")
            + pynutil.delete("fractional_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        optional_fractional = pynini.closure(fractional + delete_space, 0, 1)

        quantity = (
            pynutil.delete("quantity:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        optional_quantity = pynini.closure(pynutil.insert(" ") + quantity + delete_space, 0, 1)

        # Keep the prefix if exists and add a dash
        optional_prefix = pynini.closure(
            pynutil.delete("prefix:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_ALPHA, 1)
            + pynutil.insert("-")
            + pynutil.delete("\"")
            + delete_space,
            0,
            1,
        )

        graph = optional_prefix + optional_integer + optional_fractional + optional_quantity
        self.numbers = graph
        graph = optionl_sign + graph
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()


if __name__ == "__main__":

    from nemo_text_processing.inverse_text_normalization.he.graph_utils import apply_fst

    g = DecimalFst().fst

    # To test this FST, remove comment out and change the input text
    # apply_fst("טקסט לבדיקה", g)
