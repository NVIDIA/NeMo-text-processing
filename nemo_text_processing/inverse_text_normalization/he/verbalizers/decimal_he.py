import pynini
from pynini.lib import pynutil
from nemo_text_processing.inverse_text_normalization.he.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space, NEMO_ALPHA


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

        integer = (
            pynutil.delete("integer_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
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

    decimal = DecimalFst().fst
    # apply_fst('decimal { integer_part: "0"  fractional_part: "33" }', decimal)
    # apply_fst('decimal { negative: "true" integer_part: "400"  fractional_part: "323" }', decimal)
    # apply_fst('decimal { integer_part: "4"  fractional_part: "5" quantity: "מיליון" }', decimal)
