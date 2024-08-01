import pynini
from pynini.lib import pynutil
from nemo_text_processing.inverse_text_normalization.he.graph_utils import NEMO_CHAR, GraphFst, delete_space


class MoneyFst(GraphFst):
    """
    Finite state transducer for verbalizing money, e.g.
        money { integer_part: "12" fractional_part: "05" currency: "$" } -> $12.05

    Args:
        decimal: DecimalFst
    """

    def __init__(self, decimal: GraphFst):
        super().__init__(name="money", kind="verbalize")
        unit = (
            pynutil.delete("currency:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_CHAR - " ", 1)
            + pynutil.delete("\"")
        )
        graph = unit + delete_space + decimal.numbers
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()


if __name__ == "__main__":
    from nemo_text_processing.inverse_text_normalization.he.graph_utils import apply_fst
    from nemo_text_processing.inverse_text_normalization.he.verbalizers.decimal_he import DecimalFst
    decimal = DecimalFst()
    money = MoneyFst(decimal).fst
    # apply_fst('money { integer_part: "3" currency: "₪" }', money)
    # apply_fst('money { integer_part: "1" currency: "₪" }', money)
    # apply_fst('money { integer_part: "47" currency: "€" }', money)
    # apply_fst('money { integer_part: "2" currency: "₪" fractional_part: "99" }', money)
    # apply_fst('money { currency: "₪" integer_part: "0" fractional_part: "05" }', money)
