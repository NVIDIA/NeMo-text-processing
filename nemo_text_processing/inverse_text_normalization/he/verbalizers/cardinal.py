import pynini
from pynini.lib import pynutil
from nemo_text_processing.inverse_text_normalization.he.graph_utils import NEMO_DIGIT, NEMO_ALPHA, GraphFst, delete_space


class CardinalFst(GraphFst):
    """
    Finite state transducer for verbalizing cardinal in Hebrew
        e.g. cardinal { prefix: "וב" integer: "3405"} -> וב-3,405
        e.g. cardinal { negative: "-" integer: "904" } -> -904
        e.g. cardinal { prefix: "כ" integer: "123" } -> כ-123

    """

    def __init__(self):
        super().__init__(name="cardinal", kind="verbalize")

        # Need parser to group digits by threes
        exactly_three_digits = NEMO_DIGIT ** 3
        at_most_three_digits = pynini.closure(NEMO_DIGIT, 1, 3)

        # Thousands separator
        group_by_threes = (
            at_most_three_digits +
            (pynutil.insert(",") + exactly_three_digits).closure()
        )

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

        # removes integer aspect
        graph = (
                pynutil.delete("integer:")
                + delete_space
                + pynutil.delete("\"")
                + pynini.closure(NEMO_DIGIT, 1)  # Accepts at least one digit
                + pynutil.delete("\"")
        )

        # Add thousands separator
        graph = graph @ group_by_threes

        self.numbers = graph

        # add prefix and sign
        graph = optional_prefix + optional_sign + graph

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()


if __name__ == "__main__":
    from nemo_text_processing.inverse_text_normalization.he.graph_utils import apply_fst

    cardinal = CardinalFst().fst
    # apply_fst('cardinal { negative: "-" integer: "204" }', cardinal)
    # apply_fst('cardinal { integer: "3204" }', cardinal)
    # apply_fst('cardinal { prefix: "ב" integer: "50"}', cardinal)
    # apply_fst('cardinal { prefix: "כ" integer: "123"}', cardinal)
    # apply_fst('cardinal { prefix: "ו" integer: "305"}', cardinal)
    # apply_fst('cardinal { prefix: "וב" integer: "3405"}', cardinal)
    # apply_fst('cardinal { integer: "10" }', cardinal)
    # apply_fst('cardinal { integer: "0" }', cardinal)
    # apply_fst('cardinal { negative: "-" integer: "5" }', cardinal)
    # apply_fst('cardinal { integer: "122" }', cardinal)
    # apply_fst('cardinal { integer: "120" }', cardinal)
    # apply_fst('cardinal { negative: "-" integer: "904" }', cardinal)
    # apply_fst('cardinal { integer: "1000" }', cardinal)
    # apply_fst('cardinal { integer: "13" }', cardinal)
    # apply_fst('cardinal { integer: "1030" }', cardinal)
    # apply_fst('cardinal { integer: "2003" }', cardinal)
    # apply_fst('cardinal { integer: "3045" }', cardinal)
    # apply_fst('cardinal { integer: "45002" }', cardinal)
    # apply_fst('cardinal { integer: "900009" }', cardinal)
    # apply_fst('cardinal { integer: "1000000" }', cardinal)
    # apply_fst('cardinal { integer: "3945002" }', cardinal)
    # apply_fst('cardinal { integer: "3045002" }', cardinal)
    # apply_fst('cardinal { prefix: "כ" integer: "123" }', cardinal)
