import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_space,
    insert_space,
)


class TelephoneFst(GraphFst):
    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="telephone", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.graph_no_tokens
        digit = pynini.compose(
            pynini.union("0", NEMO_DIGIT),
            cardinal_graph,
        ).optimize()

        separator = pynini.union("-", ".", " ")
        digit_group = (digit + pynini.closure(insert_space + digit)).optimize()

        number_part = (
            digit_group
            + pynini.cross(separator, " ")
            + digit_group
            + pynini.closure(pynini.cross(separator, " ") + digit_group)
        )
        number_part = pynutil.insert("number_part: \"") + number_part + pynutil.insert("\"")

        tagger_graph = number_part.optimize()

        number_v = (
            pynutil.delete("number_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        verbalizer_graph = number_v + delete_space

        self.final_graph = (tagger_graph @ verbalizer_graph).optimize()
        self.fst = pynutil.insert("number_part: \"") + self.final_graph + pynutil.insert("\"")
        self.fst = self.add_tokens(self.fst).optimize()
