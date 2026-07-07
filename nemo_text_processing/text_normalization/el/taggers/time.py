import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_space,
    insert_space,
)


class TimeFst(GraphFst):
    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="time", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.graph_no_tokens

        delete_sep = pynutil.delete(":")

        hour = pynini.compose(
            (pynutil.delete("0") + NEMO_DIGIT) | ("1" + NEMO_DIGIT) | ("2" + pynini.union("0", "1", "2", "3")),
            cardinal_graph,
        ).optimize()

        minutes = pynini.compose(
            pynutil.delete("0") + NEMO_DIGIT,
            cardinal_graph,
        )
        minutes |= pynini.compose(
            pynini.union(*"012345") + NEMO_DIGIT,
            cardinal_graph,
        )
        minutes = minutes.optimize()

        hm = (
            pynutil.insert("hours: \"")
            + hour
            + pynutil.insert("\"")
            + delete_sep
            + pynutil.insert("minutes: \"")
            + minutes
            + pynutil.insert("\"")
            + pynutil.insert(" preserve_order: true")
        )
        h = pynutil.add_weight(pynutil.insert("hours: \"") + hour + pynutil.insert("\"") + pynutil.delete(":00"), 0.1)
        tagger_graph = (hm | h).optimize()

        hour_v = (
            pynutil.delete("hours:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        minutes_v = (
            pynutil.delete("minutes:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        verbalizer_graph = (
            hour_v + delete_space + insert_space + minutes_v + delete_space + pynutil.delete("preserve_order: true")
        )
        verbalizer_graph |= hour_v + delete_space

        self.final_graph = (tagger_graph @ verbalizer_graph).optimize()
        self.fst = pynutil.insert("hours: \"") + self.final_graph + pynutil.insert("\"")
        self.fst = self.add_tokens(self.fst).optimize()
