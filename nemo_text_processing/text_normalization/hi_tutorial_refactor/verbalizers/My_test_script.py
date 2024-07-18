import pynini
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space
from pynini.lib import rewrite, pynutil


class CardinalFst(GraphFst):
    """
    Finite state transducer for verbalizing cardinal, e.g.
        cardinal { negative: "true" integer: "२३" } -> minus तेईस

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="verbalize", deterministic=deterministic)

        self.optional_sign = pynini.cross("negative: \"true\"", "minus ")
        if not deterministic:
            self.optional_sign |= pynini.cross("negative: \"true\"", "negative ")
            self.optional_sign |= pynini.cross("negative: \"true\"", "dash ")

        self.optional_sign = pynini.closure(self.optional_sign + delete_space, 0, 1)

        integer = pynini.closure(NEMO_NOT_QUOTE)

        self.integer = delete_space + pynutil.delete("\"") + integer + pynutil.delete("\"")
        integer = pynutil.delete("integer:") + self.integer

        self.numbers = self.optional_sign + integer
        delete_tokens = self.delete_tokens(self.numbers)
        self.fst = delete_tokens.optimize()

from nemo_text_processing.text_normalization.hi.taggers.cardinal import CardinalFst as cfst

tagger = cfst().fst
input_text = "११"  
tagger_output = rewrite.top_rewrite(input_text, tagger)
print(tagger_output)
cardinal = CardinalFst().fst   # calling cardinalFst       
                                                                                             
output = rewrite.top_rewrite(tagger_output, cardinal)           
print(output)

