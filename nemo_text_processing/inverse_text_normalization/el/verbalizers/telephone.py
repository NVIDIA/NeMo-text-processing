import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, GraphFst


class TelephoneFst(GraphFst):
    def __init__(self):
        super().__init__(name="telephone", kind="verbalize")

        number_part = pynutil.delete("number_part: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        optional_country_code = pynini.closure(
            pynutil.delete("country_code: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
            + pynini.accep(" "),
            0,
            1,
        )
        delete_tokens = self.delete_tokens(optional_country_code + number_part)
        self.fst = delete_tokens.optimize()
