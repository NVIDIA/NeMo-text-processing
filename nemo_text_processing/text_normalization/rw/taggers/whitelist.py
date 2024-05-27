from nemo_text_processing.text_normalization.en.graph_utils import GraphFst
import pynini
from pynini.lib import pynutil
from nemo_text_processing.text_normalization.rw.utils import get_abs_path


transliterations = pynini.string_file(get_abs_path("data/whitelist/kinya_transliterations.tsv")) 

class WhiteListFst(GraphFst):
    def __init__(self):
        super().__init__(name="whitelist", kind="classify")

        whitelist = transliterations
        graph = pynutil.insert("name: \"") + whitelist + pynutil.insert("\"")
        self.fst = graph.optimize()
