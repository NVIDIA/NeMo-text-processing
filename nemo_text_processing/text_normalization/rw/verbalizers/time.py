import pynini
from pynini.lib import pynutil
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
    NEMO_CHAR
)

class VerbalizeTimeFst(GraphFst):
    def __init__(self):
        super().__init__(name="time",kind="verbalize")
        hour = (pynutil.delete("hours:")+delete_space+pynutil.delete("\"")+pynini.closure(NEMO_CHAR,1,60)+pynutil.delete("\"")+delete_space+pynutil.delete("minutes:")+delete_space+pynutil.delete("\"")+pynini.closure(NEMO_CHAR,1,60)+pynutil.delete("\""))

        graph = hour 
        delete_tokens = self.delete_tokens(graph)
        
        self.fst = delete_tokens.optimize()
