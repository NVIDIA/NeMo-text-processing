from nemo_text_processing.text_normalization.en.graph_utils import GraphFst
from nemo_text_processing.text_normalization.rw.verbalizers.time import VerbalizeTimeFst
from nemo_text_processing.text_normalization.en.verbalizers.cardinal import CardinalFst

class VerbalizeFst(GraphFst):
    def __init__(self):
        super().__init__(name="verbalize", kind="verbalize")
        cardinal = CardinalFst()
        cardinal_graph = cardinal.fst
        time = VerbalizeTimeFst().fst

        graph = (
            cardinal_graph
           | time
        )
        self.fst = graph
        

