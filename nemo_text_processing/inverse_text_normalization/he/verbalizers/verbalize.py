from nemo_text_processing.text_normalization.en.graph_utils import GraphFst

from nemo_text_processing.inverse_text_normalization.he.verbalizers.date import DateFst
from nemo_text_processing.inverse_text_normalization.he.verbalizers.time import TimeFst
from nemo_text_processing.inverse_text_normalization.he.verbalizers.measure import MeasureFst
from nemo_text_processing.inverse_text_normalization.he.verbalizers.ordinal import OrdinalFst
from nemo_text_processing.inverse_text_normalization.he.verbalizers.decimal import DecimalFst
from nemo_text_processing.inverse_text_normalization.he.verbalizers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.he.verbalizers.whitelist import WhiteListFst
# from nemo_text_processing.inverse_text_normalization.he.verbalizers.electronic import ElectronicFst
# from nemo_text_processing.inverse_text_normalization.he.verbalizers.telephone import TelephoneFst
# from nemo_text_processing.inverse_text_normalization.he.verbalizers.money import MoneyFst


class VerbalizeFst(GraphFst):
    """
    Composes other verbalizer grammars in Hebrew.
    For deployment, this grammar will be compiled and exported to OpenFst Finite State Archive (FAR) File.
    More details to deployment at NeMo/tools/text_processing_deployment.
    """

    def __init__(self):
        super().__init__(name="verbalize", kind="verbalize")

        cardinal = CardinalFst()
        cardinal_graph = cardinal.fst

        ordinal_graph = OrdinalFst().fst

        decimal = DecimalFst()
        decimal_graph = decimal.fst

        measure_graph = MeasureFst(decimal=decimal, cardinal=cardinal).fst

        time_graph = TimeFst().fst

        date_graph = DateFst().fst

        whitelist_graph = WhiteListFst().fst

        # money_graph = MoneyFst(decimal=decimal).fst
        # telephone_graph = TelephoneFst().fst
        # electronic_graph = ElectronicFst().fst

        graph = (
            time_graph
            | date_graph
            | measure_graph
            | ordinal_graph
            | decimal_graph
            | cardinal_graph
            | whitelist_graph
            # | money_graph
            # | telephone_graph
            # | electronic_graph
        )
        self.fst = graph
