from nemo_text_processing.inverse_text_normalization.el.verbalizers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.el.verbalizers.date import DateFst
from nemo_text_processing.inverse_text_normalization.el.verbalizers.decimal import DecimalFst
from nemo_text_processing.inverse_text_normalization.el.verbalizers.electronic import ElectronicFst
from nemo_text_processing.inverse_text_normalization.el.verbalizers.fraction import FractionFst
from nemo_text_processing.inverse_text_normalization.el.verbalizers.measure import MeasureFst
from nemo_text_processing.inverse_text_normalization.el.verbalizers.money import MoneyFst
from nemo_text_processing.inverse_text_normalization.el.verbalizers.ordinal import OrdinalFst
from nemo_text_processing.inverse_text_normalization.el.verbalizers.telephone import TelephoneFst
from nemo_text_processing.inverse_text_normalization.el.verbalizers.time import TimeFst
from nemo_text_processing.inverse_text_normalization.el.verbalizers.whitelist import WhiteListFst
from nemo_text_processing.text_normalization.en.graph_utils import GraphFst


class VerbalizeFst(GraphFst):
    def __init__(self):
        super().__init__(name="verbalize", kind="verbalize")
        cardinal = CardinalFst()
        cardinal_graph = cardinal.fst
        ordinal_graph = OrdinalFst().fst
        decimal = DecimalFst()
        decimal_graph = decimal.fst
        measure_graph = MeasureFst(decimal=decimal, cardinal=cardinal).fst
        money_graph = MoneyFst(decimal=decimal).fst
        time_graph = TimeFst().fst
        date_graph = DateFst().fst
        whitelist_graph = WhiteListFst().fst
        telephone_graph = TelephoneFst().fst
        electronic_graph = ElectronicFst().fst
        fraction_graph = FractionFst().fst
        graph = (
            time_graph
            | date_graph
            | money_graph
            | measure_graph
            | ordinal_graph
            | decimal_graph
            | cardinal_graph
            | fraction_graph
            | whitelist_graph
            | telephone_graph
            | electronic_graph
        )
        self.fst = graph
