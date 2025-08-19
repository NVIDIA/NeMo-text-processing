# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from nemo_text_processing.inverse_text_normalization.ger.verbalizers.cardinal import (
    CardinalFst,
)
from nemo_text_processing.inverse_text_normalization.ger.verbalizers.ordinal import (
    OrdinalFst,
)
from nemo_text_processing.inverse_text_normalization.ger.verbalizers.decimal import (
    DecimalFst,
)
from nemo_text_processing.inverse_text_normalization.ger.verbalizers.fraction import (
    FractionFst,
)
from nemo_text_processing.inverse_text_normalization.ger.verbalizers.date import (
    DateFst,
)
from nemo_text_processing.inverse_text_normalization.ger.verbalizers.time import (
    TimeFst,
)
from nemo_text_processing.inverse_text_normalization.ger.verbalizers.money import (
    MoneyFst,
)
from nemo_text_processing.inverse_text_normalization.ger.verbalizers.measure import (
    MeasureFst,
)
from nemo_text_processing.inverse_text_normalization.ger.verbalizers.telephone import (
    TelephoneFst,
)
from nemo_text_processing.inverse_text_normalization.ger.graph_utils import GraphFst

from pynini.lib import pynutil


class VerbalizeFst(GraphFst):
    """
    Composes other verbalizer grammars.
    For deployment, this grammar will be compiled and exported to OpenFst Finite State Archive (FAR) File.
    More details to deployment at NeMo/tools/text_processing_deployment.
    """

    def __init__(self):
        super().__init__(name="verbalize", kind="verbalize")
        cardinal = CardinalFst()
        cardinal_graph = cardinal.fst
        ordinal = OrdinalFst()
        ordinal_graph = ordinal.fst
        decimal = DecimalFst()
        decimal_graph = decimal.fst
        fraction = FractionFst()
        fraction_graph = fraction.fst
        date = DateFst()
        date_graph = date.fst
        time = TimeFst()
        time_graph = time.fst
        money = MoneyFst()
        money_graph = money.fst
        measure = MeasureFst(cardinal, decimal, fraction)
        measure_graph = measure.fst
        telephone = TelephoneFst()
        telephone_graph = telephone.fst
        graph = (
            cardinal_graph
            | ordinal_graph
            | decimal_graph
            | fraction_graph
            | date_graph
            | time_graph
            | money_graph
            | measure_graph
            | telephone_graph
        )
        self.fst = graph


'''

import logging
from nemo_text_processing.inverse_text_normalization.ger.verbalizers.cardinal import (
    CardinalFst,
)
from nemo_text_processing.inverse_text_normalization.ger.verbalizers.ordinal import (
    OrdinalFst,
)
from nemo_text_processing.inverse_text_normalization.ger.verbalizers.decimal import (
    DecimalFst,
)
from nemo_text_processing.inverse_text_normalization.ger.verbalizers.fraction import (
    FractionFst,
)
from nemo_text_processing.inverse_text_normalization.ger.verbalizers.date import DateFst
from nemo_text_processing.inverse_text_normalization.ger.verbalizers.time import TimeFst
from nemo_text_processing.inverse_text_normalization.ger.verbalizers.money import (
    MoneyFst,
)
from nemo_text_processing.inverse_text_normalization.ger.verbalizers.measure import (
    MeasureFst,
)
from nemo_text_processing.inverse_text_normalization.ger.graph_utils import GraphFst
from pynini.lib import pynutil
import pynini

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def fst_to_string(fst):
    """Convert an FST to a readable string representation."""
    try:
        return pynini.shortestpath(fst).string()
    except Exception as e:
        return f"Error converting FST to string: {e}"


class VerbalizeFst(GraphFst):
    """
    Composes other verbalizer grammars.
    For deployment, this grammar will be compiled and exported to OpenFst Finite State Archive (FAR) File.
    More details to deployment at NeMo/tools/text_processing_deployment.
    """

    def __init__(self):
        super().__init__(name="verbalize", kind="verbalize")

        # Initialize subgraphs
        logging.debug("Initializing CardinalFst...")
        cardinal = CardinalFst()
        cardinal_graph = cardinal.fst
        logging.debug(f"Cardinal Graph (string): {fst_to_string(cardinal_graph)}")

        logging.debug("Initializing OrdinalFst...")
        ordinal = OrdinalFst()
        ordinal_graph = ordinal.fst
        logging.debug(f"Ordinal Graph (string): {fst_to_string(ordinal_graph)}")

        logging.debug("Initializing DecimalFst...")
        decimal = DecimalFst()
        decimal_graph = decimal.fst
        logging.debug(f"Decimal Graph (string): {fst_to_string(decimal_graph)}")

        logging.debug("Initializing FractionFst...")
        fraction = FractionFst()
        fraction_graph = fraction.fst
        logging.debug(f"Fraction Graph (string): {fst_to_string(fraction_graph)}")

        logging.debug("Initializing DateFst...")
        date = DateFst()
        date_graph = date.fst
        logging.debug(f"Date Graph (string): {fst_to_string(date_graph)}")

        logging.debug("Initializing TimeFst...")
        time = TimeFst()
        time_graph = time.fst
        logging.debug(f"Time Graph (string): {fst_to_string(time_graph)}")

        logging.debug("Initializing MoneyFst...")
        money = MoneyFst()
        money_graph = money.fst
        logging.debug(f"Money Graph (string): {fst_to_string(money_graph)}")

        logging.debug("Initializing MeasureFst...")
        measure = MeasureFst()
        measure_graph = measure.fst
        logging.debug(f"Measure Graph (string): {fst_to_string(measure_graph)}")

        # Compose the final graph
        logging.debug("Composing the final graph...")
        graph = (
            cardinal_graph
            | ordinal_graph
            | decimal_graph
            | fraction_graph
            | date_graph
            | time_graph
            | money_graph
            | measure_graph
        )
        logging.debug(f"Final Verbalize Graph (string): {fst_to_string(graph)}")

        self.fst = graph
'''
