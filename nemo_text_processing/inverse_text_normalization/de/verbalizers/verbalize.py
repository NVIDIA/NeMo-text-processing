# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.de.verbalizers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.de.verbalizers.decimal import DecimalFst
from nemo_text_processing.inverse_text_normalization.de.verbalizers.measure import MeasureFst
from nemo_text_processing.inverse_text_normalization.de.verbalizers.money import MoneyFst
from nemo_text_processing.inverse_text_normalization.de.verbalizers.time import TimeFst
from nemo_text_processing.text_normalization.de.verbalizers.cardinal import CardinalFst as TNCardinalVerbalizer
from nemo_text_processing.text_normalization.de.verbalizers.decimal import DecimalFst as TNDecimalVerbalizer
from nemo_text_processing.text_normalization.en.graph_utils import GraphFst


class VerbalizeFst(GraphFst):
    """
    Composes other verbalizer grammars.
    For deployment, this grammar will be compiled and exported to OpenFst Finite State Archive (FAR) File.
    More details to deployment at NeMo/tools/text_processing_deployment.
    """

    def __init__(self, project_input: bool = False):
        super().__init__(name="verbalize", kind="verbalize")
        tn_cardinal_verbalizer = TNCardinalVerbalizer(project_input=project_input)
        tn_decimal_verbalizer = TNDecimalVerbalizer(project_input=project_input)

        cardinal = CardinalFst(tn_cardinal_verbalizer=tn_cardinal_verbalizer, project_input=project_input)
        cardinal_graph = cardinal.fst
        decimal = DecimalFst(tn_decimal_verbalizer=tn_decimal_verbalizer, project_input=project_input)
        decimal_graph = decimal.fst
        measure_graph = MeasureFst(decimal=decimal, cardinal=cardinal, project_input=project_input).fst
        money_graph = MoneyFst(decimal=decimal, project_input=project_input).fst
        time_graph = TimeFst(project_input=project_input).fst
        graph = time_graph | money_graph | measure_graph | decimal_graph | cardinal_graph
        self.fst = graph
