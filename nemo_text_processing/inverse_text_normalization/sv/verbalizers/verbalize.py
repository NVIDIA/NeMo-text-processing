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

from nemo_text_processing.inverse_text_normalization.sv.verbalizers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.sv.verbalizers.date import DateFst
from nemo_text_processing.inverse_text_normalization.sv.verbalizers.decimal import DecimalFst
from nemo_text_processing.inverse_text_normalization.sv.verbalizers.electronic import ElectronicFst
from nemo_text_processing.inverse_text_normalization.sv.verbalizers.fraction import FractionFst
from nemo_text_processing.inverse_text_normalization.sv.verbalizers.ordinal import OrdinalFst
from nemo_text_processing.inverse_text_normalization.sv.verbalizers.telephone import TelephoneFst
from nemo_text_processing.inverse_text_normalization.sv.verbalizers.time import TimeFst
from nemo_text_processing.inverse_text_normalization.sv.verbalizers.whitelist import WhiteListFst
from nemo_text_processing.text_normalization.en.graph_utils import GraphFst
from nemo_text_processing.text_normalization.sv.verbalizers.cardinal import CardinalFst as TNCardinalVerbalizer


class VerbalizeFst(GraphFst):
    """
    Composes other verbalizer grammars.
    For deployment, this grammar will be compiled and exported to OpenFst Finite State Archive (FAR) File.
    More details to deployment at NeMo/tools/text_processing_deployment.
    """

    def __init__(self, project_input: bool = False):
        super().__init__(name="verbalize", kind="verbalize")
        tn_cardinal_verbalizer = TNCardinalVerbalizer(deterministic=False, project_input=project_input)

        cardinal = CardinalFst(tn_cardinal_verbalizer=tn_cardinal_verbalizer, project_input=project_input)
        cardinal_graph = cardinal.fst
        date_graph = DateFst(project_input=project_input).fst
        decimal = DecimalFst(project_input=project_input)
        decimal_graph = decimal.fst
        electronic_graph = ElectronicFst(project_input=project_input).fst
        fraction_graph = FractionFst(project_input=project_input).fst
        ordinal_graph = OrdinalFst(project_input=project_input).fst
        telephone_graph = TelephoneFst(project_input=project_input).fst
        time_graph = TimeFst(project_input=project_input).fst
        whitelist_graph = WhiteListFst(project_input=project_input).fst
        graph = time_graph | decimal_graph | cardinal_graph | ordinal_graph | date_graph | fraction_graph | electronic_graph | telephone_graph | whitelist_graph
        self.fst = graph
