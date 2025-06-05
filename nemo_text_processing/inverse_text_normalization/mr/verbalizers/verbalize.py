# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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

from nemo_text_processing.text_normalization.en.graph_utils import GraphFst
from nemo_text_processing.inverse_text_normalization.mr.verbalizers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.mr.verbalizers.date import DateFst
from nemo_text_processing.inverse_text_normalization.mr.verbalizers.decimal import DecimalFst
from nemo_text_processing.inverse_text_normalization.mr.verbalizers.time import TimeFst


class VerbalizeFst(GraphFst):
    """
    Composes other verbalizer grammars.
    For deployment, this grammar will be compiled and exported to OpenFst Finite State Archive (FAR) File.
    More details to deployment at NeMo/tools/text_processing_deployment.
    """

    def __init__(self, project_input: bool = False):
        super().__init__(name="verbalize", kind="verbalize")
        cardinal_graph = CardinalFst(project_input=project_input).fst
        decimal_graph = DecimalFst(project_input=project_input).fst
        time_graph = TimeFst(project_input=project_input).fst
        date_graph = DateFst(project_input=project_input).fst
        graph = cardinal_graph | decimal_graph | time_graph | date_graph
        self.fst = graph
