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


from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import GraphFst


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone, e.g.
        "восемь девятьсот тринадцать девятьсот восемьдесят три пятьдесят шесть ноль один" -> telephone { number_part: "8-913-983-56-01" }

    Args:
        tn_telephone: Text normalization telephone graph
    """

    def __init__(
        self,
        tn_telephone: GraphFst,
        project_input: bool = False
    ):
        super().__init__(name="telephone", kind="classify", project_input=project_input)

        tn_telephone = tn_telephone.final_graph
        graph = tn_telephone.invert().optimize()
        graph = pynutil.insert("number_part: \"") + graph + pynutil.insert("\"")
        graph = self.add_tokens(graph)
        self.fst = graph.optimize()
