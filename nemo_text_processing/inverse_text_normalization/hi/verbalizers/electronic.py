# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.hi.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space


class ElectronicFst(GraphFst):
    """
    Finite state transducer for verbalizing electronic
        e.g. tokens { electronic { username: "kumar" domain: "gmail.com" } } -> kumar@gmail.com
        e.g. tokens { electronic { domain: "https://google.com" } } -> https://google.com
        e.g. tokens { electronic { path: "C:\\Users\\HP\\Desktop" } } -> C:\\Users\\HP\\Desktop
    """

    def __init__(self):
        super().__init__(name="electronic", kind="verbalize")

        def field_graph(field_name: str) -> pynini.Fst:
            return (
                pynutil.delete(f"{field_name}:")
                + delete_space
                + pynutil.delete("\"")
                + pynini.closure(NEMO_NOT_QUOTE, 1)
                + pynutil.delete("\"")
            )

        domain_graph = field_graph("domain")
        username_graph = field_graph("username")
        path_graph = field_graph("path")

        email_graph = username_graph + pynutil.insert("@") + delete_space + domain_graph

        graph = email_graph | path_graph | domain_graph

        self.fst = self.delete_tokens(graph).optimize()
