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

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space


class ElectronicFst(GraphFst):
    """
    Finite state transducer for verbalizing electronic
        e.g. tokens { electronic { username: "cdf1" domain: "abc.edu" } } -> cdf1@abc.edu
        e.g. tokens { electronic { protocol: "www." domain: "nvidia.com" } } -> www.nvidia.com
    """

    def __init__(self, project_input: bool = False):
        super().__init__(name="electronic", kind="verbalize", project_input=project_input)

        user_name = (
            pynutil.delete("username:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        domain = (
            pynutil.delete("domain:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        protocol = (
            pynutil.delete("protocol:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        # Handle preserve_order field (optional)
        preserve_order = (
            delete_space
            + pynutil.delete("preserve_order:")
            + delete_space
            + pynutil.delete("\"")
            + (pynutil.delete("true") | pynutil.delete("false"))
            + pynutil.delete("\"")
        )
        optional_preserve_order = pynini.closure(preserve_order, 0, 1)

        # Email format: username @ domain
        email_graph = user_name + delete_space + pynutil.insert("@") + domain + optional_preserve_order

        # URL format with protocol only (for cases where only protocol field exists)
        url_protocol_only = protocol + optional_preserve_order

        # URL format with protocol + domain (for cases where both fields exist)
        url_protocol_domain = protocol + delete_space + domain + optional_preserve_order

        # Union of all formats
        graph = email_graph | url_protocol_only | url_protocol_domain

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
