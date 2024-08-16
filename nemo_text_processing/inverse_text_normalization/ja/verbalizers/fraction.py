# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.ja.graph_utils import NEMO_NOT_QUOTE, GraphFst


class FractionFst(GraphFst):
    def __init__(self):
        """
        Fitite state transducer for classifying fractions
<<<<<<< HEAD
        e.g.,
=======
        e.g., 
>>>>>>> 0a4a21c (Jp itn 20240221 (#141))
        fraction { denominator: "4" numerator: "3" } -> 3/4
        fraction { integer: "1" denominator: "4" numerator: "3" } -> 1 3/4
        fraction { integer: "1" denominator: "4" numerator: "3" } -> 1 3/4
        fraction { denominator: "√3" numerator: "1" } -> 1/√3
        fraction { denominator: "1.65" numerator: "50" } -> 50/1.65
        fraction { denominator: "2√6" numerator: "3" } -> 3/2√6
        """
        super().__init__(name="fraction", kind="verbalize")

        sign_component = pynutil.delete("negative: \"") + pynini.closure("-",1) + pynutil.delete("\"")

        integer_component = pynutil.delete("integer_part: \"") + pynini.closure(NEMO_NOT_QUOTE,1) + pynutil.delete("\"")

        denominator_component = (
            pynutil.delete("denominator: \"") + pynini.closure(NEMO_NOT_QUOTE,1) + pynutil.delete("\"")
        )

        numerator_component = pynutil.insert(" ") + pynutil.delete("numerator: \"") + pynini.closure(NEMO_NOT_QUOTE,1) + pynutil.delete("\"")

        # regular_graph = (
        #     pynini.closure((sign_component + pynutil.delete(" ")), 0, 1)
        #     + pynini.closure(integer_component + pynutil.delete(" ") + pynutil.insert(" "))
        #     + numerator_component
        #     + pynutil.delete(" ")
        #     + pynutil.insert("/")
        #     + denominator_component
        # )
        # this is the orignal grammar that worked except for integer part with space issue


        final_graph = integer_component + pynini.closure(pynutil.delete(" "), 0,1) + pynutil.insert("1 test ")+ (pynutil.insert(" ") + numerator_component + pynutil.delete(" ") + pynutil.insert("/") + denominator_component)
        # this final graph was an attempt to make fixings

        final_graph = self.delete_tokens(final_graph)
        self.fst = final_graph.optimize()
