# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.ta.graph_utils import GraphFst
from nemo_text_processing.inverse_text_normalization.ta.taggers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.ta.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_SIGMA, delete_space, insert_space
from nemo_text_processing.text_normalization.ta.graph_utils import FRACTION_WORD


def denominator_to_number() -> 'pynini.FstLike':
    """
    Undoes the locative -இல் on a spoken denominator (நான்கில் -> நான்கு): the tabulated forms of
    ``data/fraction/denominator_locative.tsv``, the regular locative (-இல் replaces the final -உ,
    a ம்-final scale word takes -த்தில்), and the hundreds compound -நூற்றில்.
    """
    tabulated = pynini.string_file(get_abs_path("data/fraction/denominator_locative.tsv"))
    regular = NEMO_SIGMA + pynini.union(pynini.cross("ில்", "ு"), pynini.cross("த்தில்", "ம்"), pynini.cross("ியில்", "ி"))
    hundreds = NEMO_SIGMA + pynini.cross("நூற்றில்", "நூறு")
    return pynini.union(tabulated, regular, hundreds).optimize()


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying spoken fractions, e.g.
        நான்கில் மூன்று -> fraction { denominator: "4" numerator: "3" }
        ஐந்து கீழ் எழுபத்தேழு -> fraction { numerator: "5" denominator: "77" }

    The locative reading (denominator first) is what ASR output carries; the கீழ் reading is
    what TN emits.

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: CardinalFst):
        super().__init__(name="fraction", kind="classify")

        numerator = pynutil.insert("numerator: \"") + cardinal.words_to_digits + pynutil.insert("\"")

        # Undo the denominator's locative form, then read it as a number. The undoing is
        # ambiguous until the number lexicon decides, so it is not determinized: that re-times the
        # number's delayed outputs and explodes.
        denominator_words = (denominator_to_number() @ cardinal.words_to_digits).optimize()
        denominator = pynutil.insert("denominator: \"") + denominator_words + pynutil.insert("\"")
        graph = denominator + delete_space + insert_space + numerator

        # "N கீழ் M" order: ஐந்து கீழ் எழுபத்தேழு -> 5/77.
        graph |= (
            numerator
            + delete_space
            + pynutil.delete(FRACTION_WORD)
            + delete_space
            + insert_space
            + pynutil.insert("denominator: \"")
            + cardinal.words_to_digits
            + pynutil.insert("\"")
        )
        self.fst = self.add_tokens(graph).optimize()
