# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
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

from nemo_text_processing.text_normalization.pt.graph_utils import NEMO_NOT_QUOTE, GraphFst, insert_space
from nemo_text_processing.text_normalization.pt.utils import get_abs_path, load_labels


class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing Portuguese fraction numbers, e.g.
        fraction { numerator: "um" denominator: "meio" morphosyntactic_features: "ordinal" } -> um meio
        fraction { integer_part: "dois" numerator: "três" denominator: "quarto" } -> dois e três quartos
        fraction { numerator: "dois" denominator: "onze" morphosyntactic_features: "avos" } -> dois onze avos

    Denominator routing (set by the tagger, unchanged here):

    * **Ordinal** (``morphosyntactic_features: "ordinal"``): denominators 2–10, 100, 1000 — spoken as
      ordinals with plural ``s`` on the denominator when the numerator is not ``um`` (``três quartos``).
    * **Avos** (``… "avos"``): all other positive integer denominators — cardinal denominator + the word
      ``avos`` (``três onze avos``). Optional ``sobre`` between numerator and denominator is controlled by
      ``data/fractions/specials.tsv`` key ``avos_between`` (empty = single space; ``sobre`` = ``… sobre …``).
    * **Mixed** numbers use ``connector`` from the same TSV (default `` e ``) after the integer part.

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="fraction", kind="verbalize", deterministic=deterministic)
        labels = load_labels(get_abs_path("data/fractions/specials.tsv"))
        spec = {r[0]: r[1] for r in labels if len(r) >= 2}
        connector_raw = spec.get("connector", " e ").strip()
        connector = insert_space + pynutil.insert(connector_raw) + insert_space
        minus = spec.get("minus", "menos ").rstrip()
        plural_suffix = spec.get("plural_suffix", "s")
        avos_word = spec.get("avos_suffix", " avos").strip()
        avos_between = spec.get("avos_between", "").strip()
        numerator_one_val = spec.get("numerator_one", "um")
        denominator_half_val = spec.get("denominator_half", "meio")

        optional_sign = pynini.closure(pynini.cross('negative: "true" ', minus) + insert_space, 0, 1)

        integer = pynutil.delete('integer_part: "') + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete('" ')

        numerator_one = pynutil.delete('numerator: "') + pynini.accep(numerator_one_val) + pynutil.delete('" ')
        numerator_rest = (
            pynutil.delete('numerator: "')
            + pynini.difference(pynini.closure(NEMO_NOT_QUOTE), pynini.accep(numerator_one_val))
            + pynutil.delete('" ')
        )

        denom_ordinal = (
            pynutil.delete('denominator: "')
            + pynini.closure(NEMO_NOT_QUOTE)
            + pynutil.delete('" morphosyntactic_features: "ordinal"')
        )
        denom_meio = (
            pynutil.delete('denominator: "')
            + pynini.accep(denominator_half_val)
            + pynutil.delete('" morphosyntactic_features: "ordinal"')
        )
        denom_avos = (
            pynutil.delete('denominator: "')
            + pynini.closure(NEMO_NOT_QUOTE)
            + pynutil.delete('" morphosyntactic_features: "avos"')
        )

        fraction_ordinal_singular = numerator_one + insert_space + denom_ordinal
        fraction_ordinal_plural = numerator_rest + insert_space + denom_ordinal + pynutil.insert(plural_suffix)
        fraction_ordinal = pynini.union(fraction_ordinal_singular, fraction_ordinal_plural)

        if avos_between:
            avos_mid = insert_space + pynutil.insert(avos_between) + insert_space
        else:
            avos_mid = insert_space

        fraction_avos = (
            pynini.union(numerator_one, numerator_rest)
            + avos_mid
            + denom_avos
            + insert_space
            + pynutil.insert(avos_word)
        )

        fraction = pynini.union(fraction_ordinal, fraction_avos)
        mixed_um_meio = integer + connector + pynutil.delete('numerator: "' + numerator_one_val + '" " ') + denom_meio
        optional_integer = pynini.closure(integer + connector + insert_space, 0, 1)
        graph = optional_sign + pynini.union(
            pynutil.add_weight(mixed_um_meio, -0.01),
            optional_integer + fraction,
        )

        self.inner_graph = graph.optimize()
        self.fst = self.delete_tokens(graph).optimize()
