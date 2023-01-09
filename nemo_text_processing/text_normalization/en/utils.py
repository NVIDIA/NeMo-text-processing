import os
import string

import pynini
from nemo_text_processing.tn_itn_utils.graph_utils import NEMO_SIGMA
from pynini.examples import plurals
from pynini.lib import pynutil


def get_abs_path(rel_path):
    """
    Get absolute path

    Args:
        rel_path: relative path to this file

    Returns absolute path
    """
    return os.path.dirname(os.path.abspath(__file__)) + '/' + rel_path


_c = pynini.union(
    "b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "q", "r", "s", "t", "v", "w", "x", "y", "z"
)
_ies = NEMO_SIGMA + _c + pynini.cross("y", "ies")
_es = NEMO_SIGMA + pynini.union("s", "sh", "ch", "x", "z") + pynutil.insert("es")
_s = NEMO_SIGMA + pynutil.insert("s")

suppletive = pynini.string_file(get_abs_path("./data/suppletive.tsv"))
graph_plural = plurals._priority_union(
    suppletive, plurals._priority_union(_ies, plurals._priority_union(_es, _s, NEMO_SIGMA), NEMO_SIGMA), NEMO_SIGMA
).optimize()

SINGULAR_TO_PLURAL = graph_plural
PLURAL_TO_SINGULAR = pynini.invert(graph_plural)
TO_LOWER = pynini.union(*[pynini.cross(x, y) for x, y in zip(string.ascii_uppercase, string.ascii_lowercase)])
TO_UPPER = pynini.invert(TO_LOWER)
