import pynini
from pynini.lib import pynutil


def apply_fst(text, fst):
  """ Given a string input, returns the output string produced by traversing the path with lowest weight.
  If no valid path accepts input string, returns an error.
  """
  try:
     print(pynini.shortestpath(text @ fst).string())
  except pynini.FstOpError:
    print(f"Error: No valid output with given input: '{text}'")

# number_graph = pynini.string_file("data/numbers/numbers.tsv")
# thousands_graph = pynini.string_file("data/numbers/thousands.tsv")
# final_graph = number_graph | thousands_graph

# apply_fst('अब्ज',final_graph)

from taggers.cardinal import CardinalFst

fst = CardinalFst().fst
apply_fst('अब्ज',fst)