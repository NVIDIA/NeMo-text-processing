import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.fr.utils import get_abs_path

def apply_fst(text, fst):
  """ Given a string input, returns the output string
  produced by traversing the path with lowest weight.
  If no valid path accepts input string, returns an
  error.
  """
  try:
     print(pynini.shortestpath(text @ fst).string())
  except pynini.FstOpError:
    print(f"Error: No valid output with given input: '{text}'")

zero = pynini.string_map([("zéro","0")]) # French only pronounces zeroes as stand alone
digits_map = pynini.string_map([ # pynini function that creates explicit input-output mappings for a WFST
				("un","1"),
				("une","1"),
				("deux","2"),
				("trois","3"),
				("quatre","4"),
				("cinq","5"),
				("six","6"),
				("sept","7"),
				("huit","8"),
				("neuf","9")
])

digits = pynini.string_file("data/numbers/digits.tsv")

teens = pynini.string_map([
    ("onze", "11"),
    ("douze", "12"), 
    ("treize", "13"),
    ("quatorze", "14"),
    ("quinze", "16"), 
])

tens = pynini.string_map([("dix", "1")])
delete_hyphen = pynini.closure(pynutil.delete("-"), 0, 1) # Applies a closure from 0-1 of operation. Equivalent to regex /?/

graph_tens = tens + delete_hyphen + digits
graph_tens_and_teens = graph_tens | teens

graph_digits = digits | pynutil.insert("0")

apply_fst("un", graph_tens_and_teens)