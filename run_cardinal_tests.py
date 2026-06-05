# run_cardinal_tests.py  --  simple checker for the Cardinal exercise.
# Usage (from the repo root, inside your conda env):
#   python run_cardinal_tests.py <lang> <tn|itn> <path-to-test_cases_cardinal.txt>
# Example:
#   python run_cardinal_tests.py LANGCODE DIRECTION test_cases_cardinal.txt
import importlib
import sys

import pynini


def apply_fst(fst, text):
    lattice = text @ fst
    if lattice.num_states() == 0:
        return None  # input was rejected by the grammar
    out = pynini.shortestpath(lattice)
    return out.string() if out.num_states() else None


def main():
    lang, direction, path = sys.argv[1], sys.argv[2], sys.argv[3]
    base = "text_normalization" if direction == "tn" else "inverse_text_normalization"
    print("Script started")
    tagger = importlib.import_module(f"nemo_text_processing.{base}.{lang}.taggers.cardinal").CardinalFst().fst
    verbalizer = importlib.import_module(f"nemo_text_processing.{base}.{lang}.verbalizers.cardinal").CardinalFst().fst

    passed = failed = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or "~" not in line:
                continue
            inp, expected = [s.strip() for s in line.split("~", 1)]
            tagged = apply_fst(tagger, inp)
            result = apply_fst(verbalizer, tagged) if tagged is not None else None
            if result == expected:
                passed += 1
            else:
                failed += 1
                print(f"FAIL: {inp!r}  ->  got {result!r}, expected {expected!r}")

    print(f"\n{passed} passed, {failed} failed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
