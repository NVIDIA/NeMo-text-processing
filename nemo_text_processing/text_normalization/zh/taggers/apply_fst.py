import pynini


def apply_fst(text, fst):  # applies this function for test purposes: apply_fst("the text",the fst_you_built)
    try:
        print(pynini.shortestpath(text @ fst).string())
    except:
        print(f"Error: No valid output with given input: ' {text}'")
