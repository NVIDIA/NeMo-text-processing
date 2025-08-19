# Test

import pynini

# from taggers.cardinal import CardinalFst
from verbalizers.cardinal import CardinalFst


def apply_fst(text, fst):
    """Given a string input, returns the output string
    produced by traversing the path with lowest weight.
    If no valid path accepts input string, returns an
    error.
    """
    try:
        print(pynini.shortestpath(text @ fst).string())
    except pynini.FstOpError:
        print(f"Error: No valid output with given input: '{text}'")


example = 'cardinal { negative: "-" integer: "85.000.101" }'

cardinal = CardinalFst().fst
# ordinal = OrdinalFst(cardinal).fst

apply_fst(example, cardinal)


'''
# Test
def apply_fst(text, fst):
    """Given a string input, returns the output string
    produced by traversing the path with lowest weight.
    If no valid path accepts input string, returns an
    error.
    """
    try:
        print(pynini.shortestpath(text @ fst).string())
    except pynini.FstOpError:
        print(f"Error: No valid output with given input: '{text}'")


example = "erstes jahrhundert"

cardinal = CardinalFst()
ordinal = OrdinalFst(cardinal).fst

apply_fst(example, cardinal)
'''


# Working test with the __main__ guard

from nemo_text_processing.inverse_text_normalization.ger.taggers.cardinal import (
    CardinalFst,
)


def apply_fst(text, fst):
    """Given a string input, returns the output string
    produced by traversing the path with lowest weight.
    If no valid path accepts input string, returns an
    error.
    """
    try:
        print(pynini.shortestpath(text @ fst).string())
    except pynini.FstOpError:
        print(f"Error: No valid output with given input: '{text}'")


def main():
    example = "zwei komma sechs"
    cardinal = CardinalFst()
    decimal = DecimalFst(cardinal).fst
    apply_fst(example, decimal)


if __name__ == "__main__":
    main()


# Tests 02/25/25

from nemo_text_processing.inverse_text_normalization.ger.verbalizers.cardinal import (
    CardinalFst,
)
from nemo_text_processing.inverse_text_normalization.ger.verbalizers.decimal import (
    DecimalFst,
)
from nemo_text_processing.inverse_text_normalization.ger.verbalizers.fraction import (
    FractionFst,
)
from nemo_text_processing.inverse_text_normalization.ger.taggers.measure import (
    MeasureFst as measure_tagger,
)
from nemo_text_processing.inverse_text_normalization.ger.taggers.cardinal import (
    CardinalFst as cardinal_tagger,
)
from nemo_text_processing.inverse_text_normalization.ger.taggers.decimal import (
    DecimalFst as decimal_tagger,
)
from nemo_text_processing.inverse_text_normalization.ger.taggers.fraction import (
    FractionFst as fraction_tagger,
)


def apply_fst(text, fst):
    """Given a string input, returns the output string
    produced by traversing the path with lowest weight.
    If no valid path accepts input string, returns an
    error.
    """
    try:
        print(pynini.shortestpath(text @ fst).string())
    except pynini.FstOpError:
        print(f"Error: No valid output with given input: '{text}'")


def main():
    # instantiates taggers
    cardinal_tagger = cardinal_tagger()
    decimal_tagger = decimal_tagger()
    fraction_tagger = fraction_tagger()
    # actual tagger
    tagger_grammar = measure_tagger(
        cardinal_tagger, decimal_tagger, fraction_tagger
    ).fst
    example = "drei ohm"
    tagged_output = pynini.shortestpath(example @ tagger_grammar).string()
    print(tagged_output)
    cardinal = CardinalFst()
    decimal = DecimalFst()
    fraction = FractionFst()
    measure = MeasureFst(cardinal, decimal, fraction).fst
    apply_fst(tagged_output, measure)


if __name__ == "__main__":
    main()
