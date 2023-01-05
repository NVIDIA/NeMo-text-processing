from typing import List


def extend_list_with_mutations(input: List) -> List:
    out = []

    UPPER_VOWELS = "AEIOUÁÉÍÓÚ"
    LOWER_VOWELS = "aeiouáéíóú"

    for word in input:
        out.append(word)
        if word[0] in UPPER_VOWELS:
            out.append(f"h{word}")
            out.append(f"n{word}")
            out.append(f"t{word}")
        elif word[0] in LOWER_VOWELS:
            out.append(f"h{word}")
            out.append(f"n-{word}")
            out.append(f"t-{word}")

    return out