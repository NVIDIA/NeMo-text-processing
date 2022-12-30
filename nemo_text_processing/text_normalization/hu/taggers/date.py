

def _days_morphological():
    with open(get_abs_path("data/dates/days.tsv")) as inf:
        for line in inf.readlines():
            l, r = line.strip().split("\t")
            last_char = r[-1]
            assert last_char in ["a", "e"]
            accent_char = "é" if last_char == "e" else "á"
            rounded_char = "ő" if last_char == "e" else "ó"
            stem = r[:-1]
            endings = {
                "{num}-{last_char}": "{stem}{last_char}",
                "{num}-{accent_char}n": "{stem}{accent_char}n",
                "{num}-i": "{word}i",
                "{num}-{last_char}i": "{word}i",
                "{num}-{accent_char}ig": "{stem}{accent_char}ig",
                "{num}-{accent_char}t{rounded_char}l": "{stem}{accent_char}t{rounded_char}l"
            }