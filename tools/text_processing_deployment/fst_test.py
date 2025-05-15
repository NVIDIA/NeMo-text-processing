import argparse
from pynini.lib import rewrite
from nemo_text_processing.text_normalization.normalize import Normalizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_file", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--grammars_dir", type=str, required=True)
    parser.add_argument("--language", type=str, required=True)
    args = parser.parse_args()

    print(f"[INFO] Loading normalizer with grammars from {args.grammars_dir}")
    normalizer = Normalizer(
        input_case="lower_cased", lang=args.language, cache_dir=args.grammars_dir, overwrite_cache=False
    )

    print(f"[INFO] Reading input test file: {args.input_data_file}")
    with open(args.input_data_file, encoding="utf-8") as f:
        lines = f.readlines()

    total = 0
    passed = 0

    for line in lines:
        if "~" not in line:
            continue
        input_text, expected_output = line.strip().split("~")
        pred = normalizer.normalize(input_text)

        if pred.strip() == expected_output.strip():
            print(f" ALLOW: {input_text} → {pred}")
            passed += 1
        else:
            print(f" DENY:  {input_text} → {pred} (expected: {expected_output})")
        total += 1

    print(f"\n[RESULT] Passed {passed}/{total} ({(passed/total)*100:.2f}%)")
