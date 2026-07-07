# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.import pynini
from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer

itn = InverseNormalizer(lang="ta", overwrite_cache=True)

total = 0
passed = 0
failed = []
skipped = []

with open(
    "tests/nemo_text_processing/ta/data_inverse_text_normalization/blind_test_case.txt",
    encoding="utf-8",
) as f:

    for line_num, line in enumerate(f, start=1):
        line = line.strip()

        if not line:
            continue

        parts = line.split("\t")

        if len(parts) != 3:
            skipped.append((line_num, line))
            continue

        category, expected, spoken = parts

        try:
            pred = itn.inverse_normalize(spoken, verbose=False)
        except Exception as e:
            pred = f"ERROR: {e}"

        total += 1

        if pred == expected:
            passed += 1
        else:
            failed.append((category, spoken, expected, pred))

# Save failures
with open("failures.txt", "w", encoding="utf-8") as out:
    for category, spoken, expected, pred in failed:
        out.write(f"CATEGORY: {category}\n")
        out.write(f"INPUT: {spoken}\n")
        out.write(f"EXPECTED: {expected}\n")
        out.write(f"PREDICTED: {pred}\n")
        out.write("=" * 80 + "\n")

print(f"\nPassed: {passed}/{total}")
print(f"Failed: {len(failed)}")
print(f"Skipped: {len(skipped)}")
print(f"Accuracy: {100 * passed / total:.2f}%")

print("\nFirst 20 skipped lines:\n")

for line_num, line in skipped[:20]:
    print("=" * 80)
    print("LINE   :", line_num)
    print("CONTENT:", repr(line))

print("\nFirst 20 failures:\n")

for category, spoken, expected, pred in failed[:20]:
    print("=" * 80)
    print("CATEGORY :", category)
    print("INPUT    :", spoken)
    print("EXPECTED :", expected)
    print("PREDICTED:", pred)
