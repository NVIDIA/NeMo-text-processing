# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure native normalization against supplied JSONL written/spoken references."""

import json
import re
import time
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

from byte_transducer import NativePipeline


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--grammars", type=Path, required=True)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--mismatches", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    native = NativePipeline(args.library, args.grammars)
    setup_seconds = time.perf_counter() - started
    counts: Counter[str] = Counter()
    seconds: Counter[str] = Counter()
    started = time.perf_counter()
    try:
        with args.mismatches.open("w") as differences:
            for path in args.input:
                with path.open() as source:
                    for line in source:
                        row = json.loads(line)
                        written = row["written"]
                        counts["rows"] += 1
                        begin = time.perf_counter()
                        output = native.normalize(written)
                        seconds["native_normalization"] += time.perf_counter() - begin
                        if output is None:
                            counts["no_path"] += 1
                            differences.write(json.dumps({"written": written, "error": "no path"}) + "\n")
                            continue
                        reference = row["spoken"]
                        if output != reference:
                            counts["exact_differences"] += 1
                            counts["word_token_differences"] += re.findall(r"\w+", output) != re.findall(
                                r"\w+", reference
                            )
                            differences.write(
                                json.dumps(
                                    {
                                        "written": written,
                                        "candidate": output,
                                        "reference": reference,
                                    }
                                )
                                + "\n"
                            )
        report = {
            "counts": dict(counts),
            "seconds": dict(seconds),
            "setup_seconds": setup_seconds,
            "elapsed_seconds": time.perf_counter() - started,
            "scope": "Includes JSONL input and mismatch output; excludes source-token decoding and corpus output.",
            "remaining_python": "JSONL benchmark input/output and call wrapper; use normalize_stream for a native record loop.",
        }
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        native.close()


if __name__ == "__main__":
    main()
