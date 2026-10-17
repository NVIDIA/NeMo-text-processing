# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export an existing deterministic English NeMo normalizer for native evaluation."""

from argparse import ArgumentParser
from pathlib import Path

from byte_transducer import export_fst
from nemo_text_processing.text_normalization.normalize import Normalizer


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    normalizer = Normalizer(
        input_case="cased",
        lang="en",
        cache_dir=str(args.cache_dir),
        overwrite_cache=False,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, grammar in (
        ("tagger", normalizer.tagger.fst),
        ("verbalizer", normalizer.verbalizer.fst),
        ("postprocessor", normalizer.post_processor.fst),
    ):
        export_fst(grammar, args.output_dir / (name + ".bin"))


if __name__ == "__main__":
    main()
