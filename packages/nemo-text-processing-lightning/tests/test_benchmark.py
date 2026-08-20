# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
# limitations under the License.

"""What this package is worth, measured.

Deselected by default because they are slow and machine-dependent:

    pytest -m benchmark -s --tn_cache_dir=/path/to/grammars

Each one prints a table and asserts a floor well under what it actually
measures, so the numbers are informative and a real regression still fails.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from test_integration import use_lightning

pytestmark = pytest.mark.benchmark

# One fixed sentence per density, so repeating a template varies length with
# the number of semiotic classes held constant.
TEMPLATES = {
    "plain": "The quick brown fox jumps over the lazy dog near the river bank today.",
    "light": "The team shipped 5 builds to the staging cluster this week.",
    "medium": "On 3/4/2023 we shipped 5 units at $5.50 each to the west coast.",
    "heavy": (
        "Call 555-0105 before 5:30 p.m. on 3/4/2023 about the $5,204.50 invoice, "
        "up 55% from Q4, shipping 12 kg to nemo@example.com."
    ),
}


def best_of(fn, text, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(text)
        best = min(best, time.perf_counter() - t0)
    return best


def test_tagging_speedup(tagger, normalizer):
    """Tagging, which is 96.7% of normalization runtime, against pynini."""
    import pynini

    def pynini_tag(text):
        lattice = pynini.escape(text) @ normalizer.tagger.fst
        return pynini.shortestpath(lattice, nshortest=1, unique=True).string()

    print(f"\n{'density':8s} {'copies':>6s} {'chars':>6s} {'pynini':>10s} {'nemo-text-processing-lightning':>10s} {'speedup':>8s}")
    speedups = []
    for name, template in TEMPLATES.items():
        for copies in (1, 4, 16, 32):
            text = " ".join([template] * copies)
            t_base = best_of(pynini_tag, text)
            t_ours = best_of(tagger.tag, text)
            speedups.append(t_base / t_ours)
            print(
                f"{name:8s} {copies:6d} {len(text):6d} {t_base * 1e3:9.1f}ms "
                f"{t_ours * 1e3:9.1f}ms {t_base / t_ours:7.1f}x"
            )
    assert min(speedups) > 1.5, f"slowest cell only {min(speedups):.2f}x"


def test_end_to_end_speedup(normalizer, tagger, inputs, monkeypatch):
    """The whole `normalize()` call, tagging substituted, by density and length.

    Both axes matter. Density, because lookahead removes *dead* hypotheses and
    dense text keeps more genuinely alive. Length, because the pipeline's
    non-tagging stages are a fixed cost that a short input cannot amortise --
    see `test_tagging_share_of_pipeline` for the ceiling that implies.
    """
    cases = [("corpus (200 short inputs)", inputs[:200])]
    for name, template in TEMPLATES.items():
        for copies in (4, 32):
            cases.append((f"{name}, {copies} sentences", [" ".join([template] * copies)]))

    stock_times = []
    for _, texts in cases:
        t0 = time.perf_counter()
        for text in texts:
            normalizer.normalize(text)
        stock_times.append(time.perf_counter() - t0)

    use_lightning(monkeypatch, tagger)
    print(f"\n{'input':28s} {'stock':>8s} {'nemo-text-processing-lightning':>9s} {'speedup':>8s}")
    speedups = []
    for (label, texts), stock in zip(cases, stock_times):
        t0 = time.perf_counter()
        for text in texts:
            normalizer.normalize(text)
        ours = time.perf_counter() - t0
        speedups.append(stock / ours)
        print(f"{label:28s} {stock:7.2f}s {ours:8.2f}s {stock / ours:7.2f}x")
    print(f"{'':28s} {'':>8s} {'range':>9s} "
          f"{min(speedups):.1f}-{max(speedups):.1f}x")
    assert min(speedups) > 1.2, f"slowest case only {min(speedups):.2f}x"


def test_tagging_share_of_pipeline(normalizer, inputs):
    """How much of `normalize()` is tagging -- the ceiling on any tagger change.

    Speeding tagging by `s` when it is fraction `f` of runtime gives at best
    1 / ((1 - f) + f / s), so this is what bounds every other number here. It is
    also why short inputs look worse: the non-tagging stages are a fixed cost
    per call, and a ten-character input cannot amortise them.
    """
    import pynini

    def measure(texts):
        t0 = time.perf_counter()
        for text in texts:
            normalizer.normalize(text)
        total = time.perf_counter() - t0
        t0 = time.perf_counter()
        for text in texts:
            lattice = pynini.escape(text) @ normalizer.tagger.fst
            pynini.shortestpath(lattice, nshortest=1, unique=True).string()
        return time.perf_counter() - t0, total

    cases = [("corpus (200 short inputs)", inputs[:200])]
    cases += [
        (f"{name}, 32 sentences", [" ".join([TEMPLATES[name]] * 32)])
        for name in ("plain", "medium", "heavy")
    ]
    print(f"\n{'input':28s} {'tagging':>9s} {'total':>8s} {'share':>7s}")
    shares = []
    for label, texts in cases:
        tag, total = measure(texts)
        shares.append(tag / total)
        print(f"{label:28s} {tag:8.2f}s {total:7.2f}s {tag / total * 100:6.1f}%")
    assert max(shares) > 0.9, f"tagging is only {max(shares) * 100:.0f}% of the pipeline"


def test_concurrency_scaling(tagger):
    """Threads through one Tagger. A GIL-holding implementation cannot beat 1.0x."""
    texts = [
        "Call 555-0105 before 5:30 p.m. on 3/4/2023 about the $5,204.50 invoice.",
        "The quick brown fox jumps over the lazy dog 3 times.",
        "Résumé costs €25 on 1/2/2024.",
    ]
    rounds = 40

    t0 = time.perf_counter()
    for _ in range(rounds * 8):
        for text in texts:
            tagger.tag(text)
    sequential = time.perf_counter() - t0

    print(f"\n{'threads':>8s} {'wall':>9s} {'speedup':>8s}")
    print(f"{1:8d} {sequential:8.2f}s {1.0:7.2f}x")
    best = 1.0
    for n_threads in (2, 4, 8):
        per_thread = (rounds * 8) // n_threads

        def work():
            for _ in range(per_thread):
                for text in texts:
                    tagger.tag(text)

        workers = [threading.Thread(target=work) for _ in range(n_threads)]
        t0 = time.perf_counter()
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        wall = time.perf_counter() - t0
        best = max(best, sequential / wall)
        print(f"{n_threads:8d} {wall:8.2f}s {sequential / wall:7.2f}x")
    assert best > 1.5, f"no parallel speedup: best {best:.2f}x"


def test_preparation_is_a_one_off(far_path, tmp_path):
    """Cost of preparing the lookahead artifact, and what it buys on load."""
    import nemo_text_processing_lightning

    t0 = time.perf_counter()
    tagger = nemo_text_processing_lightning.Tagger.from_far(far_path, cache_dir=tmp_path)
    cold = time.perf_counter() - t0

    t0 = time.perf_counter()
    nemo_text_processing_lightning.Tagger.from_far(far_path, cache_dir=tmp_path)
    warm = time.perf_counter() - t0

    far_mb = far_path.stat().st_size / 1e6
    art_mb = Path(tagger.artifact_path).stat().st_size / 1e6
    print(
        f"\nprepare {cold:.2f}s cold, {warm:.2f}s from cache; "
        f"FAR {far_mb:.1f} MB -> artifact {art_mb:.1f} MB"
    )
    assert warm < cold, (cold, warm)
