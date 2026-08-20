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

"""nemo-text-processing-lightning's tagger against the pynini baseline, over the English test corpus.

The tagger is weighted over the tropical semiring and its shortest path is not
unique, so "same string as pynini" is the wrong contract: two correct
implementations can return different members of a tied set. Some of those ties
are not cosmetic — `31/31/100` tags as either `thirty one/thirty one/one
hundred` or `thirty one slash thirty one slash one hundred`, both at weight
1.12010, and they do not sound the same.

So the assertion is the invariant that actually matters: **never return a path
that costs more than pynini's**. Where the two disagree, both strings are priced
through the same lattice and the weights must be equal.

This compares tagged strings because that is all this package produces. Whether
the *normalized* output is unchanged belongs with the change that wires the
tagger into `Normalizer`.
"""

from __future__ import annotations

import time

def squash(text: str) -> str:
    return " ".join(text.split())


def weight_of(lattice, candidate: str):
    """Cost of one specific output string through `lattice`, or None if absent."""
    import pynini

    proj = pynini.project(lattice, "output").optimize()
    inter = pynini.intersect(proj, pynini.accep(pynini.escape(candidate)))
    if inter.start() == pynini.NO_STATE_ID:
        return None
    return float(pynini.shortestpath(inter).paths().weight())


def test_tagged_output_is_never_worse_than_pynini(tagger, normalizer, inputs):
    import pynini

    worse, tied = [], []
    t_base = t_ours = 0.0
    for text in inputs:
        t0 = time.perf_counter()
        lattice = pynini.escape(text) @ normalizer.tagger.fst
        base = pynini.shortestpath(lattice, nshortest=1, unique=True).string()
        t_base += time.perf_counter() - t0

        t0 = time.perf_counter()
        ours = tagger.tag(text)
        t_ours += time.perf_counter() - t0

        if squash(base) == squash(ours):
            continue
        w_base, w_ours = weight_of(lattice, base), weight_of(lattice, ours)
        if w_ours is None or w_base is None or w_ours > w_base:
            worse.append((text, base, ours, w_base, w_ours))
        else:
            tied.append((text, base, ours, w_base))

    n = len(inputs)
    print(
        f"\n{n - len(tied) - len(worse)}/{n} identical, {len(tied)} tied on weight, "
        f"{len(worse)} worse; {t_base:.2f}s baseline vs {t_ours:.2f}s nemo-text-processing-lightning "
        f"({t_base / t_ours:.1f}x)"
    )
    for text, a, b, w in tied[:5]:
        print(f"  tie @ {w:.5f}  {text!r}\n    pynini  : {a[:70]!r}\n    nemo-text-processing-lightning: {b[:70]!r}")
    for text, a, b, wa, wb in worse[:5]:
        print(f"  WORSE {text!r}: pynini {wa} -> {a[:60]!r}, nemo-text-processing-lightning {wb} -> {b[:60]!r}")
    assert not worse, f"{len(worse)} of {n} inputs took a costlier path than pynini"


def test_tag_releases_the_gil(tagger):
    """A second thread must make progress while a composition runs.

    pywrapfst holds the lock for the whole composition; not holding it is a
    primary reason this package exists. Measured as ticks-per-second in a
    sibling thread, with and without a tagging load, so the assertion is a ratio
    rather than a wall-clock threshold.
    """
    import threading

    text = (
        "On January 5th, 2021 the company reported $1,234.56 in revenue, "
        "up 12.5% from Q3, at 3:45 PM ET. " * 4
    )

    def ticker(stop, out):
        n = 0
        while not stop.is_set():
            n += 1
        out.append(n)

    def measure(load: bool) -> int:
        stop = threading.Event()
        out: list[int] = []
        thread = threading.Thread(target=ticker, args=(stop, out))
        thread.start()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 1.0:
            if load:
                tagger.tag(text)
            else:
                time.sleep(0.005)
        stop.set()
        thread.join()
        return out[0]

    idle = measure(load=False)
    busy = measure(load=True)
    print(f"\nticker: {idle} ticks/s idle, {busy} ticks/s while tagging ({busy / idle:.2f}x)")
    # A GIL-holding implementation drives this to near zero.
    assert busy > 0.20 * idle, (idle, busy)


def test_concurrent_tagging_is_consistent_and_scales(tagger):
    """Eight threads through one Tagger: same answers, real parallelism.

    Consistency, because releasing the GIL means several compositions genuinely
    run at once over one shared lookahead FST -- it is read-only and each
    composition builds its own matchers, but that is worth asserting rather than
    assuming. And scaling, because a GIL-holding implementation cannot beat 1.0x
    here however many cores it is given.
    """
    import threading

    texts = [
        "Call 555-0105 before 5:30 p.m. on 3/4/2023 about the $5,204.50 invoice.",
        "The quick brown fox jumps over the lazy dog 3 times.",
        "Résumé costs €25 on 1/2/2024.",
    ]
    expected = [tagger.tag(t) for t in texts]
    rounds, threads = 40, 8
    errors: list = []

    def work():
        try:
            for _ in range(rounds):
                for i, text in enumerate(texts):
                    if tagger.tag(text) != expected[i]:
                        errors.append(("mismatch", i))
        except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
            errors.append(exc)

    workers = [threading.Thread(target=work) for _ in range(threads)]
    t0 = time.perf_counter()
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    parallel = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(rounds * threads):
        for text in texts:
            tagger.tag(text)
    sequential = time.perf_counter() - t0

    print(
        f"\n{threads} threads {parallel:.2f}s vs sequential {sequential:.2f}s "
        f"({sequential / parallel:.2f}x)"
    )
    assert not errors, errors[:3]
    assert sequential / parallel > 1.5, (sequential, parallel)
