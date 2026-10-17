# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare the native evaluator against OpenFst on bounded weighted graphs."""

import random
import subprocess
import tempfile
import unittest
from pathlib import Path

import pynini
from .byte_transducer import ByteTransducer, NativePipeline, export_fst


class ByteTransducerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.library = cls.root / "libtn_byte_transducer.so"
        subprocess.run(
            [
                "g++",
                "-std=c++17",
                "-O3",
                "-Wall",
                "-Wextra",
                "-shared",
                "-fPIC",
                str(Path(__file__).with_suffix(".cc").with_name("byte_transducer.cc")),
                "-o",
                str(cls.library),
            ],
            check=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def check_graph(self, graph: pynini.Fst, inputs: list[str]) -> None:
        path = self.root / "grammar.bin"
        export_fst(graph, path)
        runtime = ByteTransducer(self.library, path)
        try:
            for text in inputs:
                with self.subTest(text=text):
                    actual = runtime.rewrite(text)
                    lattice = pynini.compose(pynini.accep(pynini.escape(text)), graph)
                    expected_cost = (
                        float(
                            pynini.shortestdistance(lattice, reverse=True)[
                                lattice.start()
                            ]
                        )
                        if lattice.start() >= 0
                        else float("inf")
                    )
                    self.assertEqual(actual is None, expected_cost == float("inf"))
                    if actual is not None:
                        output, cost = actual
                        self.assertAlmostEqual(cost, expected_cost, places=5)
                        constrained = pynini.compose(
                            lattice, pynini.accep(pynini.escape(output))
                        )
                        self.assertGreaterEqual(constrained.start(), 0)
                        output_cost = float(
                            pynini.shortestdistance(constrained, reverse=True)[
                                constrained.start()
                            ]
                        )
                        self.assertAlmostEqual(output_cost, expected_cost, places=5)
        finally:
            runtime.close()

    def test_random_acyclic_weighted_graphs(self) -> None:
        rng = random.Random(81)
        for _ in range(50):
            graph = pynini.Fst()
            for _ in range(12):
                graph.add_state()
            graph.set_start(0)
            graph.set_final(11, pynini.Weight("tropical", rng.randrange(-3, 4) / 8))
            for source in range(11):
                for target in range(source + 1, 12):
                    if rng.random() < 0.22 or target == source + 1:
                        graph.add_arc(
                            source,
                            pynini.Arc(
                                rng.choice([0, 97, 98]),
                                rng.choice([0, 120, 121]),
                                pynini.Weight("tropical", rng.randrange(-8, 9) / 8),
                                target,
                            ),
                        )
            self.check_graph(graph, ["", "a", "b", "ab", "ba", "aaa", "abab"])

    def test_unicode_and_literal_metacharacters(self) -> None:
        graph = pynini.union(
            pynini.cross(pynini.escape("café [x]"), pynini.escape("café [output]")),
            pynini.cross("12", "twelve"),
            pynini.cross("", "empty"),
        )
        self.check_graph(graph, ["café [x]", "12", "", "no match", "12"])

    def test_consuming_cycles(self) -> None:
        graph = pynini.closure(pynini.cross("a", "xyz"))
        self.check_graph(graph, ["", "a", "aaaa", "b", "a"])

    def test_epsilon_cycle_with_nonnegative_cost(self) -> None:
        graph = pynini.Fst()
        graph.add_state()
        graph.add_state()
        graph.set_start(0)
        graph.set_final(1)
        graph.add_arc(0, pynini.Arc(0, 120, pynini.Weight("tropical", 1), 0))
        graph.add_arc(0, pynini.Arc(97, 121, pynini.Weight("tropical", 0), 1))
        self.check_graph(graph, ["a", "", "aa", "a"])

    def test_invalid_and_closed_handles(self) -> None:
        path = self.root / "invalid.bin"
        path.write_bytes(b"not a transducer")
        with self.assertRaises(RuntimeError):
            ByteTransducer(self.library, path)
        export_fst(pynini.accep("x"), path)
        runtime = ByteTransducer(self.library, path)
        runtime.close()
        runtime.close()
        with self.assertRaises(RuntimeError):
            runtime.rewrite("x")

    def test_native_pipeline_field_reordering(self) -> None:
        tags = 'tokens { money { currency: "dollars" integer: "twelve" } }'
        reordered = ' tokens {  money { integer: "twelve" currency: "dollars"  }  } '
        export_fst(pynini.cross("input", tags), self.root / "tagger.bin")
        export_fst(
            pynini.cross(reordered, "twelve dollars"), self.root / "verbalizer.bin"
        )
        export_fst(
            pynini.cross("twelve dollars", "TWELVE DOLLARS"),
            self.root / "postprocessor.bin",
        )
        native = NativePipeline(self.library, self.root)
        try:
            self.assertEqual(native.normalize_core("input"), "TWELVE DOLLARS")
            self.assertIsNone(native.normalize_core("no path"))
            self.assertEqual(native.normalize_core("input"), "TWELVE DOLLARS")
        finally:
            native.close()

    def test_native_pipeline_preserves_explicit_field_order(self) -> None:
        tags = 'tokens { money { currency: "dollars" integer: "twelve" preserve_order: true } }'
        forbidden = ' tokens {  money { integer: "twelve" currency: "dollars" preserve_order: true  }  } '
        export_fst(pynini.cross("input", tags), self.root / "tagger.bin")
        export_fst(
            pynini.cross(forbidden, "twelve dollars"), self.root / "verbalizer.bin"
        )
        export_fst(pynini.accep("twelve dollars"), self.root / "postprocessor.bin")
        native = NativePipeline(self.library, self.root)
        try:
            self.assertIsNone(native.normalize_core("input"))
        finally:
            native.close()

    def test_nonbyte_labels_are_rejected(self) -> None:
        graph = pynini.Fst()
        graph.add_state()
        graph.add_state()
        graph.set_start(0)
        graph.set_final(1)
        graph.add_arc(0, pynini.Arc(1000, 97, pynini.Weight("tropical", 0), 1))
        with self.assertRaises(ValueError):
            export_fst(graph, self.root / "unsupported.bin")

    def test_float_tie_after_shared_merge(self) -> None:
        graph = pynini.Fst()
        for _ in range(4):
            graph.add_state()
        graph.set_start(0)
        graph.set_final(3)
        for src, dst, ilabel, olabel, weight in [
            (0, 1, 97, 72, 10.0),
            (0, 2, 97, 76, 9.0),
            (2, 1, 0, 0, 0.0),
            (1, 3, 0, 0, 1e8),
        ]:
            graph.add_arc(
                src, pynini.Arc(ilabel, olabel, pynini.Weight("tropical", weight), dst)
            )
        path = self.root / "merge.bin"
        export_fst(graph, path)
        runtime = ByteTransducer(self.library, path)
        try:
            expected = pynini.shortestpath(pynini.accep("a") @ graph).string()
            self.assertEqual(expected, "L")
            self.assertEqual(runtime.rewrite("a")[0], expected)
        finally:
            runtime.close()

    def test_prepared_lookahead_arc_order(self) -> None:
        graph = pynini.Fst()
        for _ in range(5):
            graph.add_state()
        graph.set_start(0)
        graph.set_final(4)
        for src, dst, ilabel, olabel, weight in [
            (0, 1, 0, 0, 1.0),
            (1, 2, 0, 120, 0.0),
            (1, 3, 97, 121, 0.0),
            (2, 4, 97, 122, 0.0),
            (3, 4, 0, 113, 0.0),
        ]:
            graph.add_arc(
                src, pynini.Arc(ilabel, olabel, pynini.Weight("tropical", weight), dst)
            )
        path = self.root / "lookahead.bin"
        export_fst(graph, path)
        for mode, expected in [("transducer", "yq"), ("lookahead", "xz")]:
            runtime = ByteTransducer(self.library, path, mode=mode)
            try:
                self.assertEqual(runtime.rewrite("a")[0], expected)
            finally:
                runtime.close()

    def test_projected_output_against_top_rewrite(self) -> None:
        from pynini.lib import rewrite

        rng = random.Random(173)
        for _ in range(30):
            graph = pynini.Fst()
            for _ in range(10):
                graph.add_state()
            graph.set_start(0)
            graph.set_final(9)
            for src in range(9):
                for dst in range(src + 1, 10):
                    if dst == src + 1 or rng.random() < 0.2:
                        graph.add_arc(
                            src,
                            pynini.Arc(
                                rng.choice([0, 97, 98]),
                                rng.choice([0, 120, 121]),
                                pynini.Weight("tropical", rng.randrange(-3, 4) / 8),
                                dst,
                            ),
                        )
            path = self.root / "projected.bin"
            export_fst(graph, path)
            runtime = ByteTransducer(self.library, path, mode="output")
            try:
                for text in ["", "a", "b", "ab", "ba", "aaa", "abab"]:
                    try:
                        expected = rewrite.top_rewrite(text, graph)
                    except rewrite.Error:
                        expected = None
                    actual = runtime.rewrite(text)
                    with self.subTest(text=text):
                        self.assertEqual(
                            None if actual is None else actual[0], expected
                        )
            finally:
                runtime.close()

    def test_lookahead_phase_merge_before_float_rounding(self) -> None:
        graph = pynini.Fst()
        for _ in range(5):
            graph.add_state()
        graph.set_start(0)
        for state, weight in [(0, 0), (1, 1), (4, 0)]:
            graph.set_final(state, pynini.Weight("tropical", weight))
        for src, dst, ilabel, olabel, weight in [
            (0, 4, 98, 120, -1e-8),
            (0, 2, 98, 122, 0),
            (2, 3, 98, 122, 0),
            (3, 1, 98, 121, -10),
            (4, 2, 0, 120, 0),
        ]:
            graph.add_arc(
                src, pynini.Arc(ilabel, olabel, pynini.Weight("tropical", weight), dst)
            )
        path = self.root / "phase-merge.bin"
        export_fst(graph, path)
        runtime = ByteTransducer(self.library, path, mode="lookahead")
        try:
            # The prepared OpenFst reference retains the cheaper prefix at the merge.
            # Compressing the next path first rounds both totals to the same float.
            self.assertEqual(runtime.rewrite("bbb"), ("xxzy", -9.0))
        finally:
            runtime.close()

    def test_compatibility_modes_reject_epsilon_cycles(self) -> None:
        graph = pynini.closure(pynini.cross("", "x")) + pynini.cross("a", "y")
        path = self.root / "cycle.bin"
        export_fst(graph, path)
        for mode in ["lookahead", "output"]:
            runtime = ByteTransducer(self.library, path, mode=mode)
            try:
                with self.assertRaisesRegex(RuntimeError, "Input-epsilon cycles"):
                    runtime.rewrite("a")
            finally:
                runtime.close()

    def test_random_exact_outputs(self) -> None:
        from pynini.lib import rewrite

        rng = random.Random(20261017)
        for case in range(1200):
            graph = pynini.Fst()
            n = rng.randrange(4, 15)
            for _ in range(n):
                graph.add_state()
            ids = list(range(n))
            if case % 2:
                rng.shuffle(ids)
            graph.set_start(ids[0])
            for pos, state in enumerate(ids):
                if pos == n - 1 or rng.random() < 0.15:
                    graph.set_final(
                        state,
                        pynini.Weight("tropical", rng.choice([0, 0, 0.125, 1, 2])),
                    )
                for later in range(pos + 1, n):
                    if later == pos + 1 or rng.random() < 0.25:
                        graph.add_arc(
                            state,
                            pynini.Arc(
                                rng.choice([0, 0, 97, 98]),
                                rng.choice([0, 120, 121, 122]),
                                pynini.Weight(
                                    "tropical",
                                    0
                                    if case >= 600
                                    else rng.choice(
                                        [0, 0, 0, -1e-8, 0.1, 0.2, 1e8, -10]
                                    ),
                                ),
                                ids[later],
                            ),
                        )
            if case >= 600:
                if case % 3 == 0:
                    target = graph.add_state()
                    graph.set_final(target)
                    graph.add_arc(
                        graph.start(),
                        pynini.Arc(99, 120, pynini.Weight("tropical", 1), target),
                    )
                if case % 2:
                    graph.properties(pynini.ACYCLIC, True)
            graph.arcsort("ilabel")
            path = self.root / "random-exact.bin"
            export_fst(graph, path)
            for mode in ["transducer", "output"]:
                runtime = ByteTransducer(self.library, path, mode=mode)
                try:
                    for text in ["", "a", "b", "ab", "ba", "aa", "bbb", "abab"]:
                        if mode == "output":
                            try:
                                expected = rewrite.top_rewrite(text, graph)
                            except rewrite.Error:
                                expected = None
                        else:
                            lattice = pynini.accep(text) @ graph
                            expected = (
                                pynini.shortestpath(lattice).string()
                                if lattice.start() >= 0
                                else None
                            )
                        actual = runtime.rewrite(text)
                        with self.subTest(case=case, mode=mode, text=text):
                            self.assertEqual(
                                None if actual is None else actual[0], expected
                            )
                finally:
                    runtime.close()

    def test_unweighted_reference_queue_order(self) -> None:
        # OpenFst AutoQueue chooses LIFO when acyclicity is not already known.
        graph = pynini.Fst()
        for _ in range(7):
            graph.add_state()
        graph.set_start(2)
        graph.set_final(4)
        for src, dst, ilabel, olabel in [
            (0, 1, 0, 122),
            (0, 4, 98, 122),
            (1, 4, 98, 120),
            (2, 0, 0, 120),
            (2, 1, 0, 121),
        ]:
            graph.add_arc(
                src, pynini.Arc(ilabel, olabel, pynini.Weight.one("tropical"), dst)
            )
        graph.arcsort("ilabel")
        expected = pynini.shortestpath(pynini.accep("b") @ graph).string()
        self.assertEqual(expected, "yx")
        path = self.root / "unweighted.bin"
        for known_acyclic in [False, True]:
            for unused_weighted_branch in [False, True]:
                variant = graph.copy()
                if unused_weighted_branch:
                    target = variant.add_state()
                    variant.set_final(target)
                    variant.add_arc(
                        variant.start(),
                        pynini.Arc(99, 120, pynini.Weight("tropical", 1), target),
                    )
                if known_acyclic:
                    variant.properties(pynini.ACYCLIC, True)
                expected = pynini.shortestpath(pynini.accep("b") @ variant).string()
                export_fst(variant, path)
                runtime = ByteTransducer(self.library, path)
                try:
                    self.assertEqual(runtime.rewrite("b")[0], expected)
                finally:
                    runtime.close()


if __name__ == "__main__":
    unittest.main()
