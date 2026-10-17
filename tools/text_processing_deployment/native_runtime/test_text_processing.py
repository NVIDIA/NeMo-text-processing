# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Differential tests for native English text stages against their Python references."""

import ctypes
import logging
import random
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

import pynini
from sacremoses import MosesDetokenizer

from nemo_text_processing.text_normalization.data_loader_utils import post_process_punct, pre_process

from .byte_transducer import NativePipeline, export_fst


class TextProcessingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_logging_disable = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        source = Path(__file__).parent
        cls.library = cls.root / "native.so"
        cls.scalar = cls.root / "scalar.so"
        for library, flags in ((cls.library, []), (cls.scalar, ["-DTN_DISABLE_SIMD"])):
            subprocess.run(
                [
                    "g++",
                    "-std=c++17",
                    "-O3",
                    "-Wall",
                    "-Wextra",
                    "-shared",
                    "-fPIC",
                    *flags,
                    str(source / "byte_transducer.cc"),
                    "-o",
                    str(library),
                ],
                check=True,
            )
        cls.libs = [ctypes.CDLL(str(p)) for p in (cls.library, cls.scalar)]
        for lib in cls.libs:
            lib.tn_text_stage.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_size_t,
                ctypes.c_char_p,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_size_t),
            ]
            lib.tn_text_stage.restype = ctypes.c_int
        cls.detokenizer = MosesDetokenizer(lang="en")
        cls.executable = cls.root / "normalize_stream"
        subprocess.run(
            [
                "g++",
                "-std=c++17",
                "-O3",
                str(source / "byte_transducer.cc"),
                str(source / "normalize_stream.cc"),
                "-o",
                str(cls.executable),
            ],
            check=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()
        logging.disable(cls.previous_logging_disable)

    def stage(self, stage: int, written: str, output: str) -> str:
        a, b = written.encode(), output.encode()
        results = []
        for lib in self.libs:
            result, length = ctypes.c_void_p(), ctypes.c_size_t()
            status = lib.tn_text_stage(stage, a, len(a), b, len(b), ctypes.byref(result), ctypes.byref(length))
            self.assertEqual(status, 0)
            results.append(ctypes.string_at(result.value, length.value).decode())
        self.assertEqual(results[0], results[1], "SIMD/scalar mismatch")
        return results[0]

    def check_pair(self, written: str, output: str) -> None:
        written = " ".join(written.split())
        self.assertEqual(self.stage(0, written, ""), pre_process(written).strip())
        detokenized = self.detokenizer.detokenize([output], unescape=False)
        self.assertEqual(self.stage(1, "", output), detokenized)
        expected = " ".join(post_process_punct(input=written, normalized_text=detokenized).split())
        self.assertEqual(self.stage(2, written, output), expected)

    def test_public_punctuation_and_unicode_cases(self) -> None:
        pairs = [
            ("[12] test' example", "[ twelve ] test ' example"),
            ("The Jones' house.", "The Jones ' house ."),
            (chr(96) * 2 + "A test''", "\" A test ' '"),
            ("a:b:c", "a : b c"),
            ("a:: b", "a : : b"),
            ("a [x] b", "a [ x ] b"),
            ("a\u00a0b\u2003c", "a\u00a0b\u2003c"),
            ("$5.00 and €3", "$ five . zero zero and € three"),
            ("你好 世界 한 국", "你好 世界 한 국"),
            ("a @-@ b @-@ c", "a @-@ b @-@ c"),
            ("' é ' 中文", "' é ' 中文"),
            ("'\U00010400", "'\U00010400"),
            ("", ""),
        ]
        for a, b in pairs:
            with self.subTest(written=a, output=b):
                self.check_pair(a, b)

    def test_seeded_text_stage_equivalence(self) -> None:
        rng = random.Random(914)
        units = [
            "word",
            "James",
            "café",
            "你好",
            "국",
            "'s",
            "'é",
            "a",
            "",
            " ",
            "\t",
            "\u00a0",
            "\u2003",
            "[",
            "]",
            "(",
            ")",
            "$",
            "€",
            "¿",
            "¡",
            ":",
            ";",
            ".",
            ",",
            "!",
            "?",
            "\\",
            "%",
            "'",
            '"',
            chr(96) * 2,
            "''",
            "„",
            "“",
            "”",
            "a:b",
            "@-@",
            "\U00010400",
            "\u1100",
            "\uffdc",
            "\U0002ffff",
        ]
        for _ in range(2000):
            a = " ".join(rng.choices(units, k=rng.randrange(30)))
            b = " ".join(rng.choices(units, k=rng.randrange(30)))
            with self.subTest(written=a, output=b):
                self.check_pair(a, b)

    def test_invalid_utf8(self) -> None:
        for value in (
            b"\xc0\x80",
            b"\xed\xa0\x80",
            b"\xf4\x90\x80\x80",
            b"\xff",
            b"\xe2\x82",
        ):
            for lib in self.libs:
                result, length = ctypes.c_void_p(), ctypes.c_size_t()
                status = lib.tn_text_stage(
                    0,
                    value,
                    len(value),
                    b"",
                    0,
                    ctypes.byref(result),
                    ctypes.byref(length),
                )
                self.assertEqual(status, -1)

    def test_stream_and_complete_api(self) -> None:
        export_fst(pynini.cross("x", 'tokens { name: "word" }'), self.root / "tagger.bin")
        export_fst(
            pynini.cross(' tokens { name: "word"  } ', "word"),
            self.root / "verbalizer.bin",
        )
        export_fst(pynini.accep("word"), self.root / "postprocessor.bin")
        values = ["x", " \nx\t", "", "\u2003x\u00a0"]
        native = NativePipeline(self.library, self.root)
        try:
            self.assertEqual([native.normalize(s) for s in values], ["word", "word", "", "word"])
            self.assertEqual(native.normalize_batch(values), ["word", "word", "", "word"])
            self.assertEqual(native.normalize_batch([]), [])
            with self.assertRaisesRegex(RuntimeError, "Batch record 1"):
                native.normalize_batch(["x", "unmatched", "x"])
            self.assertEqual(native.normalize_batch(["x"]), ["word"])
            for malformed in [b"\x02\x00", struct.pack("<I", 5) + b"x"]:
                output, size = ctypes.c_void_p(1), ctypes.c_size_t(1)
                status = native.lib.tn_pipeline_normalize_batch(
                    native.handle,
                    malformed,
                    len(malformed),
                    ctypes.byref(output),
                    ctypes.byref(size),
                )
                self.assertEqual(status, -1)
                self.assertIsNone(output.value)
                self.assertEqual(size.value, 0)
            self.assertEqual(native.normalize_batch(values), ["word", "word", "", "word"])

        finally:
            native.close()
        source = self.root / "input.frames"
        target = self.root / "output.frames"
        source.write_bytes(b"".join(struct.pack("<I", len(s.encode())) + s.encode() for s in values))
        subprocess.run([str(self.executable), str(self.root), str(source), str(target)], check=True)
        expected = b"".join(struct.pack("<I", len(s)) + s for s in [b"word", b"word", b"", b"word"])
        self.assertEqual(target.read_bytes(), expected)
        source.write_bytes(b"\x02\x00")
        result = subprocess.run(
            [str(self.executable), str(self.root), str(source), str(target)],
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"Truncated input frame header", result.stderr)

    def test_embedded_nul_matches_reference(self) -> None:
        from types import SimpleNamespace

        from nemo_text_processing.text_normalization.normalize import Normalizer
        from nemo_text_processing.text_normalization.token_parser import TokenParser

        values = ["a" * n + "\0word" for n in (0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65)]
        values += ["René\0 word", "x\0", "\u2003\0\u00a0", "\0", " \0 \0 ", "[\0x]", "x \0 y"]
        normalizer = Normalizer.__new__(Normalizer)
        normalizer.lang = "en"
        normalizer.parser = TokenParser()
        normalizer.max_number_of_permutations_per_split = 729
        normalizer.moses_detokenizer = self.detokenizer
        prepared = {pre_process(" ".join(value.split())).strip().replace("\0", "") for value in values}
        normalizer.tagger = SimpleNamespace(
            fst=pynini.union(*(pynini.cross(pynini.escape(value), 'tokens { name: "word" }') for value in prepared))
        )
        normalizer.verbalizer = SimpleNamespace(fst=pynini.cross(' tokens { name: "word"  } ', "word"))
        normalizer.post_processor = SimpleNamespace(fst=pynini.accep("word"))
        # Also check the early-return path, which must retain the original NUL.
        for failed_verbalization in (False, True):
            if failed_verbalization:
                normalizer.verbalizer.fst = pynini.cross("unreachable", "word")
            for name in ("tagger", "verbalizer", "post_processor"):
                filename = "postprocessor" if name == "post_processor" else name
                export_fst(getattr(normalizer, name).fst, self.root / (filename + ".bin"))
            expected = [
                normalizer.normalize(" ".join(value.split()), punct_pre_process=True, punct_post_process=True)
                for value in values
            ]
            for library in (self.library, self.scalar):
                native = NativePipeline(library, self.root)
                try:
                    with self.subTest(library=library.name, failed_verbalization=failed_verbalization):
                        self.assertEqual([native.normalize(value) for value in values], expected)
                        self.assertEqual(native.normalize_batch(values), expected)
                        self.assertEqual(native.normalize_batch(values), expected)
                finally:
                    native.close()

    def test_failed_verbalization_matches_reference(self) -> None:
        from types import SimpleNamespace

        from nemo_text_processing.text_normalization.normalize import Normalizer
        from nemo_text_processing.text_normalization.token_parser import TokenParser

        normalizer = Normalizer.__new__(Normalizer)
        normalizer.lang = "en"
        normalizer.parser = TokenParser()
        normalizer.max_number_of_permutations_per_split = 729
        normalizer.moses_detokenizer = self.detokenizer
        normalizer.verbalizer = SimpleNamespace(fst=pynini.cross("unreachable", "word"))
        normalizer.post_processor = SimpleNamespace(fst=pynini.accep("word"))
        for tags in [" ", 'tokens { name: "word" }', 'tokens { name: "" }']:
            normalizer.tagger = SimpleNamespace(fst=pynini.cross(pynini.escape("[ x ]"), tags))
            for name, component in [
                ("tagger", normalizer.tagger),
                ("verbalizer", normalizer.verbalizer),
                ("postprocessor", normalizer.post_processor),
            ]:
                export_fst(component.fst, self.root / (name + ".bin"))
            native = NativePipeline(self.library, self.root)
            try:
                expected = normalizer.normalize("[x]", punct_pre_process=True, punct_post_process=True)
                self.assertEqual(native.normalize("[x]"), expected)
            finally:
                native.close()


if __name__ == "__main__":
    unittest.main()
