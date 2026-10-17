# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export tropical byte transducers and exercise the experimental native evaluator."""

from __future__ import annotations

import ctypes
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import pynini


def export_fst(fst: pynini.Fst, path: Path) -> None:
    """Export byte arcs and cached acyclicity used by reference queue selection."""
    import pynini

    if fst.arc_type() != "standard" or fst.weight_type() != "tropical":
        raise ValueError("Only standard tropical arcs are supported")
    states = list(fst.states())
    if states != list(range(len(states))) or fst.start() < 0:
        raise ValueError("Expected dense states and a valid start")
    arcs = []
    records = []
    for state in states:
        begin = len(arcs)
        for arc in sorted(fst.arcs(state), key=lambda a: a.ilabel):
            if not 0 <= arc.ilabel <= 255 or not 0 <= arc.olabel <= 255:
                raise ValueError("Only epsilon and byte labels are supported")
            weight = float(arc.weight)
            arcs.append((arc.ilabel, arc.olabel, arc.nextstate, weight))
        records.append((begin, len(arcs), float(fst.final(state))))
    with path.open("wb") as out:
        out.write(
            struct.pack(
                "<8sIIII",
                b"TNBYTE02",
                len(records),
                len(arcs),
                fst.start(),
                int(bool(fst.properties(pynini.ACYCLIC, False))),
            )
        )
        for record in records:
            out.write(struct.pack("<IIf", *record))
        for arc in arcs:
            out.write(struct.pack("<IIIf", *arc))


class ByteTransducer:
    """One reusable native workspace; calls on a handle must be sequential."""

    def __init__(
        self,
        library: Path,
        grammar: Path,
        *,
        mode: Literal["transducer", "lookahead", "output"] = "transducer",
    ) -> None:
        """Select direct, prepared-lookahead, or projected-output path semantics."""
        modes = {"transducer": 0, "lookahead": 1, "output": 2}
        if mode not in modes:
            raise ValueError(f"Unsupported evaluator mode: {mode}")
        self.lib = ctypes.CDLL(str(library.absolute()))
        self.lib.tn_load.argtypes = [ctypes.c_char_p]
        self.lib.tn_load.restype = ctypes.c_void_p
        self.lib.tn_delete.argtypes = [ctypes.c_void_p]
        self.lib.tn_delete.restype = None
        self.lib.tn_error.restype = ctypes.c_char_p
        self.lib.tn_rewrite.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.tn_rewrite.restype = ctypes.c_int
        self.lib.tn_set_mode.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.tn_set_mode.restype = ctypes.c_int
        self.handle = self.lib.tn_load(str(grammar.absolute()).encode())
        if not self.handle:
            raise RuntimeError(self.lib.tn_error().decode())
        if self.lib.tn_set_mode(self.handle, modes[mode]):
            self.close()
            raise RuntimeError(self.lib.tn_error().decode())

    def close(self) -> None:
        """Release native memory."""
        if self.handle:
            self.lib.tn_delete(self.handle)
            self.handle = None

    def rewrite(self, text: str) -> tuple[str, float] | None:
        """Return a minimum-cost output, or None if no accepting path exists."""
        if not self.handle:
            raise RuntimeError("Transducer is closed")
        encoded = text.encode()
        output = ctypes.c_void_p()
        size = ctypes.c_size_t()
        cost = ctypes.c_float()
        status = self.lib.tn_rewrite(
            self.handle,
            encoded,
            len(encoded),
            ctypes.byref(output),
            ctypes.byref(size),
            ctypes.byref(cost),
        )
        if status < 0:
            raise RuntimeError(self.lib.tn_error().decode())
        if status:
            return None
        return ctypes.string_at(output.value, size.value).decode(), cost.value


class NativePipeline:
    """Native classifier, tag parser, reordering, verbalizer and postprocessing FST.

    normalize includes English punctuation handling and Unicode whitespace collapse.
    normalize_core exposes the transducer stages for compatibility diagnostics.
    """

    def __init__(self, library: Path, grammar_directory: Path) -> None:
        self.lib = ctypes.CDLL(str(library.absolute()))
        self.lib.tn_error.restype = ctypes.c_char_p
        self.lib.tn_pipeline_load.argtypes = [ctypes.c_char_p] * 3
        self.lib.tn_pipeline_load.restype = ctypes.c_void_p
        self.lib.tn_pipeline_delete.argtypes = [ctypes.c_void_p]
        self.lib.tn_pipeline_delete.restype = None
        self.lib.tn_pipeline_rewrite.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.lib.tn_pipeline_rewrite.restype = ctypes.c_int
        self.lib.tn_pipeline_normalize.argtypes = self.lib.tn_pipeline_rewrite.argtypes
        self.lib.tn_pipeline_normalize.restype = ctypes.c_int
        paths = [
            (grammar_directory / (name + ".bin")).absolute() for name in ("tagger", "verbalizer", "postprocessor")
        ]
        self.handle = self.lib.tn_pipeline_load(*(str(path).encode() for path in paths))
        if not self.handle:
            raise RuntimeError(self.lib.tn_error().decode())

    def close(self) -> None:
        """Release native grammar and workspace storage."""
        if self.handle:
            self.lib.tn_pipeline_delete(self.handle)
            self.handle = None

    def normalize_core(self, text: str) -> str | None:
        """Normalize preprocessed UTF-8 input with no Python work between stages."""
        return self._normalize(text, complete=False)

    def normalize(self, text: str) -> str | None:
        """Run all English normalization stages, including punctuation, in native code."""
        return self._normalize(text, complete=True)

    def normalize_batch(self, texts: list[str]) -> list[str]:
        """Run a bounded batch in native code and fail without returning partial results.

        Embedded NUL characters are unsupported, including inside otherwise valid text.
        A failing record raises RuntimeError with its zero-based batch index.
        The handle remains usable after failure; no partial result is returned.
        Corpus callers should explicitly reject and count malformed records before batching.
        """
        if not self.handle:
            raise RuntimeError("Pipeline is closed")
        encoded = [text.encode() for text in texts]
        frames = b"".join(struct.pack("<I", len(text)) + text for text in encoded)
        function = self.lib.tn_pipeline_normalize_batch
        function.argtypes = self.lib.tn_pipeline_rewrite.argtypes
        function.restype = ctypes.c_int
        output, size = ctypes.c_void_p(), ctypes.c_size_t()
        if function(self.handle, frames, len(frames), ctypes.byref(output), ctypes.byref(size)):
            raise RuntimeError(self.lib.tn_error().decode())
        data = ctypes.string_at(output.value, size.value)
        result, position = [], 0
        while position < len(data):
            if len(data) - position < 4:
                raise RuntimeError("Truncated native batch header")
            length = struct.unpack_from("<I", data, position)[0]
            position += 4
            if length > len(data) - position:
                raise RuntimeError("Truncated native batch record")
            result.append(data[position : position + length].decode())
            position += length
        if len(result) != len(texts):
            raise RuntimeError("Native batch record count changed")
        return result

    def _normalize(self, text: str, *, complete: bool) -> str | None:
        if not self.handle:
            raise RuntimeError("Pipeline is closed")
        encoded = text.encode()
        output = ctypes.c_void_p()
        size = ctypes.c_size_t()
        function = self.lib.tn_pipeline_normalize if complete else self.lib.tn_pipeline_rewrite
        status = function(self.handle, encoded, len(encoded), ctypes.byref(output), ctypes.byref(size))
        if status < 0:
            raise RuntimeError(self.lib.tn_error().decode())
        if status:
            return None
        return ctypes.string_at(output.value, size.value).decode()
