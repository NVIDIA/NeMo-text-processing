# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Thin Python wrapper around Sparrowhawk's ``normalizer_main`` binary.

``normalizer_main`` is an interactive C++ tool: it loads a pair of compiled WFST
grammars (``classify/tokenize_and_classify.far`` + ``verbalize/verbalize.far``)
once at start-up, then reads one utterance per line from *stdin* and writes the
normalized result — one line — to *stdout*.

We keep the process alive so the (slow) grammar load happens only once, and talk
to it over its stdin/stdout pipes. Two things make this reliable:

* **Line buffering.** A C++ program writing to a pipe is block-buffered by
  default, so its single line of output would never be flushed until the buffer
  fills — deadlocking a request/response loop. We launch it under ``stdbuf -oL``
  (configurable) to force line buffering. See ``STDBUF_PREFIX``.
* **stderr is drained separately.** Sparrowhawk logs diagnostics to stderr; we
  read those on a background thread so they can never fill the pipe and block the
  child, and so stdout carries *only* the normalized text.
"""

import logging
import os
import queue
import shlex
import subprocess
import threading
from typing import List, Optional

log = logging.getLogger("sparrowhawk")


class NormalizerError(RuntimeError):
    pass


class NormalizerWorker:
    """A single long-running ``normalizer_main`` process for one direction."""

    def __init__(self, cmd: List[str], cwd: str, io_timeout: float = 15.0):
        self.cmd = cmd
        self.cwd = cwd
        self.io_timeout = io_timeout
        self._lock = threading.Lock()
        self.proc: Optional[subprocess.Popen] = None
        self._out_q: "queue.Queue[Optional[str]]" = queue.Queue()
        self._start()

    # -- process lifecycle -------------------------------------------------
    def _start(self) -> None:
        log.info("starting: %s (cwd=%s)", " ".join(self.cmd), self.cwd)
        # A fresh queue per process so a crash/restart can never leave a stale
        # line behind that would desync request N with response N-1.
        self._out_q = queue.Queue()
        self.proc = subprocess.Popen(
            self.cmd,
            cwd=self.cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,  # line-buffered on our (write) side
        )
        threading.Thread(target=self._pump_stdout, args=(self.proc,), daemon=True).start()
        threading.Thread(target=self._drain_stderr, args=(self.proc,), daemon=True).start()

    def _pump_stdout(self, proc: subprocess.Popen) -> None:
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                self._out_q.put(line)
        finally:
            self._out_q.put(None)  # sentinel: process closed stdout / died

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        for line in proc.stderr:  # type: ignore[union-attr]
            log.debug("[normalizer stderr] %s", line.rstrip())

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.close()  # type: ignore[union-attr]
            except Exception:
                pass
            self.proc.terminate()

    # -- request/response --------------------------------------------------
    def normalize(self, text: str) -> str:
        # One request is one line, so collapse any embedded newlines.
        text = text.replace("\r", " ").replace("\n", " ").strip()
        if not text:
            return ""
        with self._lock:
            try:
                return self._exchange(text)
            except (BrokenPipeError, ValueError, NormalizerError) as e:
                # Restart once and retry — covers a crashed/closed child.
                log.warning("normalizer exchange failed (%s); restarting worker", e)
                self._start()
                return self._exchange(text)

    def _exchange(self, text: str) -> str:
        if not self.alive():
            self._start()
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()
        try:
            line = self._out_q.get(timeout=self.io_timeout)
        except queue.Empty:
            self._start()  # reset to a known-good state
            raise NormalizerError(f"timed out after {self.io_timeout}s waiting for output")
        if line is None:
            raise NormalizerError("normalizer_main closed its output (process died)")
        return line.strip()


class EnginePool:
    """A fixed pool of workers for one direction, enabling real concurrency.

    Each ``normalizer_main`` process is single-threaded and serializes requests,
    so throughput scales with the number of workers. ``normalize`` checks a
    worker out of a blocking queue (natural backpressure) and returns it after.
    """

    def __init__(self, name: str, cmd: List[str], cwd: str, size: int = 2, io_timeout: float = 15.0):
        self.name = name
        self._q: "queue.Queue[NormalizerWorker]" = queue.Queue()
        self.workers = [NormalizerWorker(cmd, cwd, io_timeout=io_timeout) for _ in range(max(1, size))]
        for w in self.workers:
            self._q.put(w)

    def normalize(self, text: str) -> str:
        w = self._q.get()
        try:
            return w.normalize(text)
        finally:
            self._q.put(w)

    def healthy(self) -> bool:
        return all(w.alive() for w in self.workers)

    def close(self) -> None:
        for w in self.workers:
            w.close()


def build_cmd(binary: str, config_name: str, stdbuf_prefix: str) -> List[str]:
    """Assemble the argv for launching ``normalizer_main`` under stdbuf."""
    prefix = shlex.split(stdbuf_prefix) if stdbuf_prefix.strip() else []
    return prefix + [binary, f"--config={config_name}"]


def pool_from_env(name: str, cwd: str) -> EnginePool:
    """Create an :class:`EnginePool` for ``name`` ('tn'/'itn') rooted at ``cwd``."""
    binary = os.environ.get("SPARROWHAWK_BIN", "normalizer_main")
    config_name = os.environ.get("SPARROWHAWK_CONFIG", "sparrowhawk_configuration.ascii_proto")
    stdbuf_prefix = os.environ.get("STDBUF_PREFIX", "stdbuf -oL -eL")
    workers = int(os.environ.get("WORKERS_PER_DIRECTION", "2"))
    io_timeout = float(os.environ.get("IO_TIMEOUT", "15"))
    cmd = build_cmd(binary, config_name, stdbuf_prefix)
    return EnginePool(name, cmd, cwd, size=workers, io_timeout=io_timeout)
