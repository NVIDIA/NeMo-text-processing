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

"""nemo-text-processing-lightning — apply prebuilt OpenFst grammars, fast, without pynini.

Scope is the tagger.  Tagging is 96.7% of text-normalization runtime, so this
is effectively the whole speedup with the smallest possible native surface.

    import nemo_text_processing_lightning

    nemo_text_processing_lightning.has_lookahead()                    # -> True
    tagger = nemo_text_processing_lightning.Tagger.from_far(
        "en_tn_True_deterministic_cased__tokenize.far",
        key="tokenize_and_classify",
        cache_dir="~/.cache/nemo_text_processing_lightning",
    )
    tagger.tag("It costs $25.50.")               # GIL released for the compose

`from_far` prepares the lookahead artifact on first use -- invert the tagger,
convert it to `olabel_lookahead` -- and caches it under `cache_dir`, keyed on a
hash of the source FAR so a regenerated grammar invalidates it.  Preparation is
~0.35 s; the artifact loads faster than the FAR it came from.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import _lightning
from ._lightning import __openfst_version__, has_lookahead

__all__ = ["Tagger", "has_lookahead", "default_cache_dir", "__openfst_version__"]

__version__ = "0.1.0.dev0"


def default_cache_dir() -> Path:
    """Where prepared artifacts land when the caller does not say.

    `NEMO_TPL_CACHE_DIR`, else `$XDG_CACHE_HOME/nemo_text_processing_lightning`, else
    `~/.cache/nemo_text_processing_lightning`.
    """
    env = os.environ.get("NEMO_TPL_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "nemo_text_processing_lightning"


class Tagger:
    """A prepared tagger grammar.  Construct with :meth:`from_far`."""

    __slots__ = ("_impl",)

    def __init__(self, impl):
        self._impl = impl

    @classmethod
    def from_far(cls, far_path, key: str = "tokenize_and_classify",
                 cache_dir=None) -> Tagger:
        """Read `key` out of the FAR and prepare (or reuse) its lookahead form.

        `cache_dir` is created if missing.  Pass `cache_dir=False` to prepare in
        memory every time and never touch the disk.
        """
        far_path = Path(far_path).expanduser()
        if not far_path.exists():
            raise FileNotFoundError(str(far_path))
        if cache_dir is False:
            resolved = ""
        else:
            directory = Path(cache_dir).expanduser() if cache_dir is not None \
                else default_cache_dir()
            directory.mkdir(parents=True, exist_ok=True)
            resolved = str(directory)
        return cls(_lightning.Tagger.from_far(str(far_path), key, resolved))

    def tag(self, text: str) -> str:
        """Tag `text`, returning the tagged string.

        Takes raw text: the acceptor is built from UTF-8 bytes, so pynini's
        `escape()` (which only exists to get past its string-compiler syntax) is
        neither needed nor wanted here.
        """
        return self._impl.tag(text)

    __call__ = tag

    @property
    def num_states(self) -> int:
        return self._impl.num_states

    @property
    def artifact_path(self) -> str:
        """Path of the cached prepared artifact ("" when cache_dir=False)."""
        return self._impl.artifact_path

    @property
    def prepared(self) -> bool:
        """True if this instance built the artifact; False if it reused a cached one."""
        return self._impl.prepared

    @property
    def relabel_pairs(self) -> dict:
        """The label renumbering the `olabel_lookahead` conversion applied.

        Exposed for tests only.  Nothing in the tag path needs it in Python --
        that is the point of doing this in C++.
        """
        return dict(self._impl.relabel_pairs)

    def __repr__(self) -> str:
        return f"<nemo_text_processing_lightning.Tagger {self.num_states} states>"
