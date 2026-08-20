# nemo-fst

Applies prebuilt OpenFst grammars. It does not compile them — that is pynini's
job and stays pynini's job.

The point is `olabel_lookahead` composition: OpenFst's `LabelLookAheadMatcher`
refuses to expand a tagger branch that cannot consume the next byte. Roughly
98% of what a plain composition against `tokenize_and_classify` builds is
discarded immediately as unreachable, because every semiotic class is live at
once and a single digit opens about eight of them. Skipping those branches is
worth **4.8–11.5x** on tagging, and the composition **releases the GIL**, which
`pywrapfst` does not.

pynini as published cannot reach this: its wheels ship no lookahead plugin and
register no lookahead FST type.

## Scope

The tagger only. Tagging is 96.7% of text-normalization runtime, so this is
effectively the whole speedup with the smallest native surface that gets it.
The verbalizer, post-processor and the `Normalizer` integration are separate.

```python
import nemo_fst

nemo_fst.has_lookahead()                       # -> True

tagger = nemo_fst.Tagger.from_far(
    "en_tn_True_deterministic_cased__tokenize.far",
    key="tokenize_and_classify",
    cache_dir="~/.cache/nemo_fst",
)
tagger.tag("It costs $25.50.")                 # -> 'tokens { money { ... } }'
```

`from_far` prepares a lookahead artifact the first time it sees a FAR and caches
it under `cache_dir`, keyed on a hash of the source. Preparation is ~0.25 s; the
artifact is ~13 MB against the FAR's 7.5 MB and *loads faster*, because the
lookahead type is a `ConstFst` with contiguous arc storage.

## Building

Needs an OpenFst built with `--enable-lookahead-fsts`. **No distribution ships
one** -- Homebrew's `openfst` and the usual Linux packages are all built without
it -- so build it first:

```bash
bash scripts/build_openfst.sh $HOME/.local/openfst
export OPENFST_PREFIX=$HOME/.local/openfst

uv sync            # or: pip install .
```

Linux and macOS. The linkers spell "export nothing but the module init symbol"
differently, so the build offers every spelling and keeps the ones the toolchain
accepts, rather than guessing from the platform.

OpenFst 1.8.4. It reads grammars pynini wrote with 1.8.3 -- the formats are
unchanged, and the tests run against a 1.8.3-written FAR. 1.8.3 itself does not
compile under Clang, which is why it is not supported here.

**macOS is untested.** The platform handling is written but no macOS machine was
available to run it on; treat the first build there as the real test.

`scripts/build_openfst.sh` produces such a prefix, and cibuildwheel runs it once
per container so every wheel links the same OpenFst. When the prefix has static
archives they are linked in, leaving the extension with no external OpenFst
dependency and no rpath — 1.3 MB, and relocatable. A shared-only prefix still
works for local development, with an rpath pointing back at it.

## Testing

From this directory, with `OPENFST_PREFIX` set:

```bash
uv run pytest --tn_cache_dir=/path/to/grammars
```

That runs the 20 tests that need neither pynini nor a compiled grammar --
`test_standalone.py` and `test_cache.py`, which use the ~10 KB toy grammars in
`tests/data/`. They are also what verifies a built wheel: pynini publishes
manylinux x86_64 wheels only, and a wheel-test container has no grammar cache
either, so anything needing the English tagger would silently skip there.

The rest compare against pynini and need `pynini` and `nemo_text_processing`
present; without them they skip, saying so. For the full 37, use an environment
that has both -- the repository's own, with this package built into it:

```bash
OPENFST_PREFIX=$PREFIX ./build.sh          # builds in place
python -m pytest --tn_cache_dir=/path/to/grammars
```

The tests run in order of consequence. `test_coexistence.py` is first on
purpose: this package and pynini each carry their own OpenFst into the same
interpreter, and if hidden visibility is wrong nothing below it means anything.

`test_integration.py` substitutes the tagging step on `Normalizer` at runtime
and compares the end of the whole pipeline against the stock one, so the
package is exercised through the code that will eventually call it.

Benchmarks are deselected by default:

```bash
uv run --package nemo-fst pytest -m benchmark -s --tn_cache_dir=/path/to/grammars
```

On one aarch64 machine, against the English test corpus:

| | |
| --- | --- |
| tagging | 4.6–11.2x |
| `normalize()` end to end | 4.1–8.6x on scripts, 2.6x on single-token inputs |
| 8 threads through one `Tagger` | 8.3x over sequential |
| preparing an artifact | 0.28 s once, 0.02 s from cache |

Both speedups fall with semiotic density — lookahead removes *dead* hypotheses,
and dense text keeps more genuinely alive — and rise with length, because
composition is superlinear and the pipeline's non-tagging stages are a fixed
cost per call. Tagging is 91% of `normalize()` on a ten-character input and 95%
on a paragraph, which is what bounds the end-to-end column.

## Two things that will bite anyone editing this

**Composition consults lookahead data through its left operand only.** The
tagger naturally sits on the right, so the inverses are composed and the result
inverted: `A ∘ B = (B⁻¹ ∘ A⁻¹)⁻¹`. Get it backwards and you get a zero-state
result, or silently wrong output.

**The relabelling map does not survive `Write`/`Read`.** The conversion
renumbers the labels it indexes, and `LabelReachableData::label2index_` is not
serialized. A reloaded artifact still composes — its interval sets are intact —
but `RelabelPairs` returns an *empty* map, which is an identity relabelling,
which produces output that parses and is garbage. The map is therefore written
beside the artifact, renamed into place before the FST, and an entry missing its
map counts as a cache miss.

## Using it from nemo_text_processing

Used automatically once installed:

```bash
pip install nemo_text_processing[runtime]
```

`fast_tagger=False` or `NEMO_FAST_TAGGER=0` forces the pynini path. Being unable
to use it -- not installed, no compiled grammar, an OpenFst without lookahead --
falls back to pynini, loudly if `fast_tagger=True` asked for it and quietly if
nobody did.

Where the grammar admits two readings at the same cost the two implementations
may return different ones. That is a property of the grammar rather than of
either implementation, and neither reading is more correct, so the test corpus
records both. What the differential test pins is the property that does matter:
this never returns a costlier parse than pynini.
