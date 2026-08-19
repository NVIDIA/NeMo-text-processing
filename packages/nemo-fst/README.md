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

Needs an OpenFst built with `--enable-lookahead-fsts`:

```bash
./configure --prefix=$PREFIX --enable-grm --enable-far --enable-lookahead-fsts
make && make install

OPENFST_PREFIX=$PREFIX pip install .
```

`scripts/build_openfst.sh` produces such a prefix, and cibuildwheel runs it once
per container so every wheel links the same OpenFst. When the prefix has static
archives they are linked in, leaving the extension with no external OpenFst
dependency and no rpath — 1.3 MB, and relocatable. A shared-only prefix still
works for local development, with an rpath pointing back at it.

## Testing

```bash
OPENFST_PREFIX=$PREFIX uv sync --package nemo-fst
uv run --package nemo-fst pytest --tn_cache_dir=/path/to/grammars
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

Off unless asked for:

```python
Normalizer(input_case="cased", lang="en", cache_dir=..., fast_tagger=True)
```

or `NEMO_FAST_TAGGER=1`. Without it, or when this package is missing, tagging
stays on pynini and the caller sees a warning only if they asked for the fast
path and did not get it.

It is opt-in rather than automatic because the tagger's shortest path is not
unique. Where the grammar admits two readings at the same cost the two
implementations may return different ones, and a few of the repository's own
tests pin the reading pynini happens to produce. Opting in accepts that; it does
not mean a costlier parse, which the differential test rules out.
