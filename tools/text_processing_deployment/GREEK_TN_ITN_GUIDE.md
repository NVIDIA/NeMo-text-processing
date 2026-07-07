# Greek (el) Text Normalization & Inverse Text Normalization — Usage & Deployment

This guide covers using the Greek TN/ITN grammars from Python and deploying them as a
C++ [Sparrowhawk](https://github.com/google/sparrowhawk) service on a **Linux x86_64** box.

- **TN (Text Normalization)** turns written text into its spoken form, for TTS
  (`25/12/2024` → `είκοσι πέντε Δεκεμβρίου δύο χιλιάδες είκοσι τέσσερα`).
- **ITN (Inverse Text Normalization)** turns spoken/ASR output into written form
  (`δεκατέσσερα Ιουλίου` → `14 Ιουλίου`, `δέκα μέτρα` → `10 m`).

---

## 1. Prerequisites

- Linux x86_64 (native — no emulation needed; the Sparrowhawk image installs `thrax`,
  which conda-forge only ships for `x86_64`/`linux-64`, **not** arm64).
- [Miniconda/Anaconda](https://docs.conda.io/en/latest/miniconda.html).
- For the C++ deployment: Docker Engine (`docker` + a running daemon).

---

## 2. Install the Python package (from this repo)

The Greek grammars ship with this source tree, so install from source (not the PyPI wheel):

```bash
git clone <this-repo-url> NeMo-text-processing
cd NeMo-text-processing
git checkout feat/greek-tn-itn        # or the branch/tag that contains Greek

conda create -y -n nemo_tn python=3.10
conda activate nemo_tn
conda install -y -c conda-forge pynini=2.1.6.post1
pip install -e .
```

Verify:

```bash
python -c "from nemo_text_processing.text_normalization.normalize import Normalizer; \
print(Normalizer(lang='el', input_case='cased').normalize('3,14 και 21ος', verbose=False))"
# -> τρία κόμμα δεκατέσσερα και εικοστός πρώτος
```

---

## 3. Use it from Python

### Text normalization (for TTS)

```python
from nemo_text_processing.text_normalization.normalize import Normalizer

# input_case='cased'  -> keeps capitalization (recommended for real text)
# input_case='lower_cased' -> assume input is already lowercased
# cache_dir caches compiled grammars as .far (first build ~60s, then instant)
tn = Normalizer(lang='el', input_case='cased', cache_dir='el_grammars')

tn.normalize('Στις 25/12/2024 πλήρωσα 15,50€', verbose=False)
# 'Στις είκοσι πέντε Δεκεμβρίου δύο χιλιάδες είκοσι τέσσερα πλήρωσα δεκαπέντε ευρώ και πενήντα λεπτά'
```

### Inverse text normalization (for ASR output)

```python
from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer

itn = InverseNormalizer(lang='el', cache_dir='el_grammars')
itn.inverse_normalize('δέκα μέτρα', verbose=False)   # '10 m'
```

> **Cache invalidation:** after editing any grammar, delete `cache_dir` or pass
> `overwrite_cache=True`, otherwise the stale `.far` is reused.

---

## 4. Use it from the command line

```bash
# TN — single string
python nemo_text_processing/text_normalization/normalize.py \
    --language el --input_case cased --text "3,14 και 21ος"

# TN — a file (one sentence per line), parallelized
python nemo_text_processing/text_normalization/normalize.py \
    --language el --input_case cased --input_file in.txt --output_file out.txt --n_jobs 8

# ITN
python nemo_text_processing/inverse_text_normalization/inverse_normalize.py \
    --language el --text "δεκατέσσερα Ιουλίου"
```

---

## 5. Run the test suite (optional)

```bash
python -m pytest tests/nemo_text_processing/el/ -m "not pleasefixme" --cpu -q
```

---

## 6. Deploy as a Sparrowhawk C++ service

Sparrowhawk runs the compiled `.far` grammars in C++ without a Python runtime — suitable for
production serving. Everything below runs from `tools/text_processing_deployment/`.

```bash
cd tools/text_processing_deployment
```

### 6.1 Export the grammars (.far)

The exported grammars are already committed under
`el_tn_grammars_cased/` and `el_itn_grammars_cased/` (each with `classify/tokenize_and_classify.far`
and `verbalize/verbalize.far`). To regenerate them:

```bash
# TN
bash export_grammars.sh --GRAMMARS=tn_grammars  --LANGUAGE=el --INPUT_CASE=cased --MODE=export
# ITN
bash export_grammars.sh --GRAMMARS=itn_grammars --LANGUAGE=el --INPUT_CASE=cased --MODE=export
```

This calls `pynini_export.py` (which has an `el` branch registered) and writes
`<lang>_<grammars>_<input_case>/{classify,verbalize}/*.far`.

> If you run `pynini_export.py` directly instead of via the script, make sure
> `nemo_text_processing` is importable (`pip install -e .`, or prefix with
> `PYTHONPATH=<repo-root>`).

### 6.2 Build the Sparrowhawk Docker image

On native Linux x86_64 the image builds cleanly (compiles OpenFst, re2, thrax, protobuf and
Sparrowhawk from source — first build takes a while):

```bash
bash docker/build.sh            # produces image  sparrowhawk:latest
```

### 6.3 Interactive shell (smoke test)

`export_grammars.sh --MODE=interactive` builds the image (if needed), mounts the grammars, and
drops you into the container:

```bash
# Text normalization
bash export_grammars.sh --GRAMMARS=tn_grammars --LANGUAGE=el --INPUT_CASE=cased --MODE=interactive
# inside the container:
echo "Στις 25/12/2024 πλήρωσα 15,50€" | \
    normalizer_main --config=sparrowhawk_configuration.ascii_proto

# Inverse text normalization
bash export_grammars.sh --GRAMMARS=itn_grammars --LANGUAGE=el --INPUT_CASE=cased --MODE=interactive
# inside the container:
echo "δεκατέσσερα Ιουλίου" | \
    normalizer_main --config=sparrowhawk_configuration.ascii_proto
```

`normalizer_main` reads one input per line from **stdin** and writes the normalized result to
**stdout** — this is the entry point you wrap for serving.

### 6.4 Run the grammar test suite in the container

```bash
bash export_grammars.sh --GRAMMARS=tn_grammars  --LANGUAGE=el --INPUT_CASE=cased --MODE=test
bash export_grammars.sh --GRAMMARS=itn_grammars --LANGUAGE=el --INPUT_CASE=cased --MODE=test
```

These run `tests/nemo_text_processing/el/test_sparrowhawk_{normalization,inverse_text_normalization}.sh`
inside the container against the committed test cases.

### 6.5 How the grammars are wired

`docker/launch.sh` mounts the two `.far` directories into the Sparrowhawk grammar path:

```
<lang>_<grammars>_<input_case>/classify   -> /workspace/sparrowhawk/documentation/grammars/en_toy/classify
<lang>_<grammars>_<input_case>/verbalize  -> /workspace/sparrowhawk/documentation/grammars/en_toy/verbalize
```

`sparrowhawk_configuration.ascii_proto` (in `documentation/grammars/`) points at
`classify/tokenize_and_classify.far` and `verbalize/verbalize.far`. To serve TN and ITN
simultaneously, run two containers (or two config/grammar dirs) — one per direction — since each
config points at a single classify+verbalize pair.

### 6.6 Serving pattern

For a long-running service, start the container with the grammars mounted and keep
`normalizer_main` fed via a socket/HTTP wrapper of your choice, e.g.:

```bash
docker run -i --rm \
  -v "$PWD/el_tn_grammars_cased/classify:/workspace/sparrowhawk/documentation/grammars/en_toy/classify" \
  -v "$PWD/el_tn_grammars_cased/verbalize:/workspace/sparrowhawk/documentation/grammars/en_toy/verbalize" \
  -w /workspace/sparrowhawk/documentation/grammars \
  sparrowhawk:latest \
  normalizer_main --config=sparrowhawk_configuration.ascii_proto
```

Then pipe requests to the container's stdin (one utterance per line). Wrap this with a small
server process (gRPC/HTTP) that owns the stdin/stdout pipes for a production endpoint.

---

## 7. Greek-specific behavior notes

- **Numbers** render in the neuter citation form; the thousands multiplier is feminine
  (`3000` → `τρεις χιλιάδες`); millions/billions stay neuter. Range: 0–999,999,999,999.
- **Ordinals** inflect for full gender/case (`1ος`→`πρώτος`, `21η`→`εικοστή πρώτη`, `3ου`→`τρίτου`).
- **Measures (ITN)** output SI symbols as the canonical form (`δέκα μέτρα` → `10 m`).
- **Acronyms** are handled **only via the whitelist**. There is deliberately no generic
  all-caps letter-splitter: Greek acronym pronunciation is irregular and not derivable from
  spelling (ΕΡΤ→"ερτ", ΚΚΕ→"κου-κου-ε", ΕΥΔΑΠ→"εϊδάπ"), and a blanket splitter would mangle
  ordinary all-caps words (ΕΛΛΑΔΑ, ΙΝΤΡΑΚΟΜ). Add known acronyms with their correct spoken form
  to the whitelist TSV.

---

## 8. Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `ModuleNotFoundError: nemo_text_processing` when exporting | Package not importable from the tools dir. `pip install -e .` or prefix with `PYTHONPATH=<repo-root>`. |
| Docker build fails on `conda install thrax=1.3.4` with `PackagesNotFoundError` / `linux-aarch64` | You are on arm64 (e.g. Apple Silicon). Build on x86_64, or force `--platform linux/amd64` (slow QEMU emulation). |
| Grammar changes not taking effect | Stale cache. Delete `cache_dir` / the `*_grammars_*` dir, or pass `--overwrite_cache` / `overwrite_cache=True`. |
| Want TN and ITN from one endpoint | Run two containers/configs; each Sparrowhawk config serves a single classify+verbalize pair. |
