# Sparrowhawk TN/ITN HTTP server

A small FastAPI service that serves **Text Normalization (TN)** and **Inverse Text
Normalization (ITN)** through Google [Sparrowhawk](https://github.com/google/sparrowhawk) —
the C++ backend that runs the compiled `.far` WFST grammars exported from
`nemo_text_processing`, no Python grammar runtime required.

One container serves **both directions** by running two persistent
`normalizer_main` processes (one for TN, one for ITN), each behind a small worker
pool, wrapped by a JSON HTTP API.

```
              ┌──────────────────────── nemo-sparrowhawk-server ───────────────────────┐
  HTTP  ─────▶│  FastAPI (app.py)                                                        │
  :8000       │     ├─ /tn   ─▶ EnginePool("tn")  ─▶ normalizer_main (TN grammars)  x N  │
              │     └─ /itn  ─▶ EnginePool("itn") ─▶ normalizer_main (ITN grammars) x N  │
              └──────────────────────────────────────────────────────────────────────────┘
```

## Endpoints

| Method | Path         | Body                                            | Returns |
|--------|--------------|-------------------------------------------------|---------|
| GET    | `/health`    | —                                               | `{"status","engines":{"tn","itn"}}` |
| GET    | `/`          | —                                               | service metadata |
| POST   | `/tn`        | `{"text":"..."}` or `{"texts":[...]}`           | `{"direction":"tn","results":[...]}` |
| POST   | `/itn`       | `{"text":"..."}` or `{"texts":[...]}`           | `{"direction":"itn","results":[...]}` |
| POST   | `/normalize` | `{"direction":"tn"\|"itn","text":"..."}`        | `{"direction","results":[...]}` |

## Quick start (Docker)

Runs from `tools/text_processing_deployment/` (the Docker build context). Requires
the exported grammars — Greek cased ones are already committed as
`el_tn_grammars_cased/` and `el_itn_grammars_cased/`.

```bash
cd tools/text_processing_deployment

# Builds the sparrowhawk:latest base image if missing, then the server image,
# then runs it on :8000. Add EXPORT=1 to re-export the grammars first.
bash server/build_and_run.sh
```

Then:

```bash
curl -s localhost:8000/health

curl -s localhost:8000/tn  -H 'content-type: application/json' \
  -d '{"text":"Στις 25/12/2024 πλήρωσα 15,50€"}'
# {"direction":"tn","results":["Στις είκοσι πέντε Δεκεμβρίου δύο χιλιάδες είκοσι τέσσερα πλήρωσα δεκαπέντε ευρώ και πενήντα λεπτά"]}

curl -s localhost:8000/itn -H 'content-type: application/json' \
  -d '{"text":"δεκατέσσερις Ιουλίου"}'
# {"direction":"itn","results":["14 Ιουλίου"]}

# batch
curl -s localhost:8000/normalize -H 'content-type: application/json' \
  -d '{"direction":"itn","texts":["δέκα μέτρα","δεκατέσσερις Ιουλίου"]}'
```

## Building manually

```bash
cd tools/text_processing_deployment

# 1. base Sparrowhawk image (slow first time; needs Linux x86_64 — see the guide)
bash docker/build.sh                       # -> sparrowhawk:latest

# 2. server image (context = this dir so it can COPY the grammar dirs)
docker build -f server/Dockerfile -t nemo-sparrowhawk-server \
  --build-arg GRAMMARS_TN_SRC=el_tn_grammars_cased \
  --build-arg GRAMMARS_ITN_SRC=el_itn_grammars_cased .

# 3. run
docker run --rm -p 8000:8000 nemo-sparrowhawk-server
```

### Serving a different language

Export the grammars for your language, then point the build args at them:

```bash
bash export_grammars.sh --GRAMMARS=tn_grammars  --LANGUAGE=de --INPUT_CASE=cased --MODE=export
bash export_grammars.sh --GRAMMARS=itn_grammars --LANGUAGE=de --INPUT_CASE=cased --MODE=export
docker build -f server/Dockerfile -t nemo-sparrowhawk-de \
  --build-arg GRAMMARS_TN_SRC=de_tn_grammars_cased \
  --build-arg GRAMMARS_ITN_SRC=de_itn_grammars_cased .
```

To swap grammars **without rebuilding**, bind-mount over the baked-in dirs:

```bash
docker run --rm -p 8000:8000 \
  -v "$PWD/de_tn_grammars_cased:/app/grammars/tn:ro" \
  -v "$PWD/de_itn_grammars_cased:/app/grammars/itn:ro" \
  nemo-sparrowhawk-server
```

If a direction's grammars are absent, that direction is simply disabled and its
endpoint returns `503`; the other keeps working.

## Configuration (environment variables)

| Var | Default | Meaning |
|-----|---------|---------|
| `PORT` / `HOST` | `8000` / `0.0.0.0` | HTTP bind |
| `WORKERS_PER_DIRECTION` | `2` | persistent `normalizer_main` processes per direction (throughput) |
| `IO_TIMEOUT` | `15` | seconds to wait for a normalized line before restarting the worker |
| `STDBUF_PREFIX` | `stdbuf -oL -eL` | forces line buffering on the child (see below); set empty to disable |
| `SPARROWHAWK_BIN` | `normalizer_main` | path to the binary |
| `SPARROWHAWK_CONFIG` | `sparrowhawk_configuration.ascii_proto` | config filename inside each serving dir |
| `GRAMMARS_TN` / `GRAMMARS_ITN` | `/app/grammars/{tn,itn}` | source of the `.far` grammars |
| `LOG_LEVEL` | `INFO` | set `DEBUG` to log the child's stderr |

## Why `stdbuf -oL`

`normalizer_main` writes results with C++ stdio, which is **block-buffered** when
stdout is a pipe. In a persistent request/response loop that means the single
result line is never flushed until the buffer fills — a deadlock. Launching the
child under `stdbuf -oL -eL` forces line buffering so each result is flushed
immediately. stderr is additionally drained on a background thread so diagnostics
can't fill the pipe, and so stdout carries only the normalized text (one line per
request). This matches the contract the repo's `test_sparrowhawk_*.sh` scripts
rely on (`echo … | normalizer_main … | tail -n 1`).

## Local test (no Sparrowhawk image needed)

`tests/test_local.sh` runs the full HTTP stack against `tests/fake_normalizer_main.py`,
a stand-in that honours the same stdin→stdout line contract. It validates the pipe
wrapper, the pool, and every endpoint on any platform:

```bash
PYTHON=python3 bash server/tests/test_local.sh
```

## Notes

* A single `normalizer_main` process is single-threaded; concurrency comes from
  `WORKERS_PER_DIRECTION`. The FastAPI handler offloads the blocking pipe I/O to a
  threadpool, and the pool's checkout queue provides backpressure.
* Keep `uvicorn --workers 1`: the engine pools live in-process, so extra uvicorn
  workers would multiply the number of `normalizer_main` processes. Scale out with
  container replicas behind a load balancer instead.
* Sparrowhawk (and thus this image) is **Linux x86_64** — see
  [`GREEK_TN_ITN_GUIDE.md`](../GREEK_TN_ITN_GUIDE.md) §1/§8 for the platform caveat.
