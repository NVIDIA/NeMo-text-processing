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

"""HTTP server exposing Sparrowhawk TN and ITN over a small JSON API.

Endpoints
---------
* ``GET  /health``          liveness + which engines are up
* ``POST /tn``              {"text": "..."} | {"texts": [...]}  -> normalized
* ``POST /itn``             {"text": "..."} | {"texts": [...]}  -> normalized
* ``POST /normalize``       {"text": "...", "direction": "tn"|"itn"}
* ``GET  /``                service metadata

Directions are enabled when their grammar working directory exists (``TN_DIR`` /
``ITN_DIR``). Each direction is backed by a pool of persistent ``normalizer_main``
processes (see :mod:`normalizer`).
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from normalizer import EnginePool, pool_from_env
from pydantic import BaseModel, Field

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("server")

TN_DIR = os.environ.get("TN_DIR", "/workspace/serve/tn")
ITN_DIR = os.environ.get("ITN_DIR", "/workspace/serve/itn")
LANGUAGE = os.environ.get("LANGUAGE", "el")

ENGINES: Dict[str, EnginePool] = {}


def _config_present(direction_dir: str) -> bool:
    return os.path.isfile(
        os.path.join(direction_dir, os.environ.get("SPARROWHAWK_CONFIG", "sparrowhawk_configuration.ascii_proto"))
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Spin up an engine pool for each direction whose grammars are present.
    for name, d in (("tn", TN_DIR), ("itn", ITN_DIR)):
        if _config_present(d):
            log.info("initializing '%s' engine from %s", name, d)
            ENGINES[name] = pool_from_env(name, d)
        else:
            log.warning("skipping '%s': no config found under %s", name, d)
    if not ENGINES:
        log.error("no engines available — check TN_DIR/ITN_DIR and grammar export")
    yield
    for pool in ENGINES.values():
        pool.close()


app = FastAPI(
    title="NeMo Sparrowhawk TN/ITN",
    description="Text Normalization and Inverse Text Normalization served by Google Sparrowhawk.",
    version="1.0.0",
    lifespan=lifespan,
)


class NormalizeRequest(BaseModel):
    text: Optional[str] = Field(default=None, description="Single input string.")
    texts: Optional[List[str]] = Field(default=None, description="Batch of input strings.")


class DirectedRequest(NormalizeRequest):
    direction: str = Field(description="'tn' (written->spoken) or 'itn' (spoken->written).")


class NormalizeResponse(BaseModel):
    direction: str
    results: List[str]


def _inputs(req: NormalizeRequest) -> List[str]:
    if req.texts is not None:
        return req.texts
    if req.text is not None:
        return [req.text]
    raise HTTPException(status_code=422, detail="provide 'text' or 'texts'")


async def _run(direction: str, req: NormalizeRequest) -> NormalizeResponse:
    pool = ENGINES.get(direction)
    if pool is None:
        raise HTTPException(status_code=503, detail=f"direction '{direction}' not available")
    items = _inputs(req)
    # Offload the blocking pipe I/O to the threadpool; the pool's internal queue
    # bounds real concurrency to the number of worker processes.
    results = [await run_in_threadpool(pool.normalize, t) for t in items]
    return NormalizeResponse(direction=direction, results=results)


@app.get("/")
def root():
    return {
        "service": "nemo-sparrowhawk-tn-itn",
        "language": LANGUAGE,
        "directions": sorted(ENGINES.keys()),
        "endpoints": ["/health", "/tn", "/itn", "/normalize"],
    }


@app.get("/health")
def health():
    status = {name: ("up" if pool.healthy() else "down") for name, pool in ENGINES.items()}
    ok = bool(ENGINES) and all(v == "up" for v in status.values())
    return {"status": "ok" if ok else "degraded", "engines": status}


@app.post("/tn", response_model=NormalizeResponse)
async def tn(req: NormalizeRequest):
    return await _run("tn", req)


@app.post("/itn", response_model=NormalizeResponse)
async def itn(req: NormalizeRequest):
    return await _run("itn", req)


@app.post("/normalize", response_model=NormalizeResponse)
async def normalize(req: DirectedRequest):
    direction = req.direction.lower()
    if direction not in ("tn", "itn"):
        raise HTTPException(status_code=422, detail="direction must be 'tn' or 'itn'")
    return await _run(direction, req)
