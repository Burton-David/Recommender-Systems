"""FastAPI recommendation service — the Python baseline.

Loads a model exported by ``serving.python.export_model`` and exposes
``GET /recommend?user_id=<id>&n=<count>`` returning a JSON object
``{"items": [...]}``. The numerical work per request is one matrix-vector
multiply (user factor times the item-factor matrix) followed by an
argpartition for the top-N — both numpy operations land on the platform's
BLAS, which is the part Go can't easily match for this workload.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException

_MODEL_PATH = Path(os.environ.get("RECSYS_MODEL", "serving/model.json"))


def _load_model(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    return {
        "user_ids": np.asarray(payload["user_ids"]),
        "item_ids": np.asarray(payload["item_ids"]),
        "user_factors": np.asarray(payload["user_factors"], dtype=np.float32),
        "item_factors_T": np.asarray(payload["item_factors_T"], dtype=np.float32),
    }


_MODEL = _load_model(_MODEL_PATH)
_USER_INDEX = {int(uid): idx for idx, uid in enumerate(_MODEL["user_ids"])}

app = FastAPI()


@app.get("/recommend")
def recommend(user_id: int, n: int = 10) -> dict[str, list]:
    idx = _USER_INDEX.get(user_id)
    if idx is None:
        raise HTTPException(status_code=404, detail="unknown user_id")
    user_factor = _MODEL["user_factors"][idx]
    scores = _MODEL["item_factors_T"] @ user_factor  # (n_items,)
    if n >= scores.size:
        order = np.argsort(-scores)
    else:
        partial = np.argpartition(-scores, n)[:n]
        order = partial[np.argsort(-scores[partial])]
    return {"items": _MODEL["item_ids"][order].tolist()}


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}
