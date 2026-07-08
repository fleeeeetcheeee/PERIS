"""Model artifact store. Each artifact: model.txt + meta.json, versioned with a
training-data hash. artifacts/current.json points at the promoted incumbent."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import pandas as pd

from core import paths

CURRENT = paths.ARTIFACTS / "current.json"


def data_hash(df: pd.DataFrame, feature_cols: list[str]) -> str:
    """Deterministic hash of the training slice: shape, date range, and value digest."""
    h = hashlib.sha256()
    h.update(str(df.shape).encode())
    h.update(str(df["date"].min()).encode())
    h.update(str(df["date"].max()).encode())
    h.update(",".join(feature_cols).encode())
    h.update(pd.util.hash_pandas_object(df[feature_cols].round(8), index=False).values.tobytes())
    return h.hexdigest()[:12]


def save_artifact(booster: lgb.Booster, meta: dict[str, Any]) -> Path:
    name = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{meta['data_hash']}"
    adir = paths.ARTIFACTS / name
    adir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(adir / "model.txt"))
    (adir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return adir


def promote(artifact_dir: Path) -> None:
    paths.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    CURRENT.write_text(json.dumps({"artifact": artifact_dir.name}))


def load_current() -> tuple[lgb.Booster, dict[str, Any]] | None:
    if not CURRENT.exists():
        return None
    name = json.loads(CURRENT.read_text())["artifact"]
    adir = paths.ARTIFACTS / name
    model_path = adir / "model.txt"
    meta_path = adir / "meta.json"
    if not model_path.exists() or not meta_path.exists():
        return None
    booster = lgb.Booster(model_file=str(model_path))
    meta = json.loads(meta_path.read_text())
    return booster, meta


def incumbent_metric(name: str = "mean_ic") -> float | None:
    loaded = load_current()
    if loaded is None:
        return None
    _, meta = loaded
    return meta.get("harness", {}).get(name)
