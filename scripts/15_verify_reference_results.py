#!/usr/bin/env python
"""Redistribution-safe verification of frozen MSRR reference artifacts."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference_results"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest() -> int:
    manifest = ROOT / "FILE_MANIFEST.csv"
    require(manifest.is_file(), "FILE_MANIFEST.csv missing")
    checked = 0
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            rel = row["relative_path"]
            if rel == "FILE_MANIFEST.csv":
                continue
            path = ROOT / rel
            require(path.is_file(), f"manifest file missing: {rel}")
            require(path.stat().st_size == int(row["bytes"]), f"size mismatch: {rel}")
            require(sha256(path) == row["sha256"], f"hash mismatch: {rel}")
            checked += 1
    return checked


def main() -> int:
    dims = pd.read_csv(REF / "ablation" / "OOF_RESULTS.csv")
    observed = dict(zip(dims.variant, dims.feature_dim))
    require(observed["A0_RAW"] == 792, "RAW dimension mismatch")
    require(observed["A6_FULL_MSRR"] == 2446, "MSRR dimension mismatch")
    require(len(dims) == 7, "ablation variant count mismatch")

    hiro = pd.read_csv(REF / "hiroshima" / "OOF_RESULTS.csv")
    require(set(hiro.backbone) == {"ElasticNet-Logistic", "MLP", "HistGradientBoosting", "XGBoost"}, "four-backbone panel incomplete")
    require(set(hiro.representation) == {"RAW", "MSRR"}, "representation panel incomplete")
    require(len(hiro) == 8, "cross-backbone row count mismatch")

    placebo = pd.read_csv(REF / "placebo" / "123_OOF_METRICS.csv")
    require(set(placebo.variant) == {"RAW", "TRUE_MSRR", "GLOBAL_CENTERED", "SHUFFLED_CONTEXT"}, "placebo panel incomplete")

    kyushu = pd.read_csv(REF / "kyushu" / "124D_EXTERNAL_METRICS.csv")
    require(set(kyushu.representation) == {"RAW", "MSRR"}, "Kyushu primary panel incomplete")

    for path in REF.rglob("*.csv"):
        frame = pd.read_csv(path)
        metric_columns = [c for c in ["AUROC", "AUPRC", "StrictPair", "Edge"] if c in frame]
        for column in metric_columns:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            require(((values >= 0) & (values <= 1)).all(), f"metric outside [0,1]: {path.name}:{column}")

    for name in ["XGBoost_RAW_OOF_margin.npy", "XGBoost_MSRR_OOF_margin.npy"]:
        values = np.load(REF / "hiroshima" / name)
        require(values.shape == (15168,), f"unexpected OOF shape: {name}")

    checked = verify_manifest()
    print(f"PASS reference results verified; manifest_files_checked={checked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

