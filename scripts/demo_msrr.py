#!/usr/bin/env python
"""Redistribution-safe smoke test of the canonical 120B MSRR implementation."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "scripts" / "120B_RUN_MSRR_CROSS_BACKBONE_5FOLD_NO_LIGHTGBM_ONE_SHOT.py"
EXAMPLE = ROOT / "data" / "example" / "synthetic_msrr_demo.npz"


def load_authority():
    spec = importlib.util.spec_from_file_location("msrr120b_demo_authority", AUTHORITY)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("CANONICAL_MSRR_LOADER_MISSING")
    spec.loader.exec_module(module)
    return module


def main() -> int:
    authority = load_authority()
    with np.load(EXAMPLE) as data:
        matched = data["matched_raw"].astype(np.float32)
        scores = data["scores"].astype(np.float64)

    n_sets = matched.shape[0]
    if matched.shape != (n_sets, 3, 792) or not 5 <= n_sets <= 10:
        raise RuntimeError(f"BAD_SYNTHETIC_INPUT_SHAPE={matched.shape}")

    static92 = matched[:, :, :92].reshape(-1, 92)
    rain = matched[:, :, 92:].reshape(-1, 70, 10)
    pair_rows = np.arange(n_sets * 3, dtype=np.int64).reshape(n_sets, 3)
    msrr, anchors = authority.msrr_features(pair_rows, static92, rain)
    msrr3 = msrr.reshape(n_sets, 3, 2446)

    shared_s = matched[:, :, :92].mean(axis=1, keepdims=True)
    shared_r = matched[:, :, 92:].reshape(n_sets, 3, 70, 10).mean(axis=1, keepdims=True)
    expected = np.concatenate(
        [
            matched[:, :, :92],
            matched[:, :, :92] - shared_s,
            np.abs(matched[:, :, :92] - shared_s),
            matched[:, :, 92:],
            (matched[:, :, 92:].reshape(n_sets, 3, 70, 10) - shared_r).reshape(n_sets, 3, 700),
            np.abs(matched[:, :, 92:].reshape(n_sets, 3, 70, 10) - shared_r).reshape(n_sets, 3, 700),
            authority.dynsum7(
                (matched[:, :, 92:].reshape(n_sets, 3, 70, 10) - shared_r).reshape(-1, 70, 10)
            ).reshape(n_sets, 3, 70),
        ],
        axis=2,
    ).astype(np.float32)
    numerical_ok = bool(
        anchors.tolist() == list(range(n_sets * 3))
        and np.allclose(msrr3, expected, rtol=0.0, atol=1e-6)
    )

    y = np.tile(np.array([1, 0, 0], dtype=np.int8), n_sets)
    metrics = authority.metrics_from_ordered_rows(scores.reshape(-1), y)
    expected_strict = float(((scores[:, 0] > scores[:, 1]) & (scores[:, 0] > scores[:, 2])).mean())
    expected_edge = float(np.column_stack((scores[:, 0] > scores[:, 1], scores[:, 0] > scores[:, 2])).mean())
    strict_ok = bool(np.isclose(metrics["StrictPair"], expected_strict))
    edge_ok = bool(np.isclose(metrics["Edge"], expected_edge))
    passed = numerical_ok and strict_ok and edge_ok and msrr3.shape == (n_sets, 3, 2446)

    print(f"Input shape: {matched.shape}")
    print(f"MSRR output shape: {msrr3.shape}")
    print(f"MSRR numerical verification: {'PASS' if numerical_ok else 'FAIL'}")
    print(f"StrictPair verification: {'PASS' if strict_ok else 'FAIL'} ({metrics['StrictPair']:.6f})")
    print(f"Edge verification: {'PASS' if edge_ok else 'FAIL'} ({metrics['Edge']:.6f})")
    print(f"STATUS: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
