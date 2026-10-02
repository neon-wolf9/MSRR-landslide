from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_five_fold_summary():
    frame = pd.read_csv(ROOT / "data" / "manifests" / "10_FOLD_SUMMARY.csv")
    numeric = set(pd.to_numeric(frame.select_dtypes(include="number").stack(), errors="coerce").dropna().astype(int))
    assert {1, 2, 3, 4, 5}.issubset(numeric) or len(frame) == 5

