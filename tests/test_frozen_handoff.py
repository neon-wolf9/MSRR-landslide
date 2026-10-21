"""Lightweight structure checks for archived 120B hand-off, not evaluation."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "experiments" / "MSRR_CROSS_BACKBONE_5FOLD_V1B"
BACKBONES = ("ElasticNet_Logistic", "MLP", "HistGradientBoosting", "XGBoost")


def _table(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def test_frozen_120b_handoff_structure():
    assert (HANDOFF / "RUN_MANIFEST.json").is_file()
    oof = _table(HANDOFF / "OOF_RESULTS.csv")
    folds = _table(HANDOFF / "FOLD_METRICS.csv")
    selected = _table(HANDOFF / "VALIDATION_SELECTION.csv")
    assert len(oof) == 8
    assert len(folds) == 40
    assert selected
    assert {"backbone", "representation", "AUROC", "AUPRC", "StrictPair", "Edge"} <= set(oof[0])
    assert {"human_fold", "backbone", "representation", "config_id", "validation_score"} <= set(selected[0])
    for backbone in BACKBONES:
        for representation in ("RAW", "MSRR"):
            array_path = HANDOFF / f"{backbone}_{representation}_OOF_margin.npy"
            scores = np.load(array_path, mmap_mode="r")
            assert scores.shape == (15168,)
            assert np.issubdtype(scores.dtype, np.number)


def test_128_authority_record_paths():
    a3 = ROOT / "experiments" / "MSRR_128A3_BENCHMARK_ALIGNED_PU_PROTOCOL"
    b0 = ROOT / "experiments" / "MSRR_128B0_BASELINE_ADAPTER_SMOKE_TEST"
    names = (
        (a3, "128A3_PU_BAGGING_SPEC.json"),
        (a3, "128A3_SPY_PU_BRF_SPEC.json"),
        (a3, "128A3_PU_PULL_SPEC.json"),
        (b0, "128B0_PU_BAGGING_IMPLEMENTATION.json"),
        (b0, "128B0_SPY_PU_BRF_IMPLEMENTATION.json"),
        (b0, "128B0_PU_PULL_IMPLEMENTATION.json"),
        (b0, "128B0_READY_FOR_128B1.json"),
    )
    for directory, filename in names:
        record = json.loads((directory / filename).read_text(encoding="utf-8-sig"))
        assert isinstance(record, dict) and record


def test_124d_freeze_record_at_125_input_path():
    path = (
        ROOT
        / "external"
        / "kyushu_2017_asakura_toho"
        / "100_external_validation"
        / "MSRR_KYUSHU2017_EXTERNAL_V1D"
        / "124D_PREDICTION_FREEZE.json"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "2b29a76b2a61cfbe743be40b79b50f214521ceb6bec18035ea2b84cd2aedb2b0"
    )
    record = json.loads(path.read_text(encoding="utf-8-sig"))
    assert record["prediction_rows_per_representation"] == 5076
