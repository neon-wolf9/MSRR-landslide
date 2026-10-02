import importlib.util
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]


def load_authority():
    path = ROOT / "scripts" / "120B_RUN_MSRR_CROSS_BACKBONE_5FOLD_NO_LIGHTGBM_ONE_SHOT.py"
    spec = importlib.util.spec_from_file_location("msrr120b_metrics_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_strictpair_and_edge_definitions():
    scores = np.array([0.9, 0.2, 0.1, 0.4, 0.6, 0.3])
    triplets = scores.reshape(-1, 3)
    margins = triplets[:, :1] - triplets[:, 1:]
    strict_pair = (margins.min(axis=1) > 0).mean()
    edge = (margins > 0).mean()
    assert strict_pair == 0.5
    assert edge == 0.75


def test_canonical_metric_wrappers_and_matched_set_margin():
    authority = load_authority()
    scores = np.array([[0.9, 0.2, 0.1], [0.4, 0.6, 0.3]], dtype=np.float64)
    y = np.array([1, 0, 0, 1, 0, 0], dtype=np.int8)
    got = authority.metrics_from_ordered_rows(scores.reshape(-1), y)
    probability = authority.sigmoid(scores.reshape(-1))
    assert np.isclose(got["AUROC"], roc_auc_score(y, probability))
    assert np.isclose(got["AUPRC"], average_precision_score(y, probability))
    assert got["StrictPair"] == 0.5
    assert got["Edge"] == 0.75
    matched_set_margin = scores[:, 0] - scores[:, 1:].max(axis=1)
    np.testing.assert_allclose(matched_set_margin, np.array([0.7, -0.2]))
