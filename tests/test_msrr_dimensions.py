from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_authority():
    path = ROOT / "scripts" / "119_RUN_MSRR_REPRESENTATION_ABLATION_5FOLD_ONE_SHOT.py"
    spec = importlib.util.spec_from_file_location("msrr119", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_frozen_dimensions_and_real_transform():
    authority = load_authority()
    assert authority.EXPECTED_DIMS["A0_RAW"] == 792
    assert authority.EXPECTED_DIMS["A6_FULL_MSRR"] == 2446
    static = np.arange(3 * 92, dtype=np.float32).reshape(3, 92)
    rain = np.arange(3 * 70 * 10, dtype=np.float32).reshape(3, 70, 10)
    components, anchors = authority.build_components(np.array([[0, 1, 2]]), static, rain)
    assert anchors.tolist() == [0, 1, 2]
    assert authority.make_variant("A0_RAW", components).shape == (3, 792)
    assert authority.make_variant("A6_FULL_MSRR", components).shape == (3, 2446)
