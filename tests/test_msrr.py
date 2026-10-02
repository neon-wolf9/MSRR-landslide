from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def authority():
    path = ROOT / "scripts" / "120B_RUN_MSRR_CROSS_BACKBONE_5FOLD_NO_LIGHTGBM_ONE_SHOT.py"
    spec = importlib.util.spec_from_file_location("msrr120b_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def synthetic():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(5, 3, 792)).astype(np.float32)
    return x, x[:, :, :92].reshape(-1, 92), x[:, :, 92:].reshape(-1, 70, 10), np.arange(15).reshape(5, 3)


def test_msrr_shape_context_deviations_and_temporal_summaries():
    mod = authority()
    raw, static, rain, pairs = synthetic()
    out, anchors = mod.msrr_features(pairs, static, rain)
    z = out.reshape(5, 3, 2446)
    s_mean = raw[:, :, :92].mean(axis=1, keepdims=True)
    r4 = raw[:, :, 92:].reshape(5, 3, 70, 10)
    r_mean = r4.mean(axis=1, keepdims=True)
    r_rel = r4 - r_mean
    assert raw.shape == (5, 3, 792)
    assert z.shape == (5, 3, 2446)
    assert anchors.tolist() == list(range(15))
    np.testing.assert_allclose(z[:, :, :92], raw[:, :, :92])
    np.testing.assert_allclose(z[:, :, 92:184], raw[:, :, :92] - s_mean, atol=1e-6)
    np.testing.assert_allclose(z[:, :, 184:276], np.abs(raw[:, :, :92] - s_mean), atol=1e-6)
    np.testing.assert_allclose(z[:, :, 976:1676], r_rel.reshape(5, 3, 700), atol=1e-6)
    np.testing.assert_allclose(z[:, :, 1676:2376], np.abs(r_rel).reshape(5, 3, 700), atol=1e-6)
    np.testing.assert_allclose(z[:, :, 2376:], mod.dynsum7(r_rel.reshape(-1, 70, 10)).reshape(5, 3, 70))


def test_shared_mean_is_permutation_symmetric_and_role_free():
    mod = authority()
    raw, _, _, _ = synthetic()
    perm = np.array([2, 0, 1])

    def transform(values):
        s = values[:, :, :92].reshape(-1, 92)
        r = values[:, :, 92:].reshape(-1, 70, 10)
        p = np.arange(15).reshape(5, 3)
        return mod.msrr_features(p, s, r)[0].reshape(5, 3, 2446)

    original = transform(raw)
    permuted = transform(raw[:, perm])
    inverse = np.argsort(perm)
    # Float32 reduction order may differ by a few ULPs after member permutation.
    np.testing.assert_allclose(permuted[:, inverse], original, rtol=0.0, atol=3e-6)
    assert tuple(inspect.signature(mod.msrr_features).parameters) == ("pair_rows", "static92", "rain")
