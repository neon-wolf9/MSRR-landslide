#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
126_RUN_HIROSHIMA_KYUSHU_DOMAIN_SHIFT_CHARACTERIZATION.py

Post-hoc domain-shift characterization
======================================

Scientific question
-------------------
The frozen internal/external experiments have established two empirical facts:

1) absolute discrimination decreases from Hiroshima-2018 internal OOF to
   Kyushu-2017 external transfer;
2) the MSRR > RAW representation advantage is preserved across four learner
   families.

Experiment 126 does NOT train, tune, select, refit, or rescue any predictive
model. Its purpose is only to characterize how the two event domains differ
and whether the matched-relative signal structure remains recognizable.

Primary analyses (label-agnostic)
---------------------------------
A. Static model-input shift
   - 90 formal continuous static fields + soil-missing indicator
   - all compared after the exact full-Hiroshima preprocessor used for 124D/125
   - metrics: Wasserstein distance, KS statistic, PSI, median/mean shift

B. Categorical shift
   - road_nearest_road_class
   - Jensen-Shannon distance and frequency table

C. Rainfall-process shift
   - 10 formal rainfall/API fields, 70 steps
   - per-sample trajectory summaries
   - mean temporal-profile difference over the 70 steps

D. Matched-context geometry shift
   - within-set mean absolute deviation (MAD) for static and rainfall inputs
   - label-agnostic: only the P/C1/C2 set membership is used

Secondary analysis (outcome-conditioned, descriptive only)
-----------------------------------------------------------
E. Inventory-positive vs matched-control relative-signal preservation
   - contrast = inventory-positive - mean(two matched controls)
   - static: 91 continuous model channels (category excluded)
   - rainfall: per-field temporal-mean contrast
   - reports cross-event Spearman/Pearson/cosine/sign agreement
   - this is descriptive and is NOT used for model selection or redefinition

Important label semantics
-------------------------
"control" means matched non-inventory control. It is not interpreted as a
confirmed geological absence.

Run
---
python ^
  <PROJECT_ROOT>\scripts\126_RUN_HIROSHIMA_KYUSHU_DOMAIN_SHIFT_CHARACTERIZATION.py

Output
------
<PROJECT_ROOT>\experiments\MSRR_126_DOMAIN_SHIFT_CHARACTERIZATION
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from scipy.spatial.distance import jensenshannon
from scipy.stats import (
    ks_2samp,
    pearsonr,
    spearmanr,
    wasserstein_distance,
)

ROOT = Path(__file__).resolve().parents[1]

KYUSHU = (
    ROOT
    / "external"
    / "kyushu_2017_asakura_toho"
    / "99_frozen_dataset"
)

OUT125 = (
    ROOT
    / "external"
    / "kyushu_2017_asakura_toho"
    / "100_external_validation"
    / "MSRR_KYUSHU2017_EXTERNAL_FULL_PANEL_V2"
)

OUT = (
    ROOT
    / "experiments"
    / "MSRR_126_DOMAIN_SHIFT_CHARACTERIZATION"
)

EXPECTED_K_ROWS = 5076
EXPECTED_K_SETS = 1692


# =============================================================================
# 0. GENERAL UTILITIES
# =============================================================================

def log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (OUT / "126_RUN.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def jwrite(path: Path, payload: Any) -> None:
    def conv(x):
        if isinstance(x, dict):
            return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        return x

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            conv(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def locate_script(preferred_name: str, pattern: str) -> Path:
    preferred = ROOT / "scripts" / preferred_name
    if preferred.exists():
        return preferred
    cands = sorted((ROOT / "scripts").glob(pattern))
    if not cands:
        raise FileNotFoundError(
            f"Cannot locate {preferred_name} under {ROOT / 'scripts'}"
        )
    return cands[0]


def import_script(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(
        module_name,
        str(path),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import script: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def finite_1d(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64).reshape(-1)
    return z[np.isfinite(z)]


def safe_iqr(x: np.ndarray) -> float:
    z = finite_1d(x)
    if len(z) == 0:
        return float("nan")
    q1, q3 = np.quantile(z, [0.25, 0.75])
    return float(q3 - q1)


def safe_div(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) < 1e-12:
        return float("nan")
    return float(a / b)


def psi_from_reference(
    ref: np.ndarray,
    ext: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """
    Population Stability Index with bins fixed ONLY from Hiroshima reference.
    This is descriptive. No threshold-based gate is used.
    """
    a = finite_1d(ref)
    b = finite_1d(ext)

    if len(a) == 0 or len(b) == 0:
        return float("nan")

    q = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(a, q)
    edges = np.unique(edges)

    if len(edges) < 3:
        return float("nan")

    edges = edges.astype(np.float64)
    edges[0] = -np.inf
    edges[-1] = np.inf

    ca, _ = np.histogram(a, bins=edges)
    cb, _ = np.histogram(b, bins=edges)

    pa = ca.astype(np.float64) / max(ca.sum(), 1)
    pb = cb.astype(np.float64) / max(cb.sum(), 1)

    pa = np.clip(pa, eps, None)
    pb = np.clip(pb, eps, None)

    pa /= pa.sum()
    pb /= pb.sum()

    return float(np.sum((pb - pa) * np.log(pb / pa)))


def shift_metrics(
    h: np.ndarray,
    k: np.ndarray,
) -> Dict[str, float]:
    """
    Both h and k are already in the same Hiroshima-fitted model-input scale
    unless explicitly stated otherwise.
    """
    a = finite_1d(h)
    b = finite_1d(k)

    if len(a) == 0 or len(b) == 0:
        return {
            "h_n": len(a),
            "k_n": len(b),
            "h_mean": np.nan,
            "k_mean": np.nan,
            "h_median": np.nan,
            "k_median": np.nan,
            "h_std": np.nan,
            "k_std": np.nan,
            "h_iqr": np.nan,
            "k_iqr": np.nan,
            "delta_mean": np.nan,
            "delta_median": np.nan,
            "abs_delta_median": np.nan,
            "wasserstein": np.nan,
            "ks_statistic": np.nan,
            "ks_pvalue": np.nan,
            "psi_hiroshima_bins": np.nan,
        }

    hm = float(np.mean(a))
    km = float(np.mean(b))
    hmed = float(np.median(a))
    kmed = float(np.median(b))
    hs = float(np.std(a))
    ks = float(np.std(b))
    hi = safe_iqr(a)
    ki = safe_iqr(b)

    kres = ks_2samp(
        a,
        b,
        alternative="two-sided",
        mode="auto",
    )

    return {
        "h_n": int(len(a)),
        "k_n": int(len(b)),
        "h_mean": hm,
        "k_mean": km,
        "h_median": hmed,
        "k_median": kmed,
        "h_std": hs,
        "k_std": ks,
        "h_iqr": hi,
        "k_iqr": ki,
        "delta_mean": float(km - hm),
        "delta_median": float(kmed - hmed),
        "abs_delta_median": float(abs(kmed - hmed)),
        "wasserstein": float(wasserstein_distance(a, b)),
        "ks_statistic": float(kres.statistic),
        "ks_pvalue": float(kres.pvalue),
        "psi_hiroshima_bins": float(
            psi_from_reference(a, b)
        ),
    }


def top_records(
    df: pd.DataFrame,
    metric: str,
    n: int = 15,
    cols: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    if metric not in df.columns:
        return []
    keep = [c for c in cols if c in df.columns] + [metric]
    return (
        df.sort_values(metric, ascending=False)
        .head(n)[keep]
        .to_dict(orient="records")
    )


# =============================================================================
# 1. FORMAL AUTHORITY LOADING
# =============================================================================

def load_authorities():
    p124 = locate_script(
        "124D_RUN_KYUSHU2017_FROZEN_EXTERNAL_VALIDATION.py",
        "124D_RUN_KYUSHU2017_FROZEN_EXTERNAL_VALIDATION*.py",
    )

    m124 = import_script(
        "domain_shift_124d_authority",
        p124,
    )

    # Domain-shift outputs must never overwrite 124D.
    m124.OUT = OUT
    m124.log = log

    (
        p120,
        p26,
        m120,
        m26,
        bundle,
        runner,
        base,
    ) = m124.load_authorities()

    m120.OUT = OUT
    m120.log = log

    if hasattr(runner, "log"):
        runner.log = log

    return (
        p124,
        p120,
        p26,
        m124,
        m120,
        m26,
        bundle,
        runner,
        base,
    )


# =============================================================================
# 2. EXTERNAL RAW INPUT DISCOVERY + ALIGNMENT
# =============================================================================

def load_external_raw(
    m124: Any,
    base: Any,
    bundle: Any,
):
    inv = m124.inventory(KYUSHU)

    inv.to_csv(
        OUT / "126_EXTERNAL_FILE_INVENTORY.csv",
        index=False,
        encoding="utf-8-sig",
    )

    raw_static_fields = (
        list(bundle.ss["continuous_fields_in_order"])
        + [bundle.cat]
    )

    matched_path = m124.discover_by_schema(
        inv,
        [m124.EXPECTED["matched_rows"]],
        require_unit=True,
        require_pair=True,
        require_role_or_label=True,
        positive_name_tokens=[
            "matched",
            "pair",
            "triplet",
            "sample",
            "index",
            "formal",
        ],
        negative_name_tokens=[
            "dynamic",
            "rain",
            "192",
            "70",
            "prediction",
        ],
        label="MATCHED_INDEX",
    )

    static_path = m124.discover_by_schema(
        inv,
        [
            m124.EXPECTED["matched_rows"],
            m124.EXPECTED["master_grid_rows"],
        ],
        required_exact_cols=raw_static_fields,
        require_unit=True,
        positive_name_tokens=[
            "static",
            "feature",
            "raw",
            "92",
            "matched",
            "grid",
        ],
        negative_name_tokens=[
            "dynamic",
            "rain",
            "prediction",
            "result",
        ],
        label="RAW_STATIC_92",
    )

    dynamic70_path = m124.discover_by_schema(
        inv,
        [m124.EXPECTED["dynamic70_rows"]],
        required_exact_cols=list(base.RAIN_FIELDS),
        require_unit=True,
        positive_name_tokens=[
            "70",
            "dynamic",
            "rain",
            "view",
            "matched",
            "model",
        ],
        negative_name_tokens=[
            "192",
            "prediction",
            "result",
        ],
        label="DYNAMIC70_RAW_VIEW",
    )

    matched_raw = m124.read_table(matched_path)
    static_raw = m124.read_table(static_path)
    dynamic_raw = m124.read_table(dynamic70_path)

    model_index, pair_k = m124.build_external_model_index(
        matched_raw,
    )

    if len(model_index) != EXPECTED_K_ROWS:
        raise RuntimeError(
            f"Kyushu model_index rows={len(model_index)} expected={EXPECTED_K_ROWS}"
        )

    if pair_k.shape != (EXPECTED_K_SETS, 3):
        raise RuntimeError(
            f"Kyushu pair tensor={pair_k.shape} expected={(EXPECTED_K_SETS, 3)}"
        )

    return {
        "inventory": inv,
        "matched_path": matched_path,
        "static_path": static_path,
        "dynamic70_path": dynamic70_path,
        "matched_raw": matched_raw,
        "static_raw": static_raw,
        "dynamic_raw": dynamic_raw,
        "model_index": model_index,
        "pair_k": pair_k,
    }


# =============================================================================
# 3. CATEGORY ALIGNMENT
# =============================================================================

def external_static_in_model_order(
    static_raw: pd.DataFrame,
    model_index: pd.DataFrame,
    m124: Any,
) -> pd.DataFrame:
    unit_col = m124.find_col(
        static_raw.columns,
        m124.UNIT_ALIASES,
    )
    if unit_col is None:
        raise RuntimeError(
            "Cannot locate unit_id in Kyushu static table"
        )

    s = static_raw.copy()
    s["_unit_join"] = s[unit_col].astype(str)

    if s["_unit_join"].duplicated().any():
        raise RuntimeError(
            "Kyushu static table has duplicate unit_id rows"
        )

    idx = model_index[["model_row", "unit_id"]].copy()
    idx["_unit_join"] = idx["unit_id"].astype(str)

    merged = idx.merge(
        s,
        on="_unit_join",
        how="left",
        validate="one_to_one",
        suffixes=("_model", ""),
        sort=False,
    )

    if len(merged) != len(model_index):
        raise RuntimeError(
            "Kyushu static merge row-count mismatch"
        )

    merged = merged.sort_values(
        "model_row",
        kind="mergesort",
    ).reset_index(drop=True)

    return merged


# =============================================================================
# 4. RAINFALL SUMMARY FUNCTIONS
# =============================================================================

RAIN_SUMMARY_NAMES = [
    "mean",
    "std",
    "max",
    "min",
    "last",
    "p90",
    "peak_position_fraction",
]


def summarize_rain(
    x: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    x: [N,70,10] on the common Hiroshima-fitted model-input scale.
    Returns each summary as [N,10].
    """
    z = np.asarray(x, dtype=np.float64)

    if z.ndim != 3 or z.shape[1:] != (70, 10):
        raise RuntimeError(
            f"bad rain shape={z.shape}"
        )

    peak = np.argmax(
        z,
        axis=1,
    ).astype(np.float64) / 69.0

    return {
        "mean": np.mean(z, axis=1),
        "std": np.std(z, axis=1),
        "max": np.max(z, axis=1),
        "min": np.min(z, axis=1),
        "last": z[:, -1, :],
        "p90": np.quantile(z, 0.90, axis=1),
        "peak_position_fraction": peak,
    }


# =============================================================================
# 5. MATCHED-CONTEXT GEOMETRY
# =============================================================================

def set_mean_abs_deviation_static(
    x: np.ndarray,
    pairs: np.ndarray,
) -> np.ndarray:
    """
    x [N,91] continuous model channels only.
    returns [Nsets,91].
    """
    tri = np.asarray(x, dtype=np.float64)[pairs]
    mu = tri.mean(axis=1, keepdims=True)
    return np.abs(tri - mu).mean(axis=1)


def set_mean_abs_deviation_rain(
    x: np.ndarray,
    pairs: np.ndarray,
) -> np.ndarray:
    """
    x [N,70,10]
    For each matched set and rainfall field, average |candidate - set mean|
    over the 3 candidates and 70 time steps.
    returns [Nsets,10].
    """
    tri = np.asarray(x, dtype=np.float64)[pairs]  # [S,3,70,10]
    mu = tri.mean(axis=1, keepdims=True)
    abs_resid = np.abs(tri - mu)
    return abs_resid.mean(axis=(1, 2))


def set_max_abs_deviation_rain(
    x: np.ndarray,
    pairs: np.ndarray,
) -> np.ndarray:
    tri = np.asarray(x, dtype=np.float64)[pairs]
    mu = tri.mean(axis=1, keepdims=True)
    return np.abs(tri - mu).max(axis=(1, 2))


# =============================================================================
# 6. OUTCOME-CONDITIONED RELATIVE SIGNAL
# =============================================================================

def p_minus_control_mean_static(
    x: np.ndarray,
    pairs: np.ndarray,
) -> np.ndarray:
    """
    Assumes pair tensor is ordered [inventory-positive, control1, control2].
    Returns [Nsets,91].
    """
    tri = np.asarray(x, dtype=np.float64)[pairs]
    return tri[:, 0, :] - tri[:, 1:, :].mean(axis=1)


def p_minus_control_mean_rain(
    x: np.ndarray,
    pairs: np.ndarray,
) -> np.ndarray:
    """
    Returns per-set per-field mean temporal contrast:
      mean_t(positive - mean(control1,control2))
    => [Nsets,10]
    """
    tri = np.asarray(x, dtype=np.float64)[pairs]
    d = tri[:, 0, :, :] - tri[:, 1:, :, :].mean(axis=1)
    return d.mean(axis=1)


def vector_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> Dict[str, float]:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)

    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    if len(x) < 3:
        return {
            "n_features": int(len(x)),
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "cosine_similarity": np.nan,
            "median_sign_agreement": np.nan,
        }

    pr = pearsonr(x, y)
    sr = spearmanr(x, y)

    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))

    cos = (
        float(np.dot(x, y) / (nx * ny))
        if nx > 0 and ny > 0
        else np.nan
    )

    nonzero = (np.abs(x) > 1e-12) | (np.abs(y) > 1e-12)
    if nonzero.any():
        sign_agree = float(
            (np.sign(x[nonzero]) == np.sign(y[nonzero])).mean()
        )
    else:
        sign_agree = np.nan

    return {
        "n_features": int(len(x)),
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "spearman_rho": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
        "cosine_similarity": cos,
        "median_sign_agreement": sign_agree,
    }


# =============================================================================
# 7. FIGURES
# =============================================================================

def make_figures(
    static_shift: pd.DataFrame,
    rain_profile: pd.DataFrame,
    context_static: pd.DataFrame,
    context_rain: pd.DataFrame,
    contrast_df: pd.DataFrame,
    rain_fields: Sequence[str],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        log(f"WARNING matplotlib unavailable: {exc}")
        return

    # Figure 1: top 20 static shifts.
    top = (
        static_shift
        .sort_values("wasserstein", ascending=False)
        .head(20)
        .sort_values("wasserstein", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        top["feature"],
        top["wasserstein"],
    )
    ax.set_xlabel(
        "Wasserstein distance on Hiroshima-fitted model scale"
    )
    ax.set_title(
        "Top static input shifts: Hiroshima 2018 vs Kyushu 2017"
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        OUT / "126_FIG_STATIC_SHIFT_TOP20.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        OUT / "126_FIG_STATIC_SHIFT_TOP20.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    # Figure 2: 10 x 70 mean rainfall profile difference.
    piv = (
        rain_profile
        .pivot(
            index="rain_field",
            columns="time_index",
            values="kyushu_minus_hiroshima_mean",
        )
        .reindex(list(rain_fields))
    )

    fig, ax = plt.subplots(figsize=(12, 5.5))
    im = ax.imshow(
        piv.to_numpy(),
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_yticks(
        np.arange(len(piv.index))
    )
    ax.set_yticklabels(
        piv.index,
        fontsize=8,
    )
    ax.set_xlabel("70-step time index")
    ax.set_title(
        "Rainfall-process domain shift on Hiroshima-fitted model scale"
    )
    fig.colorbar(
        im,
        ax=ax,
        label="Kyushu mean − Hiroshima mean",
    )
    fig.tight_layout()
    fig.savefig(
        OUT / "126_FIG_RAIN_TEMPORAL_PROFILE_SHIFT.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        OUT / "126_FIG_RAIN_TEMPORAL_PROFILE_SHIFT.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    # Figure 3: matched-context heterogeneity shift.
    cr = (
        context_rain
        .sort_values("delta_median", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        cr["feature"],
        cr["delta_median"],
    )
    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1,
    )
    ax.set_xlabel(
        "Kyushu − Hiroshima median within-set rainfall MAD"
    )
    ax.set_title(
        "Matched-context rainfall heterogeneity shift"
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        OUT / "126_FIG_MATCHED_CONTEXT_RAIN_SHIFT.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        OUT / "126_FIG_MATCHED_CONTEXT_RAIN_SHIFT.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    # Figure 4: event-to-event preservation of feature-level relative contrast.
    plot = contrast_df[
        contrast_df["analysis_group"].isin(
            ["STATIC", "RAIN_TEMPORAL_MEAN"]
        )
    ].copy()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        plot["hiroshima_median_contrast"],
        plot["kyushu_median_contrast"],
        s=28,
        alpha=0.8,
    )
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel(
        "Hiroshima median inventory-positive − control contrast"
    )
    ax.set_ylabel(
        "Kyushu median inventory-positive − control contrast"
    )
    ax.set_title(
        "Cross-event preservation of matched-relative signal"
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        OUT / "126_FIG_RELATIVE_SIGNAL_PRESERVATION.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        OUT / "126_FIG_RELATIVE_SIGNAL_PRESERVATION.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


# =============================================================================
# 8. MAIN
# =============================================================================

def main() -> int:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {OUT}"
        )

    OUT.mkdir(parents=True, exist_ok=True)

    (
        p124,
        p120,
        p26,
        m124,
        m120,
        m26,
        bundle,
        runner,
        base,
    ) = load_authorities()

    protocol = {
        "experiment": "126_DOMAIN_SHIFT_CHARACTERIZATION",
        "status": "POST_HOC_DESCRIPTIVE_ANALYSIS",
        "development_event": "Hiroshima_2018",
        "external_event": "Kyushu_2017_Asakura_Toho",
        "predictive_model_training": False,
        "predictive_model_selection": False,
        "post_external_rescue": False,
        "primary_shift_analyses_use_outcome_labels": False,
        "secondary_relative_signal_analysis_uses_inventory_role": True,
        "secondary_label_semantics": (
            "inventory-positive versus matched non-inventory controls; "
            "controls are not interpreted as confirmed geological absences"
        ),
        "common_scale": (
            "exact full-Hiroshima-fitted preprocessing used for 124D/125"
        ),
        "authority_files": {
            "124D": {
                "path": str(p124),
                "sha256": sha256(p124),
            },
            "120B": {
                "path": str(p120),
                "sha256": sha256(p120),
            },
            "26": {
                "path": str(p26),
                "sha256": sha256(p26),
            },
            "23D": {
                "path": str(Path(runner.__file__)),
                "sha256": sha256(Path(runner.__file__)),
            },
            "20A": {
                "path": str(Path(base.__file__)),
                "sha256": sha256(Path(base.__file__)),
            },
        },
    }

    jwrite(
        OUT / "126_PROTOCOL.json",
        protocol,
    )

    log("=" * 120)
    log("START 126 DOMAIN-SHIFT CHARACTERIZATION")
    log("PREDICTIVE_MODEL_TRAINING=NO")
    log("PREDICTIVE_MODEL_SELECTION=NO")
    log("PRIMARY_SHIFT_ANALYSES_OUTCOME_LABELS=NO")
    log("SECONDARY_RELATIVE_SIGNAL_ANALYSIS=DESCRIPTIVE_ONLY")
    log("=" * 120)

    # -------------------------------------------------------------------------
    # A. Exact full-Hiroshima preprocessing used for deployment.
    # -------------------------------------------------------------------------
    (
        static_h,
        rain_h,
        full_pre,
        hprep_info,
    ) = m124.fit_full_hiroshima_preprocessor(
        bundle,
        runner,
        base,
        m26,
    )

    if static_h.shape != (base.N, 92):
        raise RuntimeError(
            f"Hiroshima static shape={static_h.shape}"
        )

    if rain_h.shape != (base.N, 70, 10):
        raise RuntimeError(
            f"Hiroshima rain shape={rain_h.shape}"
        )

    pair_h = np.asarray(
        bundle.pt,
        dtype=np.int64,
    )

    y_h = bundle.sample.y_pair.to_numpy(
        np.int8,
    )

    m120.verify_pair_order(
        pair_h,
        y_h,
        "HIROSHIMA_GLOBAL",
    )

    # -------------------------------------------------------------------------
    # B. Kyushu raw discovery and identical Hiroshima-fitted transform.
    # -------------------------------------------------------------------------
    ext = load_external_raw(
        m124,
        base,
        bundle,
    )

    (
        static_k,
        rain_k,
        kprep_audit,
    ) = m124.transform_external_with_hiroshima_preprocessor(
        ext["static_raw"],
        ext["dynamic_raw"],
        ext["model_index"],
        bundle,
        runner,
        m26,
        full_pre,
        base,
    )

    if static_k.shape != (EXPECTED_K_ROWS, 92):
        raise RuntimeError(
            f"Kyushu static shape={static_k.shape}"
        )

    if rain_k.shape != (EXPECTED_K_ROWS, 70, 10):
        raise RuntimeError(
            f"Kyushu rain shape={rain_k.shape}"
        )

    pair_k = np.asarray(
        ext["pair_k"],
        dtype=np.int64,
    )

    # External evaluation order verifies slot semantics P,C1,C2.
    eval_index, eval_rows = (
        m124.build_external_evaluation_order(
            ext["matched_raw"],
            ext["model_index"],
        )
    )

    if not np.array_equal(
        eval_rows,
        pair_k,
    ):
        # pair_k is label-blind set membership/order; eval_rows is role-verified.
        # If they differ only by within-set ordering, use eval_rows ONLY for the
        # secondary outcome-conditioned contrast; primary analyses remain pair_k.
        log(
            "NOTE_KYUSHU_LABEL_VERIFIED_PAIR_ORDER_DIFFERS_FROM_LABEL_BLIND_PAIR_ORDER=YES"
        )

    # -------------------------------------------------------------------------
    # C. STATIC label-agnostic model-input shift.
    # -------------------------------------------------------------------------
    static_fields = list(bundle.cont) + [
        "soil_missing_indicator"
    ]

    if len(static_fields) != 91:
        raise RuntimeError(
            f"Expected 91 continuous model channels; got {len(static_fields)}"
        )

    static_rows = []

    for j, field in enumerate(static_fields):
        rec = {
            "feature": field,
            "feature_index": j,
            "analysis_scale": (
                "Hiroshima-fitted robust model-input scale"
            ),
            **shift_metrics(
                static_h[:, j],
                static_k[:, j],
            ),
        }
        static_rows.append(rec)

    static_shift = pd.DataFrame(
        static_rows
    ).sort_values(
        "wasserstein",
        ascending=False,
        kind="mergesort",
    )

    static_shift.to_csv(
        OUT / "126_STATIC_MODEL_INPUT_SHIFT.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # D. CATEGORICAL shift (raw categories, not arbitrary numeric encoding).
    # -------------------------------------------------------------------------
    h_cat = (
        bundle.static[bundle.cat]
        .fillna("MISSING")
        .astype(str)
    )

    k_static_ordered = external_static_in_model_order(
        ext["static_raw"],
        ext["model_index"],
        m124,
    )

    if bundle.cat not in k_static_ordered.columns:
        raise RuntimeError(
            f"Kyushu static missing category field {bundle.cat}"
        )

    k_cat = (
        k_static_ordered[bundle.cat]
        .fillna("MISSING")
        .astype(str)
    )

    levels = sorted(
        set(h_cat.unique().tolist())
        | set(k_cat.unique().tolist())
    )

    h_counts = h_cat.value_counts().reindex(
        levels,
        fill_value=0,
    )
    k_counts = k_cat.value_counts().reindex(
        levels,
        fill_value=0,
    )

    h_p = h_counts.to_numpy(np.float64)
    k_p = k_counts.to_numpy(np.float64)
    h_p /= h_p.sum()
    k_p /= k_p.sum()

    js_distance = float(
        jensenshannon(
            h_p,
            k_p,
            base=2,
        )
    )

    cat_df = pd.DataFrame(
        {
            "category": levels,
            "hiroshima_count": h_counts.to_numpy(),
            "kyushu_count": k_counts.to_numpy(),
            "hiroshima_fraction": h_p,
            "kyushu_fraction": k_p,
            "kyushu_minus_hiroshima_fraction": k_p - h_p,
        }
    )

    cat_df.to_csv(
        OUT / "126_CATEGORY_SHIFT.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # E. RAINFALL per-sample trajectory summary shift.
    # -------------------------------------------------------------------------
    hsum = summarize_rain(
        rain_h,
    )
    ksum = summarize_rain(
        rain_k,
    )

    rain_summary_rows = []

    for summary_name in RAIN_SUMMARY_NAMES:
        for j, field in enumerate(base.RAIN_FIELDS):
            rain_summary_rows.append(
                {
                    "rain_field": field,
                    "summary": summary_name,
                    "analysis_scale": (
                        "Hiroshima-fitted model-input scale"
                    ),
                    **shift_metrics(
                        hsum[summary_name][:, j],
                        ksum[summary_name][:, j],
                    ),
                }
            )

    rain_summary = pd.DataFrame(
        rain_summary_rows
    )

    rain_summary.to_csv(
        OUT / "126_RAINFALL_PROCESS_SUMMARY_SHIFT.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Mean 70-step trajectory shift.
    h_profile = rain_h.mean(
        axis=0,
        dtype=np.float64,
    )  # [70,10]

    k_profile = rain_k.mean(
        axis=0,
        dtype=np.float64,
    )

    profile_rows = []

    for t in range(70):
        for j, field in enumerate(base.RAIN_FIELDS):
            profile_rows.append(
                {
                    "time_index": t,
                    "rain_field": field,
                    "hiroshima_mean": float(
                        h_profile[t, j]
                    ),
                    "kyushu_mean": float(
                        k_profile[t, j]
                    ),
                    "kyushu_minus_hiroshima_mean": float(
                        k_profile[t, j] - h_profile[t, j]
                    ),
                }
            )

    rain_profile = pd.DataFrame(
        profile_rows
    )

    rain_profile.to_csv(
        OUT / "126_RAINFALL_TEMPORAL_PROFILE_SHIFT.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rain_profile_summary_rows = []

    for j, field in enumerate(base.RAIN_FIELDS):
        a = h_profile[:, j]
        b = k_profile[:, j]
        diff = b - a

        if np.std(a) > 0 and np.std(b) > 0:
            corr = float(
                np.corrcoef(a, b)[0, 1]
            )
        else:
            corr = np.nan

        rain_profile_summary_rows.append(
            {
                "rain_field": field,
                "profile_mae": float(
                    np.mean(np.abs(diff))
                ),
                "profile_rmse": float(
                    np.sqrt(np.mean(diff ** 2))
                ),
                "profile_max_abs_difference": float(
                    np.max(np.abs(diff))
                ),
                "profile_correlation": corr,
            }
        )

    rain_profile_summary = pd.DataFrame(
        rain_profile_summary_rows
    ).sort_values(
        "profile_rmse",
        ascending=False,
    )

    rain_profile_summary.to_csv(
        OUT / "126_RAINFALL_TEMPORAL_PROFILE_SUMMARY.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # F. MATCHED-CONTEXT GEOMETRY shift (label-agnostic).
    # -------------------------------------------------------------------------
    h_static_mad = set_mean_abs_deviation_static(
        static_h[:, :91],
        pair_h,
    )
    k_static_mad = set_mean_abs_deviation_static(
        static_k[:, :91],
        pair_k,
    )

    context_static_rows = []

    for j, field in enumerate(static_fields):
        context_static_rows.append(
            {
                "feature": field,
                "geometry_measure": "within_set_mean_absolute_deviation",
                **shift_metrics(
                    h_static_mad[:, j],
                    k_static_mad[:, j],
                ),
            }
        )

    context_static = pd.DataFrame(
        context_static_rows
    ).sort_values(
        "wasserstein",
        ascending=False,
    )

    context_static.to_csv(
        OUT / "126_MATCHED_CONTEXT_STATIC_GEOMETRY_SHIFT.csv",
        index=False,
        encoding="utf-8-sig",
    )

    h_rain_mad = set_mean_abs_deviation_rain(
        rain_h,
        pair_h,
    )
    k_rain_mad = set_mean_abs_deviation_rain(
        rain_k,
        pair_k,
    )

    h_rain_max = set_max_abs_deviation_rain(
        rain_h,
        pair_h,
    )
    k_rain_max = set_max_abs_deviation_rain(
        rain_k,
        pair_k,
    )

    context_rain_rows = []

    for j, field in enumerate(base.RAIN_FIELDS):
        mad_rec = {
            "feature": field,
            "geometry_measure": "within_set_mean_absolute_deviation",
            **shift_metrics(
                h_rain_mad[:, j],
                k_rain_mad[:, j],
            ),
        }
        context_rain_rows.append(mad_rec)

        max_rec = {
            "feature": field,
            "geometry_measure": "within_set_max_absolute_deviation",
            **shift_metrics(
                h_rain_max[:, j],
                k_rain_max[:, j],
            ),
        }
        context_rain_rows.append(max_rec)

    context_rain_all = pd.DataFrame(
        context_rain_rows
    )

    context_rain_all.to_csv(
        OUT / "126_MATCHED_CONTEXT_RAIN_GEOMETRY_SHIFT_ALL.csv",
        index=False,
        encoding="utf-8-sig",
    )

    context_rain = (
        context_rain_all[
            context_rain_all[
                "geometry_measure"
            ]
            ==
            "within_set_mean_absolute_deviation"
        ]
        .sort_values(
            "wasserstein",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # G. SECONDARY outcome-conditioned relative-signal preservation.
    # -------------------------------------------------------------------------
    # Hiroshima pair tensor is verified P,C1,C2.
    # Kyushu uses role-verified eval_rows ONLY here.
    h_static_contrast = p_minus_control_mean_static(
        static_h[:, :91],
        pair_h,
    )
    k_static_contrast = p_minus_control_mean_static(
        static_k[:, :91],
        eval_rows,
    )

    h_rain_contrast = p_minus_control_mean_rain(
        rain_h,
        pair_h,
    )
    k_rain_contrast = p_minus_control_mean_rain(
        rain_k,
        eval_rows,
    )

    contrast_rows = []

    h_static_med = np.median(
        h_static_contrast,
        axis=0,
    )
    k_static_med = np.median(
        k_static_contrast,
        axis=0,
    )

    for j, field in enumerate(static_fields):
        contrast_rows.append(
            {
                "analysis_group": "STATIC",
                "feature": field,
                "hiroshima_median_contrast": float(
                    h_static_med[j]
                ),
                "kyushu_median_contrast": float(
                    k_static_med[j]
                ),
                "absolute_cross_event_contrast_change": float(
                    abs(
                        k_static_med[j]
                        -
                        h_static_med[j]
                    )
                ),
                "same_median_direction": bool(
                    np.sign(
                        h_static_med[j]
                    )
                    ==
                    np.sign(
                        k_static_med[j]
                    )
                ),
            }
        )

    h_rain_med = np.median(
        h_rain_contrast,
        axis=0,
    )
    k_rain_med = np.median(
        k_rain_contrast,
        axis=0,
    )

    for j, field in enumerate(base.RAIN_FIELDS):
        contrast_rows.append(
            {
                "analysis_group": "RAIN_TEMPORAL_MEAN",
                "feature": field,
                "hiroshima_median_contrast": float(
                    h_rain_med[j]
                ),
                "kyushu_median_contrast": float(
                    k_rain_med[j]
                ),
                "absolute_cross_event_contrast_change": float(
                    abs(
                        k_rain_med[j]
                        -
                        h_rain_med[j]
                    )
                ),
                "same_median_direction": bool(
                    np.sign(
                        h_rain_med[j]
                    )
                    ==
                    np.sign(
                        k_rain_med[j]
                    )
                ),
            }
        )

    contrast_df = pd.DataFrame(
        contrast_rows
    )

    contrast_df.to_csv(
        OUT / "126_RELATIVE_SIGNAL_PRESERVATION_BY_FEATURE.csv",
        index=False,
        encoding="utf-8-sig",
    )

    static_similarity = vector_similarity(
        h_static_med,
        k_static_med,
    )

    rain_similarity = vector_similarity(
        h_rain_med,
        k_rain_med,
    )

    combined_similarity = vector_similarity(
        np.concatenate(
            [
                h_static_med,
                h_rain_med,
            ]
        ),
        np.concatenate(
            [
                k_static_med,
                k_rain_med,
            ]
        ),
    )

    # -------------------------------------------------------------------------
    # H. Existing 125 internal-vs-external performance context.
    # -------------------------------------------------------------------------
    transfer_path = (
        OUT125
        / "125_INTERNAL_VS_EXTERNAL_TRANSFER_GAP.csv"
    )

    if not transfer_path.exists():
        raise FileNotFoundError(
            transfer_path
        )

    transfer = pd.read_csv(
        transfer_path,
        low_memory=False,
    )

    gap_cols = [
        c for c in transfer.columns
        if c.startswith(
            "EXTERNAL_MINUS_INTERNAL_"
        )
    ]

    gap_values = (
        transfer[gap_cols]
        .to_numpy(np.float64)
        .reshape(-1)
    )

    negative_gap_count = int(
        np.sum(gap_values < 0)
    )

    nonnegative_gap_count = int(
        np.sum(gap_values >= 0)
    )

    # -------------------------------------------------------------------------
    # I. SUMMARY.
    # -------------------------------------------------------------------------
    summary = {
        "status": "PASS_126_DOMAIN_SHIFT_CHARACTERIZATION_COMPLETE",
        "analysis_type": (
            "post-hoc descriptive characterization; no predictive model training"
        ),
        "events": {
            "development": "Hiroshima_2018",
            "external": "Kyushu_2017_Asakura_Toho",
        },
        "sample_counts": {
            "hiroshima_matched_samples": int(base.N),
            "hiroshima_matched_sets": int(len(pair_h)),
            "kyushu_matched_samples": int(len(static_k)),
            "kyushu_matched_sets": int(len(pair_k)),
        },
        "absolute_performance_context_from_125": {
            "external_minus_internal_metric_cells_negative": negative_gap_count,
            "external_minus_internal_metric_cells_nonnegative": nonnegative_gap_count,
            "total_metric_cells": int(len(gap_values)),
        },
        "static_shift": {
            "feature_count": int(len(static_shift)),
            "median_wasserstein": float(
                static_shift["wasserstein"].median()
            ),
            "q90_wasserstein": float(
                static_shift["wasserstein"].quantile(0.90)
            ),
            "max_wasserstein": float(
                static_shift["wasserstein"].max()
            ),
            "median_ks_statistic": float(
                static_shift["ks_statistic"].median()
            ),
            "median_psi": float(
                static_shift["psi_hiroshima_bins"].median()
            ),
            "top15_by_wasserstein": top_records(
                static_shift,
                "wasserstein",
                15,
                [
                    "feature",
                    "delta_median",
                    "ks_statistic",
                    "psi_hiroshima_bins",
                ],
            ),
        },
        "categorical_shift": {
            "field": bundle.cat,
            "jensen_shannon_distance_base2": js_distance,
            "hiroshima_unique_categories": int(
                h_cat.nunique()
            ),
            "kyushu_unique_categories": int(
                k_cat.nunique()
            ),
        },
        "rainfall_process_shift": {
            "summary_rows": int(
                len(rain_summary)
            ),
            "median_wasserstein_across_all_summary_field_cells": float(
                rain_summary["wasserstein"].median()
            ),
            "q90_wasserstein_across_all_summary_field_cells": float(
                rain_summary["wasserstein"].quantile(0.90)
            ),
            "top15_summary_cells_by_wasserstein": top_records(
                rain_summary,
                "wasserstein",
                15,
                [
                    "rain_field",
                    "summary",
                    "delta_median",
                    "ks_statistic",
                    "psi_hiroshima_bins",
                ],
            ),
            "top_temporal_profile_fields_by_rmse": (
                rain_profile_summary
                .head(10)
                .to_dict(orient="records")
            ),
        },
        "matched_context_geometry_shift": {
            "static_median_wasserstein": float(
                context_static["wasserstein"].median()
            ),
            "static_q90_wasserstein": float(
                context_static["wasserstein"].quantile(0.90)
            ),
            "rain_mean_abs_dev_median_wasserstein": float(
                context_rain["wasserstein"].median()
            ),
            "rain_mean_abs_dev_q90_wasserstein": float(
                context_rain["wasserstein"].quantile(0.90)
            ),
        },
        "relative_signal_preservation": {
            "interpretation": (
                "inventory-positive minus mean matched-control contrast; "
                "descriptive only, controls are non-inventory controls"
            ),
            "static": static_similarity,
            "rain_temporal_mean": rain_similarity,
            "combined": combined_similarity,
        },
        "scientific_guardrail": (
            "126 characterizes observed domain shift and relative-signal preservation. "
            "It does not claim causal explanation and does not modify 119/120B/121/123/124D/125."
        ),
        "next_step": (
            "Interpret 126; then run 127 spatial matched-margin visualization."
        ),
    }

    jwrite(
        OUT / "126_DOMAIN_SHIFT_SUMMARY.json",
        summary,
    )

    # -------------------------------------------------------------------------
    # J. REPORT.
    # -------------------------------------------------------------------------
    report = f"""# 126 Hiroshima–Kyushu Domain-Shift Characterization

## Status

**PASS_126_DOMAIN_SHIFT_CHARACTERIZATION_COMPLETE**

This is a post-hoc descriptive analysis. No predictive model was trained,
selected, tuned, refit, or rescued.

## Absolute performance context

Negative external-minus-internal metric cells:
**{negative_gap_count}/{len(gap_values)}**

## Static model-input shift

- features: {len(static_shift)}
- median Wasserstein distance: {summary['static_shift']['median_wasserstein']:.6f}
- 90th percentile Wasserstein distance: {summary['static_shift']['q90_wasserstein']:.6f}
- maximum Wasserstein distance: {summary['static_shift']['max_wasserstein']:.6f}
- categorical Jensen-Shannon distance: {js_distance:.6f}

## Rainfall-process shift

- summary-field cells: {len(rain_summary)}
- median Wasserstein distance: {summary['rainfall_process_shift']['median_wasserstein_across_all_summary_field_cells']:.6f}
- 90th percentile Wasserstein distance: {summary['rainfall_process_shift']['q90_wasserstein_across_all_summary_field_cells']:.6f}

## Matched-context geometry shift

- static MAD median Wasserstein: {summary['matched_context_geometry_shift']['static_median_wasserstein']:.6f}
- rainfall MAD median Wasserstein: {summary['matched_context_geometry_shift']['rain_mean_abs_dev_median_wasserstein']:.6f}

## Relative-signal preservation

### Static
- Pearson r: {static_similarity['pearson_r']:.6f}
- Spearman rho: {static_similarity['spearman_rho']:.6f}
- cosine similarity: {static_similarity['cosine_similarity']:.6f}
- sign agreement: {static_similarity['median_sign_agreement']:.6f}

### Rainfall temporal-mean
- Pearson r: {rain_similarity['pearson_r']:.6f}
- Spearman rho: {rain_similarity['spearman_rho']:.6f}
- cosine similarity: {rain_similarity['cosine_similarity']:.6f}
- sign agreement: {rain_similarity['median_sign_agreement']:.6f}

### Combined
- Pearson r: {combined_similarity['pearson_r']:.6f}
- Spearman rho: {combined_similarity['spearman_rho']:.6f}
- cosine similarity: {combined_similarity['cosine_similarity']:.6f}
- sign agreement: {combined_similarity['median_sign_agreement']:.6f}

## Interpretation guardrail

The analysis may support statements about observed covariate/process shift and
preservation or alteration of inventory-positive versus matched-control
relative contrasts. It must not be presented as proof of causal mechanisms or
as confirmation that controls are true geological negatives.
"""

    (
        OUT
        / "126_DOMAIN_SHIFT_REPORT.md"
    ).write_text(
        report,
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # K. FIGURES.
    # -------------------------------------------------------------------------
    make_figures(
        static_shift,
        rain_profile,
        context_static,
        context_rain,
        contrast_df,
        base.RAIN_FIELDS,
    )

    log("")
    log("=" * 120)
    log("PASS_126_DOMAIN_SHIFT_CHARACTERIZATION_COMPLETE")
    log(
        f"ABSOLUTE_EXTERNAL_MINUS_INTERNAL_NEGATIVE="
        f"{negative_gap_count}/{len(gap_values)}"
    )
    log(
        f"STATIC_MEDIAN_WASSERSTEIN="
        f"{summary['static_shift']['median_wasserstein']:.6f}"
    )
    log(
        f"STATIC_Q90_WASSERSTEIN="
        f"{summary['static_shift']['q90_wasserstein']:.6f}"
    )
    log(
        f"CATEGORY_JS_DISTANCE="
        f"{js_distance:.6f}"
    )
    log(
        f"RAIN_SUMMARY_MEDIAN_WASSERSTEIN="
        f"{summary['rainfall_process_shift']['median_wasserstein_across_all_summary_field_cells']:.6f}"
    )
    log(
        f"REL_SIGNAL_STATIC_SPEARMAN="
        f"{static_similarity['spearman_rho']:.6f}"
    )
    log(
        f"REL_SIGNAL_RAIN_SPEARMAN="
        f"{rain_similarity['spearman_rho']:.6f}"
    )
    log(
        f"REL_SIGNAL_COMBINED_SPEARMAN="
        f"{combined_similarity['spearman_rho']:.6f}"
    )
    log(
        f"REL_SIGNAL_COMBINED_COSINE="
        f"{combined_similarity['cosine_similarity']:.6f}"
    )
    log(f"OUTPUT={OUT}")
    log("NEXT_STEP=SEND_126_DOMAIN_SHIFT_SUMMARY_JSON_AND_FINAL_SCREEN")
    log("=" * 120)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        err = {
            "status": "FAIL_126_DOMAIN_SHIFT_CHARACTERIZATION",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        jwrite(
            OUT / "126_FAILURE.json",
            err,
        )
        print(traceback.format_exc(), flush=True)
        raise SystemExit(1)
