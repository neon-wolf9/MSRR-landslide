#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
127B_RUN_MSRR_SPATIAL_MATCHED_MARGIN_ANALYSIS_AUTO_GEOMETRY.py

Final spatial analysis for the frozen MSRR paper
================================================

Purpose
-------
This script spatializes ONLY the already-frozen matched-set predictions.

It does NOT:
- train/refit/tune/select any predictive model;
- generate arbitrary-grid landslide probabilities;
- reinterpret non-inventory controls as confirmed negatives;
- change the 119/120B/121/123/124D/125/126 results.

Primary spatial quantity
------------------------
For each matched set S_i = {P_i, C_i1, C_i2} and each representation:

    M_i = score(P_i) - max(score(C_i1), score(C_i2))

Then:

    DeltaM_i = M_i(MSRR) - M_i(RAW)

M_i > 0 means the inventory-supported landslide receives a higher score than
both matched non-inventory controls.

The spatial anchor of each set is the inventory-supported landslide grid.
Therefore these maps are MATCHED-SET SPATIAL PRODUCTS, not wall-to-wall
probability maps.

Inputs
------
Internal Hiroshima OOF row-level predictions:
<PROJECT_ROOT>\experiments\MSRR_121_PAIRED_BOOTSTRAP\
    120B_RECOVERED_ROW_LEVEL_OOF.parquet

External Kyushu full-panel predictions:
<PROJECT_ROOT>\external\kyushu_2017_asakura_toho\
    100_external_validation\MSRR_KYUSHU2017_EXTERNAL_FULL_PANEL_V2\
    125_EXTERNAL_PREDICTIONS_FULL_PANEL_WITH_LABELS.parquet

Reference metrics:
<PROJECT_ROOT>\external\kyushu_2017_asakura_toho\
    100_external_validation\MSRR_KYUSHU2017_EXTERNAL_FULL_PANEL_V2\
    125_INTERNAL_VS_EXTERNAL_TRANSFER_GAP.csv

Hiroshima geometry:
<PROJECT_ROOT>\hiroshima_grid_250m_master.gpkg
layer hiroshima_grid_250m_master

Kyushu geometry:
auto-discovered under
<PROJECT_ROOT>\external\kyushu_2017_asakura_toho
using anchor-unit coverage. You can override it with --kyushu-grid.

Outputs
-------
<PROJECT_ROOT>\experiments\MSRR_127_SPATIAL_MATCHED_MARGIN

Key outputs:
- 127_SPATIAL_SUMMARY.json
- 127_EVENT_XGB_SPATIAL_SUMMARY.csv
- 127_CROSS_BACKBONE_SPATIAL_CONSENSUS.csv
- 127_SPATIAL_BLOCK_AUDIT.csv
- 127_HIROSHIMA_SPATIAL_RESULTS.gpkg
- 127_KYUSHU_SPATIAL_RESULTS.gpkg

Figures:
- XGBoost DeltaM maps for Hiroshima and Kyushu
- XGBoost strict-pair transition maps for Hiroshima and Kyushu
- cross-backbone consensus maps for Hiroshima and Kyushu

Run
---
python ^
  <PROJECT_ROOT>\scripts\127B_RUN_MSRR_SPATIAL_MATCHED_MARGIN_ANALYSIS_AUTO_GEOMETRY.py

Optional manual Kyushu geometry:
python ^
  <PROJECT_ROOT>\scripts\127B_RUN_MSRR_SPATIAL_MATCHED_MARGIN_ANALYSIS_AUTO_GEOMETRY.py ^
  --kyushu-grid "<PATH_TO_KYUSHU_GRID>"

Scientific terminology
----------------------
"positive" = inventory-supported landslide location.
"control" = matched non-inventory control, NOT confirmed geological absence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# 0. CONSTANTS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

INTERNAL_OOF_EXACT = (
    ROOT
    / "experiments"
    / "MSRR_121_PAIRED_BOOTSTRAP"
    / "120B_RECOVERED_ROW_LEVEL_OOF.parquet"
)

OUT125 = (
    ROOT
    / "external"
    / "kyushu_2017_asakura_toho"
    / "100_external_validation"
    / "MSRR_KYUSHU2017_EXTERNAL_FULL_PANEL_V2"
)

EXTERNAL_PRED_EXACT = (
    OUT125
    / "125_EXTERNAL_PREDICTIONS_FULL_PANEL_WITH_LABELS.parquet"
)

TRANSFER_GAP_EXACT = (
    OUT125
    / "125_INTERNAL_VS_EXTERNAL_TRANSFER_GAP.csv"
)

HIROSHIMA_GRID_EXACT = (
    ROOT
    / "hiroshima_grid_250m_master.gpkg"
)

HIROSHIMA_LAYER = "hiroshima_grid_250m_master"

KYUSHU_ROOT = (
    ROOT
    / "external"
    / "kyushu_2017_asakura_toho"
)

OUT = (
    ROOT
    / "experiments"
    / "MSRR_127_SPATIAL_MATCHED_MARGIN_V2"
)

BACKBONES = [
    "ElasticNet-Logistic",
    "MLP",
    "HistGradientBoosting",
    "XGBoost",
]

REPRESENTATIONS = ["RAW", "MSRR"]

EXPECTED = {
    "Hiroshima_2018": {
        "samples": 15168,
        "sets": 5056,
    },
    "Kyushu_2017_Asakura_Toho": {
        "samples": 5076,
        "sets": 1692,
    },
}

UNIT_ALIASES = [
    "unit_id",
    "grid_id",
    "cell_id",
    "unitid",
    "gridid",
]

PAIR_ALIASES = [
    "pair_set_id",
    "pair_id",
    "set_id",
    "triplet_id",
]


# =============================================================================
# 1. GENERAL UTILITIES
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="127 final spatial matched-margin analysis"
    )
    p.add_argument(
        "--hiroshima-grid",
        default=str(HIROSHIMA_GRID_EXACT),
        help="Hiroshima vector grid path",
    )
    p.add_argument(
        "--hiroshima-layer",
        default=HIROSHIMA_LAYER,
        help="Hiroshima GPKG layer",
    )
    p.add_argument(
        "--kyushu-grid",
        default="",
        help=(
            "Optional Kyushu vector grid path. If omitted, auto-discover "
            "under the Kyushu project root."
        ),
    )
    p.add_argument(
        "--kyushu-layer",
        default="",
        help="Optional Kyushu GPKG layer",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite previous 127 output directory",
    )
    p.add_argument(
        "--spatial-blocks",
        type=int,
        default=4,
        help="Number of coordinate quantile bins per axis for spatial audit",
    )
    return p.parse_args()


def log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (OUT / "127_RUN.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def fail(code: str, message: str) -> None:
    raise RuntimeError(f"{code}: {message}")


def jwrite(path: Path, payload: Any) -> None:
    def conv(x):
        if isinstance(x, dict):
            return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            if np.isnan(x):
                return None
            return float(x)
        if isinstance(x, np.bool_):
            return bool(x)
        return x

    path.write_text(
        json.dumps(
            conv(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize_id_value(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        try:
            f = float(s)
            if float(int(f)) == f:
                s = str(int(f))
        except Exception:
            pass
    return s


def normalize_id_series(s: pd.Series) -> pd.Series:
    return s.map(normalize_id_value)


def find_alias(columns: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    lower = {str(c).lower(): str(c) for c in columns}
    for a in aliases:
        if a.lower() in lower:
            return lower[a.lower()]
    return None


def locate_exact_or_recursive(
    exact: Path,
    root: Path,
    pattern: str,
) -> Path:
    if exact.exists():
        return exact
    cands = sorted(root.rglob(pattern))
    if not cands:
        raise FileNotFoundError(
            f"Missing {exact.name}; recursive pattern {pattern} found nothing under {root}"
        )
    if len(cands) > 1:
        log(
            f"NOTE_MULTIPLE_CANDIDATES_FOR_{exact.name}="
            + " | ".join(str(x) for x in cands[:20])
        )
    return cands[0]


def ensure_out(overwrite: bool) -> None:
    if OUT.exists():
        if overwrite:
            shutil.rmtree(OUT)
        elif any(OUT.iterdir()):
            raise FileExistsError(
                f"Output directory is non-empty: {OUT}\n"
                f"Use --overwrite only if you intentionally want to rerun visualization."
            )
    OUT.mkdir(parents=True, exist_ok=True)


def finite_stats(x: pd.Series) -> Dict[str, float]:
    a = pd.to_numeric(x, errors="coerce").to_numpy(np.float64)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "n": int(len(a)),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "q25": float(np.quantile(a, 0.25)),
        "q75": float(np.quantile(a, 0.75)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


# =============================================================================
# 2. PREDICTION INPUTS
# =============================================================================

def load_prediction_file(
    path: Path,
    event: str,
) -> pd.DataFrame:
    log(f"READ_PREDICTIONS {event}: {path}")

    df = pd.read_parquet(path)

    required = {
        "unit_id",
        "pair_set_id",
        "y_true",
        "backbone",
        "representation",
        "risk_score",
    }
    missing = required - set(df.columns)
    if missing:
        fail(
            "FAIL_127_PREDICTION_COLUMNS",
            f"{event}: missing={sorted(missing)}, columns={list(df.columns)}",
        )

    df = df.copy()
    df["unit_id"] = normalize_id_series(df["unit_id"])
    df["pair_set_id"] = normalize_id_series(df["pair_set_id"])
    df["backbone"] = df["backbone"].astype(str)
    df["representation"] = df["representation"].astype(str)
    df["y_true"] = pd.to_numeric(df["y_true"], errors="raise").astype(np.int8)
    df["risk_score"] = pd.to_numeric(
        df["risk_score"],
        errors="raise",
    ).astype(np.float64)

    if not np.isfinite(df["risk_score"].to_numpy()).all():
        fail(
            "FAIL_127_NONFINITE_RISK_SCORE",
            event,
        )

    unexpected_b = sorted(set(df["backbone"]) - set(BACKBONES))
    unexpected_r = sorted(set(df["representation"]) - set(REPRESENTATIONS))

    if unexpected_b or unexpected_r:
        fail(
            "FAIL_127_UNEXPECTED_METHOD_CELL",
            f"{event}: unexpected_backbone={unexpected_b}, unexpected_rep={unexpected_r}",
        )

    # Every method cell should contain the complete matched benchmark.
    counts = (
        df.groupby(["backbone", "representation"], dropna=False)
        .size()
        .reset_index(name="n")
    )

    expected_rows = EXPECTED[event]["samples"]

    if len(counts) != 8 or (counts["n"] != expected_rows).any():
        fail(
            "FAIL_127_METHOD_CELL_COUNTS",
            f"{event}\n{counts.to_string(index=False)}",
        )

    # Each method cell must preserve exactly one inventory positive + two controls.
    expected_sets = EXPECTED[event]["sets"]

    for b in BACKBONES:
        for r in REPRESENTATIONS:
            cell = df[
                (df["backbone"] == b)
                & (df["representation"] == r)
            ]

            g = (
                cell.groupby("pair_set_id", dropna=False)["y_true"]
                .agg(["size", "sum"])
            )

            if len(g) != expected_sets:
                fail(
                    "FAIL_127_PAIRSET_COUNT",
                    f"{event}/{b}/{r}: sets={len(g)} expected={expected_sets}",
                )

            if (g["size"] != 3).any() or (g["sum"] != 1).any():
                bad = g[
                    (g["size"] != 3)
                    | (g["sum"] != 1)
                ].head(20)
                fail(
                    "FAIL_127_PAIR_STRUCTURE",
                    f"{event}/{b}/{r}\n{bad.to_string()}",
                )

    log(
        f"PASS_PREDICTION_STRUCTURE {event}: rows={len(df)} "
        f"cells=8 sets_per_cell={expected_sets}"
    )

    return df


# =============================================================================
# 3. MATCHED-SET MARGINS
# =============================================================================

def cell_set_margin(
    cell: pd.DataFrame,
    backbone: str,
    representation: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for pair_id, g in cell.groupby(
        "pair_set_id",
        sort=False,
        dropna=False,
    ):
        p = g[g["y_true"] == 1]
        c = g[g["y_true"] == 0]

        if len(p) != 1 or len(c) != 2:
            fail(
                "FAIL_127_BAD_SET_DURING_MARGIN",
                f"{backbone}/{representation}/{pair_id}",
            )

        p_row = p.iloc[0]
        p_score = float(p_row["risk_score"])
        c_scores = c["risk_score"].to_numpy(np.float64)
        max_c = float(np.max(c_scores))
        margin = float(p_score - max_c)

        rows.append(
            {
                "pair_set_id": str(pair_id),
                "anchor_unit_id": str(p_row["unit_id"]),
                "backbone": backbone,
                "representation": representation,
                "positive_score": p_score,
                "max_control_score": max_c,
                "margin": margin,
                "strict_success": bool(margin > 0.0),
            }
        )

    return pd.DataFrame(rows)


def build_all_margins(
    pred: pd.DataFrame,
    event: str,
) -> pd.DataFrame:
    frames = []

    for b in BACKBONES:
        for r in REPRESENTATIONS:
            cell = pred[
                (pred["backbone"] == b)
                & (pred["representation"] == r)
            ]
            frames.append(
                cell_set_margin(
                    cell,
                    b,
                    r,
                )
            )

    out = pd.concat(
        frames,
        ignore_index=True,
    )

    expected = EXPECTED[event]["sets"] * 8
    if len(out) != expected:
        fail(
            "FAIL_127_MARGIN_ROW_COUNT",
            f"{event}: {len(out)} expected={expected}",
        )

    return out


def build_representation_delta(
    all_margin: pd.DataFrame,
    event: str,
) -> pd.DataFrame:
    raw = all_margin[
        all_margin["representation"] == "RAW"
    ].copy()
    msrr = all_margin[
        all_margin["representation"] == "MSRR"
    ].copy()

    raw = raw.rename(
        columns={
            "positive_score": "raw_positive_score",
            "max_control_score": "raw_max_control_score",
            "margin": "raw_margin",
            "strict_success": "raw_strict_success",
        }
    ).drop(columns=["representation"])

    msrr = msrr.rename(
        columns={
            "positive_score": "msrr_positive_score",
            "max_control_score": "msrr_max_control_score",
            "margin": "msrr_margin",
            "strict_success": "msrr_strict_success",
        }
    ).drop(columns=["representation"])

    key = [
        "pair_set_id",
        "anchor_unit_id",
        "backbone",
    ]

    out = raw.merge(
        msrr,
        on=key,
        how="inner",
        validate="one_to_one",
    )

    out["delta_margin"] = (
        out["msrr_margin"]
        - out["raw_margin"]
    )

    out["delta_positive"] = (
        out["delta_margin"] > 0.0
    )

    def transition(row: pd.Series) -> str:
        r = bool(row["raw_strict_success"])
        m = bool(row["msrr_strict_success"])
        if (not r) and m:
            return "Corrected by MSRR"
        if r and m:
            return "Retained success"
        if r and (not m):
            return "Lost by MSRR"
        return "Still unresolved"

    out["strict_pair_transition"] = out.apply(
        transition,
        axis=1,
    )

    expected = EXPECTED[event]["sets"] * 4
    if len(out) != expected:
        fail(
            "FAIL_127_DELTA_ROW_COUNT",
            f"{event}: {len(out)} expected={expected}",
        )

    return out


# =============================================================================
# 4. STRICTPAIR REPRODUCTION AUDIT
# =============================================================================

def strictpair_reproduction_audit(
    delta: pd.DataFrame,
    transfer_gap: pd.DataFrame,
    event: str,
) -> pd.DataFrame:
    rows = []

    internal = event == "Hiroshima_2018"

    for b in BACKBONES:
        d = delta[delta["backbone"] == b]

        observed_raw = float(
            d["raw_strict_success"].mean()
        )
        observed_msrr = float(
            d["msrr_strict_success"].mean()
        )

        ref = transfer_gap[
            (transfer_gap["backbone"] == b)
        ]

        if len(ref) != 2:
            fail(
                "FAIL_127_TRANSFER_REFERENCE",
                f"Backbone {b}: rows={len(ref)}",
            )

        ref_raw = ref[
            ref["representation"] == "RAW"
        ].iloc[0]
        ref_msrr = ref[
            ref["representation"] == "MSRR"
        ].iloc[0]

        prefix = (
            "INTERNAL"
            if internal
            else "EXTERNAL"
        )

        expected_raw = float(
            ref_raw[f"{prefix}_StrictPair"]
        )
        expected_msrr = float(
            ref_msrr[f"{prefix}_StrictPair"]
        )

        raw_abs_error = abs(
            observed_raw
            - expected_raw
        )
        msrr_abs_error = abs(
            observed_msrr
            - expected_msrr
        )

        rows.append(
            {
                "event": event,
                "backbone": b,
                "observed_raw_strictpair": observed_raw,
                "reference_raw_strictpair": expected_raw,
                "raw_abs_error": raw_abs_error,
                "observed_msrr_strictpair": observed_msrr,
                "reference_msrr_strictpair": expected_msrr,
                "msrr_abs_error": msrr_abs_error,
                "pass_tolerance_1e-6": bool(
                    raw_abs_error <= 1e-6
                    and msrr_abs_error <= 1e-6
                ),
            }
        )

    audit = pd.DataFrame(rows)

    if not audit["pass_tolerance_1e-6"].all():
        fail(
            "FAIL_127_STRICTPAIR_REPRODUCTION",
            audit.to_string(index=False),
        )

    return audit


# =============================================================================
# 5. GEOMETRY DISCOVERY
# =============================================================================

def import_geospatial():
    # Load Fiona before GeoPandas on Windows when it is available. This avoids
    # a native GDAL DLL load-order crash observed with the recorded wheels;
    # the backend and all scientific operations remain unchanged.
    try:
        import fiona  # noqa: F401
    except Exception:
        pass

    try:
        import geopandas as gpd
    except Exception as exc:
        raise RuntimeError(
            "GeoPandas is required for 127. "
            "Install it in the landslide environment before running 127. "
            f"Original import error: {exc}"
        )

    try:
        import shapely  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            f"Shapely is required for 127: {exc}"
        )

    return gpd


def read_vector_candidate(
    gpd: Any,
    path: Path,
    layer: str = "",
):
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return gpd.read_parquet(path)

    if suffix == ".gpkg":
        if layer:
            return gpd.read_file(
                path,
                layer=layer,
            )
        return gpd.read_file(path)

    return gpd.read_file(path)


def prepare_geometry(
    gdf,
    required_unit_ids: Sequence[str],
    label: str,
):
    if gdf is None or len(gdf) == 0:
        fail(
            "FAIL_127_EMPTY_GEOMETRY",
            label,
        )

    if "geometry" not in gdf.columns:
        fail(
            "FAIL_127_NO_GEOMETRY_COLUMN",
            label,
        )

    unit_col = find_alias(
        gdf.columns,
        UNIT_ALIASES,
    )

    if unit_col is None:
        fail(
            "FAIL_127_NO_UNIT_ID_GEOMETRY",
            f"{label}: columns={list(gdf.columns)}",
        )

    out = gdf[
        [unit_col, "geometry"]
    ].copy()

    out = out.rename(
        columns={unit_col: "unit_id"}
    )

    out["unit_id"] = normalize_id_series(
        out["unit_id"]
    )

    out = out[
        out["unit_id"] != ""
    ].copy()

    if out["unit_id"].duplicated().any():
        # Duplicate geometry rows for same unit are unsafe for a 1:1 join.
        dups = out[
            out["unit_id"].duplicated(
                keep=False
            )
        ]["unit_id"].head(20).tolist()
        fail(
            "FAIL_127_GEOMETRY_DUPLICATE_UNIT",
            f"{label}: example duplicates={dups}",
        )

    req = set(map(str, required_unit_ids))
    have = set(out["unit_id"])
    covered = len(req & have)
    coverage = covered / max(len(req), 1)

    if coverage < 0.99:
        missing = sorted(req - have)[:30]
        fail(
            "FAIL_127_GEOMETRY_LOW_COVERAGE",
            f"{label}: coverage={coverage:.6f}, missing_examples={missing}",
        )

    return out, {
        "rows": int(len(out)),
        "required_anchor_units": int(len(req)),
        "covered_anchor_units": int(covered),
        "coverage": float(coverage),
        "crs": str(out.crs),
    }


def read_hiroshima_geometry(
    gpd: Any,
    path: Path,
    layer: str,
    required_unit_ids: Sequence[str],
):
    """
    Read a specifically supplied Hiroshima geometry file.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    log(
        f"READ_HIROSHIMA_GEOMETRY path={path} layer={layer}"
    )

    try:
        gdf = gpd.read_file(
            path,
            layer=layer,
        )
    except Exception:
        log(
            "WARNING_HIROSHIMA_LAYER_READ_FAILED; trying default layer"
        )
        gdf = gpd.read_file(path)

    out, audit = prepare_geometry(
        gdf,
        required_unit_ids,
        "HIROSHIMA",
    )
    audit["path"] = str(path)
    audit["layer"] = layer
    audit["discovery"] = "explicit_existing_path"

    return out, audit


def discover_hiroshima_geometry(
    gpd: Any,
    requested_path: str,
    requested_layer: str,
    required_unit_ids: Sequence[str],
):
    """
    Robust technical recovery for 127B.

    If the originally assumed root-level master-grid path exists, use it.
    Otherwise search the project recursively and ACCEPT a candidate only if
    its unit_id coverage of the frozen Hiroshima positive anchors is >=99%.

    This changes no predictions, labels, margins, or scientific protocol.
    """
    p = Path(requested_path)

    if p.exists():
        return read_hiroshima_geometry(
            gpd,
            p,
            requested_layer,
            required_unit_ids,
        )

    log(
        f"HIROSHIMA_REQUESTED_GRID_NOT_FOUND={p}"
    )
    log(
        "HIROSHIMA_GEOMETRY_RECOVERY=SEARCH_PROJECT_AND_REQUIRE_99PCT_ANCHOR_COVERAGE"
    )

    req = set(map(str, required_unit_ids))

    # Strongly prioritized patterns first so we do not blindly inspect every
    # project vector file unless necessary.
    patterns = [
        "hiroshima_grid_250m_master.gpkg",
        "*hiroshima*250m*master*.gpkg",
        "*250m*master*.gpkg",
        "*hiroshima*grid*.gpkg",
        "*grid*250m*.gpkg",
        "*hiroshima*250m*.shp",
        "*hiroshima*grid*.shp",
        "*grid*250m*.shp",
        "*hiroshima*250m*.geojson",
        "*hiroshima*grid*.geojson",
    ]

    candidates: List[Path] = []
    seen = set()

    for pat in patterns:
        for cand in ROOT.rglob(pat):
            key = str(cand.resolve()).lower()
            if key not in seen:
                seen.add(key)
                candidates.append(cand)

    # Last-resort GPKGs are considered only if name-guided search failed.
    if not candidates:
        for cand in ROOT.rglob("*.gpkg"):
            key = str(cand.resolve()).lower()
            if key not in seen:
                seen.add(key)
                candidates.append(cand)

    candidates = sorted(
        candidates,
        key=vector_candidate_score,
    )

    if not candidates:
        fail(
            "FAIL_127B_NO_HIROSHIMA_VECTOR_CANDIDATES",
            f"No vector candidates found under {ROOT}",
        )

    attempts: List[Dict[str, Any]] = []

    for path in candidates:
        layer_names: List[str] = [""]

        if path.suffix.lower() == ".gpkg":
            try:
                import fiona
                ls = list(fiona.listlayers(path))
                if ls:
                    # Put the historically frozen layer name first when present.
                    layer_names = sorted(
                        ls,
                        key=lambda z: (
                            0 if str(z) == HIROSHIMA_LAYER else 1,
                            str(z),
                        ),
                    )
            except Exception:
                layer_names = [""]

        for layer in layer_names:
            try:
                gdf = read_vector_candidate(
                    gpd,
                    path,
                    layer,
                )

                unit_col = find_alias(
                    gdf.columns,
                    UNIT_ALIASES,
                )

                if unit_col is None or "geometry" not in gdf.columns:
                    attempts.append(
                        {
                            "path": str(path),
                            "layer": layer,
                            "status": "NO_UNIT_OR_GEOMETRY",
                            "coverage": 0.0,
                            "rows": int(len(gdf)),
                        }
                    )
                    continue

                ids = set(
                    normalize_id_series(
                        gdf[unit_col]
                    ).tolist()
                )

                coverage = len(req & ids) / max(len(req), 1)

                attempts.append(
                    {
                        "path": str(path),
                        "layer": layer,
                        "status": "READ",
                        "coverage": float(coverage),
                        "rows": int(len(gdf)),
                    }
                )

                if coverage >= 0.99:
                    out, audit = prepare_geometry(
                        gdf,
                        required_unit_ids,
                        "HIROSHIMA_AUTO",
                    )
                    audit["path"] = str(path)
                    audit["layer"] = layer
                    audit["discovery"] = "automatic_127B"

                    pd.DataFrame(attempts).to_csv(
                        OUT / "127B_HIROSHIMA_GEOMETRY_DISCOVERY_AUDIT.csv",
                        index=False,
                        encoding="utf-8-sig",
                    )

                    log(
                        f"PASS_HIROSHIMA_GEOMETRY_AUTO_DISCOVERY "
                        f"path={path} layer={layer} coverage={coverage:.6f}"
                    )

                    return out, audit

            except Exception as exc:
                attempts.append(
                    {
                        "path": str(path),
                        "layer": layer,
                        "status": (
                            f"ERROR_{type(exc).__name__}: {str(exc)[:240]}"
                        ),
                        "coverage": np.nan,
                    }
                )

    attempts_df = pd.DataFrame(attempts)
    attempts_df.to_csv(
        OUT / "127B_HIROSHIMA_GEOMETRY_DISCOVERY_AUDIT.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if len(attempts_df):
        best = (
            attempts_df
            .sort_values(
                "coverage",
                ascending=False,
                na_position="last",
            )
            .head(20)
        )
        best_text = best.to_string(index=False)
    else:
        best_text = "No readable candidates."

    fail(
        "FAIL_127B_HIROSHIMA_GEOMETRY_NOT_FOUND",
        "No candidate reached >=99% frozen-anchor unit_id coverage.\n"
        + best_text,
    )


def vector_candidate_score(path: Path) -> Tuple[int, str]:
    name = path.name.lower()
    score = 0
    if "master" in name:
        score += 50
    if "grid" in name:
        score += 40
    if "250" in name:
        score += 20
    if "static" in name:
        score += 5
    if "matched" in name:
        score += 2
    if "prediction" in name:
        score -= 100
    if "result" in name:
        score -= 50
    return (-score, str(path))


def discover_kyushu_geometry(
    gpd: Any,
    manual_path: str,
    manual_layer: str,
    required_unit_ids: Sequence[str],
):
    if manual_path:
        p = Path(manual_path)
        if not p.exists():
            raise FileNotFoundError(p)
        gdf = read_vector_candidate(
            gpd,
            p,
            manual_layer,
        )
        out, audit = prepare_geometry(
            gdf,
            required_unit_ids,
            "KYUSHU_MANUAL",
        )
        audit["path"] = str(p)
        audit["layer"] = manual_layer
        audit["discovery"] = "manual"
        return out, audit

    candidates: List[Path] = []

    for pat in [
        "*.gpkg",
        "*.geojson",
        "*.shp",
    ]:
        candidates.extend(
            KYUSHU_ROOT.rglob(pat)
        )

    # GeoParquet is allowed only for likely spatial grid files.
    for p in KYUSHU_ROOT.rglob("*.parquet"):
        low = p.name.lower()
        if (
            "grid" in low
            and (
                "master" in low
                or "geometry" in low
                or "spatial" in low
            )
        ):
            candidates.append(p)

    candidates = sorted(
        set(candidates),
        key=vector_candidate_score,
    )

    if not candidates:
        fail(
            "FAIL_127_NO_KYUSHU_VECTOR_CANDIDATES",
            str(KYUSHU_ROOT),
        )

    req = set(map(str, required_unit_ids))
    attempts = []

    for path in candidates:
        # If a GPKG has multiple layers, try all discoverable layers.
        layer_names: List[str] = [""]

        if path.suffix.lower() == ".gpkg":
            try:
                import fiona
                ls = list(fiona.listlayers(path))
                if ls:
                    layer_names = ls
            except Exception:
                layer_names = [""]

        for layer in layer_names:
            try:
                gdf = read_vector_candidate(
                    gpd,
                    path,
                    layer,
                )

                unit_col = find_alias(
                    gdf.columns,
                    UNIT_ALIASES,
                )

                if unit_col is None or "geometry" not in gdf.columns:
                    attempts.append(
                        {
                            "path": str(path),
                            "layer": layer,
                            "status": "NO_UNIT_OR_GEOMETRY",
                            "coverage": 0.0,
                        }
                    )
                    continue

                ids = set(
                    normalize_id_series(
                        gdf[unit_col]
                    ).tolist()
                )

                coverage = len(
                    req & ids
                ) / max(len(req), 1)

                attempts.append(
                    {
                        "path": str(path),
                        "layer": layer,
                        "status": "READ",
                        "coverage": float(coverage),
                        "rows": int(len(gdf)),
                    }
                )

                if coverage >= 0.99:
                    out, audit = prepare_geometry(
                        gdf,
                        required_unit_ids,
                        "KYUSHU_AUTO",
                    )
                    audit["path"] = str(path)
                    audit["layer"] = layer
                    audit["discovery"] = "automatic"

                    pd.DataFrame(attempts).to_csv(
                        OUT / "127_KYUSHU_GEOMETRY_DISCOVERY_AUDIT.csv",
                        index=False,
                        encoding="utf-8-sig",
                    )

                    return out, audit

            except Exception as exc:
                attempts.append(
                    {
                        "path": str(path),
                        "layer": layer,
                        "status": (
                            f"ERROR_{type(exc).__name__}: {str(exc)[:200]}"
                        ),
                        "coverage": np.nan,
                    }
                )

    pd.DataFrame(attempts).to_csv(
        OUT / "127_KYUSHU_GEOMETRY_DISCOVERY_AUDIT.csv",
        index=False,
        encoding="utf-8-sig",
    )

    best = (
        pd.DataFrame(attempts)
        .sort_values(
            "coverage",
            ascending=False,
            na_position="last",
        )
        .head(20)
    )

    fail(
        "FAIL_127_KYUSHU_GEOMETRY_NOT_FOUND",
        "No candidate reached >=99% anchor-unit coverage.\n"
        + best.to_string(index=False),
    )


# =============================================================================
# 6. JOIN SPATIAL RESULTS
# =============================================================================

def join_geometry(
    gpd: Any,
    table: pd.DataFrame,
    geom,
    label: str,
):
    g = geom.copy()

    g["unit_id"] = normalize_id_series(
        g["unit_id"]
    )

    t = table.copy()
    t["anchor_unit_id"] = normalize_id_series(
        t["anchor_unit_id"]
    )

    out = t.merge(
        g,
        left_on="anchor_unit_id",
        right_on="unit_id",
        how="left",
        validate="many_to_one",
    )

    if out["geometry"].isna().any():
        bad = out[
            out["geometry"].isna()
        ]["anchor_unit_id"].head(20).tolist()
        fail(
            "FAIL_127_SPATIAL_JOIN_MISSING_GEOMETRY",
            f"{label}: {bad}",
        )

    out = gpd.GeoDataFrame(
        out,
        geometry="geometry",
        crs=geom.crs,
    )

    return out


# =============================================================================
# 7. CROSS-BACKBONE CONSENSUS
# =============================================================================

def build_consensus(
    delta: pd.DataFrame,
    event: str,
) -> pd.DataFrame:
    # Every pair set must share the same positive anchor across all backbones.
    anchor_check = (
        delta.groupby("pair_set_id")["anchor_unit_id"]
        .nunique()
    )

    if (anchor_check != 1).any():
        fail(
            "FAIL_127_ANCHOR_MISMATCH_ACROSS_BACKBONES",
            f"{event}: bad_sets={(anchor_check != 1).sum()}",
        )

    grouped = delta.groupby(
        [
            "pair_set_id",
            "anchor_unit_id",
        ],
        sort=False,
    )

    out = grouped.agg(
        delta_margin_mean=(
            "delta_margin",
            "mean",
        ),
        delta_margin_median=(
            "delta_margin",
            "median",
        ),
        delta_margin_min=(
            "delta_margin",
            "min",
        ),
        delta_margin_max=(
            "delta_margin",
            "max",
        ),
        positive_backbones=(
            "delta_positive",
            "sum",
        ),
        raw_success_backbones=(
            "raw_strict_success",
            "sum",
        ),
        msrr_success_backbones=(
            "msrr_strict_success",
            "sum",
        ),
    ).reset_index()

    corrected = (
        delta.assign(
            corrected=(
                (~delta["raw_strict_success"])
                & delta["msrr_strict_success"]
            ),
            lost=(
                delta["raw_strict_success"]
                & (~delta["msrr_strict_success"])
            ),
        )
        .groupby(
            [
                "pair_set_id",
                "anchor_unit_id",
            ],
            sort=False,
        )[["corrected", "lost"]]
        .sum()
        .reset_index()
        .rename(
            columns={
                "corrected": "corrected_backbones",
                "lost": "lost_backbones",
            }
        )
    )

    out = out.merge(
        corrected,
        on=[
            "pair_set_id",
            "anchor_unit_id",
        ],
        how="left",
        validate="one_to_one",
    )

    for c in [
        "positive_backbones",
        "raw_success_backbones",
        "msrr_success_backbones",
        "corrected_backbones",
        "lost_backbones",
    ]:
        out[c] = out[c].astype(int)

    out["all4_delta_positive"] = (
        out["positive_backbones"] == 4
    )

    out["at_least3_delta_positive"] = (
        out["positive_backbones"] >= 3
    )

    if len(out) != EXPECTED[event]["sets"]:
        fail(
            "FAIL_127_CONSENSUS_SET_COUNT",
            f"{event}: {len(out)}",
        )

    return out


# =============================================================================
# 8. EVENT SUMMARY
# =============================================================================

def summarize_xgb(
    delta: pd.DataFrame,
    event: str,
) -> Dict[str, Any]:
    x = delta[
        delta["backbone"] == "XGBoost"
    ].copy()

    trans = (
        x["strict_pair_transition"]
        .value_counts()
        .reindex(
            [
                "Corrected by MSRR",
                "Retained success",
                "Lost by MSRR",
                "Still unresolved",
            ],
            fill_value=0,
        )
    )

    n = len(x)

    return {
        "event": event,
        "n_sets": int(n),
        "raw_strictpair": float(
            x["raw_strict_success"].mean()
        ),
        "msrr_strictpair": float(
            x["msrr_strict_success"].mean()
        ),
        "strictpair_gain": float(
            x["msrr_strict_success"].mean()
            - x["raw_strict_success"].mean()
        ),
        "delta_margin": finite_stats(
            x["delta_margin"]
        ),
        "delta_margin_positive_sets": int(
            x["delta_positive"].sum()
        ),
        "delta_margin_positive_fraction": float(
            x["delta_positive"].mean()
        ),
        "transition_counts": {
            str(k): int(v)
            for k, v in trans.items()
        },
        "transition_fractions": {
            str(k): float(v / n)
            for k, v in trans.items()
        },
    }


def summarize_consensus(
    consensus: pd.DataFrame,
    event: str,
) -> Dict[str, Any]:
    n = len(consensus)

    counts = (
        consensus["positive_backbones"]
        .value_counts()
        .sort_index()
    )

    return {
        "event": event,
        "n_sets": int(n),
        "delta_margin_median_across_backbones": finite_stats(
            consensus["delta_margin_median"]
        ),
        "sets_all4_backbones_delta_positive": int(
            consensus["all4_delta_positive"].sum()
        ),
        "fraction_all4_backbones_delta_positive": float(
            consensus["all4_delta_positive"].mean()
        ),
        "sets_at_least3_backbones_delta_positive": int(
            consensus["at_least3_delta_positive"].sum()
        ),
        "fraction_at_least3_backbones_delta_positive": float(
            consensus["at_least3_delta_positive"].mean()
        ),
        "positive_backbone_count_distribution": {
            str(int(k)): int(v)
            for k, v in counts.items()
        },
        "mean_positive_backbones_per_set": float(
            consensus["positive_backbones"].mean()
        ),
        "mean_corrected_backbones_per_set": float(
            consensus["corrected_backbones"].mean()
        ),
        "mean_lost_backbones_per_set": float(
            consensus["lost_backbones"].mean()
        ),
    }


# =============================================================================
# 9. SPATIAL BLOCK AUDIT
# =============================================================================

def point_coordinates(gdf):
    # Work in projected coordinates when possible.
    work = gdf.copy()

    try:
        if work.crs is not None and getattr(
            work.crs,
            "is_geographic",
            False,
        ):
            utm = work.estimate_utm_crs()
            if utm is not None:
                work = work.to_crs(utm)
    except Exception:
        pass

    pts = work.geometry.representative_point()

    return (
        pts.x.to_numpy(np.float64),
        pts.y.to_numpy(np.float64),
        str(work.crs),
    )


def qbin_labels(
    values: np.ndarray,
    q: int,
    prefix: str,
) -> pd.Series:
    s = pd.Series(values)

    try:
        b = pd.qcut(
            s,
            q=q,
            labels=False,
            duplicates="drop",
        )
    except Exception:
        b = pd.Series(
            np.zeros(len(s), dtype=int)
        )

    b = b.fillna(-1).astype(int)

    return b.map(
        lambda z: f"{prefix}{z+1}"
        if z >= 0
        else f"{prefix}NA"
    )


def build_spatial_block_audit(
    xgb_gdf,
    consensus_gdf,
    event: str,
    q: int,
) -> pd.DataFrame:
    # Use exactly one geometry per matched set.
    x = xgb_gdf.copy()

    xx, yy, coord_crs = point_coordinates(
        x
    )

    x["x_block"] = qbin_labels(
        xx,
        q,
        "X",
    )
    x["y_block"] = qbin_labels(
        yy,
        q,
        "Y",
    )
    x["spatial_block"] = (
        x["x_block"]
        + "_"
        + x["y_block"]
    )

    c = consensus_gdf[
        [
            "pair_set_id",
            "positive_backbones",
            "delta_margin_median",
            "all4_delta_positive",
            "at_least3_delta_positive",
        ]
    ].copy()

    x = x.merge(
        c,
        on="pair_set_id",
        how="left",
        validate="one_to_one",
    )

    rows = []

    for block, g in x.groupby(
        "spatial_block",
        sort=True,
    ):
        n = len(g)

        rows.append(
            {
                "event": event,
                "spatial_block": block,
                "coordinate_crs": coord_crs,
                "n_sets": int(n),
                "xgb_raw_strictpair": float(
                    g["raw_strict_success"].mean()
                ),
                "xgb_msrr_strictpair": float(
                    g["msrr_strict_success"].mean()
                ),
                "xgb_strictpair_gain": float(
                    g["msrr_strict_success"].mean()
                    - g["raw_strict_success"].mean()
                ),
                "xgb_delta_margin_mean": float(
                    g["delta_margin"].mean()
                ),
                "xgb_delta_margin_median": float(
                    g["delta_margin"].median()
                ),
                "xgb_delta_positive_fraction": float(
                    g["delta_positive"].mean()
                ),
                "xgb_corrected_fraction": float(
                    (
                        g["strict_pair_transition"]
                        == "Corrected by MSRR"
                    ).mean()
                ),
                "xgb_lost_fraction": float(
                    (
                        g["strict_pair_transition"]
                        == "Lost by MSRR"
                    ).mean()
                ),
                "consensus_delta_margin_median": float(
                    g["delta_margin_median"].median()
                ),
                "consensus_mean_positive_backbones": float(
                    g["positive_backbones"].mean()
                ),
                "consensus_all4_positive_fraction": float(
                    g["all4_delta_positive"].mean()
                ),
                "consensus_at_least3_positive_fraction": float(
                    g["at_least3_delta_positive"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# 10. FIGURES
# =============================================================================

def import_plotting():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            f"Matplotlib required for 127 figures: {exc}"
        )
    return plt


def point_view(gdf):
    work = gdf.copy()
    work["geometry"] = work.geometry.representative_point()
    return work


def save_numeric_map(
    plt,
    gdf,
    column: str,
    title: str,
    basename: str,
    legend_label: str,
) -> None:
    p = point_view(gdf)

    fig, ax = plt.subplots(
        figsize=(8.5, 7.0)
    )

    p.plot(
        ax=ax,
        column=column,
        legend=True,
        markersize=12,
        legend_kwds={
            "label": legend_label,
            "shrink": 0.75,
        },
    )

    ax.set_title(
        title,
        fontsize=12,
    )
    ax.set_axis_off()

    fig.tight_layout()

    fig.savefig(
        OUT / f"{basename}.png",
        dpi=350,
        bbox_inches="tight",
    )
    fig.savefig(
        OUT / f"{basename}.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def save_categorical_map(
    plt,
    gdf,
    column: str,
    title: str,
    basename: str,
) -> None:
    p = point_view(gdf)

    fig, ax = plt.subplots(
        figsize=(8.5, 7.0)
    )

    p.plot(
        ax=ax,
        column=column,
        categorical=True,
        legend=True,
        markersize=12,
    )

    ax.set_title(
        title,
        fontsize=12,
    )
    ax.set_axis_off()

    fig.tight_layout()

    fig.savefig(
        OUT / f"{basename}.png",
        dpi=350,
        bbox_inches="tight",
    )
    fig.savefig(
        OUT / f"{basename}.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def save_transition_count_bar(
    plt,
    event_summary: Dict[str, Any],
    basename: str,
) -> None:
    counts = event_summary["transition_counts"]

    labels = list(counts.keys())
    vals = [counts[k] for k in labels]

    fig, ax = plt.subplots(
        figsize=(8.5, 5.2)
    )

    ax.bar(
        labels,
        vals,
    )

    ax.set_ylabel("Matched sets")
    ax.set_title(
        f"{event_summary['event']}: XGBoost strict-pair transitions"
    )

    ax.tick_params(
        axis="x",
        rotation=20,
    )
    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        OUT / f"{basename}.png",
        dpi=350,
        bbox_inches="tight",
    )
    fig.savefig(
        OUT / f"{basename}.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# 11. GPKG WRITING
# =============================================================================

def write_event_gpkg(
    xgb_gdf,
    all_backbone_gdf,
    consensus_gdf,
    path: Path,
) -> None:
    if path.exists():
        path.unlink()

    xgb_gdf.to_file(
        path,
        layer="xgboost_set_margin",
        driver="GPKG",
    )

    consensus_gdf.to_file(
        path,
        layer="cross_backbone_consensus",
        driver="GPKG",
    )

    all_backbone_gdf.to_file(
        path,
        layer="all_backbone_set_margin",
        driver="GPKG",
    )


# =============================================================================
# 12. MAIN
# =============================================================================

def main() -> int:
    args = parse_args()
    ensure_out(args.overwrite)

    log("=" * 120)
    log("START 127B FINAL SPATIAL MATCHED-MARGIN ANALYSIS - AUTO GEOMETRY RECOVERY")
    log("PREDICTIVE_MODEL_TRAINING=NO")
    log("PREDICTIVE_MODEL_SELECTION=NO")
    log("ARBITRARY_GRID_INFERENCE=NO")
    log("CALIBRATED_PROBABILITY_MAP=NO")
    log("SPATIAL_UNIT=MATCHED_SET_ANCHORED_AT_INVENTORY_POSITIVE")
    log("=" * 120)

    gpd = import_geospatial()
    plt = import_plotting()

    internal_path = locate_exact_or_recursive(
        INTERNAL_OOF_EXACT,
        ROOT / "experiments",
        "120B_RECOVERED_ROW_LEVEL_OOF.parquet",
    )

    external_path = locate_exact_or_recursive(
        EXTERNAL_PRED_EXACT,
        OUT125,
        "125_EXTERNAL_PREDICTIONS_FULL_PANEL_WITH_LABELS.parquet",
    )

    if not TRANSFER_GAP_EXACT.exists():
        raise FileNotFoundError(
            TRANSFER_GAP_EXACT
        )

    transfer_gap = pd.read_csv(
        TRANSFER_GAP_EXACT,
        low_memory=False,
    )

    required_transfer = {
        "backbone",
        "representation",
        "INTERNAL_StrictPair",
        "EXTERNAL_StrictPair",
    }

    if not required_transfer.issubset(
        transfer_gap.columns
    ):
        fail(
            "FAIL_127_TRANSFER_GAP_COLUMNS",
            str(list(transfer_gap.columns)),
        )

    # -------------------------------------------------------------------------
    # A. Read frozen predictions.
    # -------------------------------------------------------------------------
    pred_h = load_prediction_file(
        internal_path,
        "Hiroshima_2018",
    )

    pred_k = load_prediction_file(
        external_path,
        "Kyushu_2017_Asakura_Toho",
    )

    # -------------------------------------------------------------------------
    # B. Compute matched-set margins.
    # -------------------------------------------------------------------------
    all_margin_h = build_all_margins(
        pred_h,
        "Hiroshima_2018",
    )

    all_margin_k = build_all_margins(
        pred_k,
        "Kyushu_2017_Asakura_Toho",
    )

    delta_h = build_representation_delta(
        all_margin_h,
        "Hiroshima_2018",
    )

    delta_k = build_representation_delta(
        all_margin_k,
        "Kyushu_2017_Asakura_Toho",
    )

    delta_h.insert(
        0,
        "event",
        "Hiroshima_2018",
    )
    delta_k.insert(
        0,
        "event",
        "Kyushu_2017_Asakura_Toho",
    )

    all_delta = pd.concat(
        [
            delta_h,
            delta_k,
        ],
        ignore_index=True,
    )

    all_delta.to_csv(
        OUT / "127_ALL_BACKBONE_MATCHED_SET_MARGINS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # C. Reproduce published StrictPair before any spatial interpretation.
    # -------------------------------------------------------------------------
    audit_h = strictpair_reproduction_audit(
        delta_h,
        transfer_gap,
        "Hiroshima_2018",
    )

    audit_k = strictpair_reproduction_audit(
        delta_k,
        transfer_gap,
        "Kyushu_2017_Asakura_Toho",
    )

    strict_audit = pd.concat(
        [
            audit_h,
            audit_k,
        ],
        ignore_index=True,
    )

    strict_audit.to_csv(
        OUT / "127_STRICTPAIR_REPRODUCTION_AUDIT.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log("PASS_STRICTPAIR_REPRODUCTION_AUDIT=8/8_EVENT_BACKBONE_ROWS")

    # -------------------------------------------------------------------------
    # D. Cross-backbone consensus.
    # -------------------------------------------------------------------------
    consensus_h = build_consensus(
        delta_h,
        "Hiroshima_2018",
    )
    consensus_h.insert(
        0,
        "event",
        "Hiroshima_2018",
    )

    consensus_k = build_consensus(
        delta_k,
        "Kyushu_2017_Asakura_Toho",
    )
    consensus_k.insert(
        0,
        "event",
        "Kyushu_2017_Asakura_Toho",
    )

    consensus_all = pd.concat(
        [
            consensus_h,
            consensus_k,
        ],
        ignore_index=True,
    )

    consensus_all.to_csv(
        OUT / "127_CROSS_BACKBONE_SPATIAL_CONSENSUS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # E. Geometry.
    # -------------------------------------------------------------------------
    xgb_h_table = delta_h[
        delta_h["backbone"] == "XGBoost"
    ].copy()

    xgb_k_table = delta_k[
        delta_k["backbone"] == "XGBoost"
    ].copy()

    h_geom, h_geom_audit = discover_hiroshima_geometry(
        gpd,
        args.hiroshima_grid,
        args.hiroshima_layer,
        xgb_h_table["anchor_unit_id"].tolist(),
    )

    k_geom, k_geom_audit = discover_kyushu_geometry(
        gpd,
        args.kyushu_grid,
        args.kyushu_layer,
        xgb_k_table["anchor_unit_id"].tolist(),
    )

    jwrite(
        OUT / "127_GEOMETRY_AUDIT.json",
        {
            "Hiroshima_2018": h_geom_audit,
            "Kyushu_2017_Asakura_Toho": k_geom_audit,
        },
    )

    # -------------------------------------------------------------------------
    # F. Spatial joins.
    # -------------------------------------------------------------------------
    xgb_h_gdf = join_geometry(
        gpd,
        xgb_h_table,
        h_geom,
        "HIROSHIMA_XGB",
    )

    xgb_k_gdf = join_geometry(
        gpd,
        xgb_k_table,
        k_geom,
        "KYUSHU_XGB",
    )

    all_h_gdf = join_geometry(
        gpd,
        delta_h,
        h_geom,
        "HIROSHIMA_ALL_BACKBONE",
    )

    all_k_gdf = join_geometry(
        gpd,
        delta_k,
        k_geom,
        "KYUSHU_ALL_BACKBONE",
    )

    cons_h_gdf = join_geometry(
        gpd,
        consensus_h,
        h_geom,
        "HIROSHIMA_CONSENSUS",
    )

    cons_k_gdf = join_geometry(
        gpd,
        consensus_k,
        k_geom,
        "KYUSHU_CONSENSUS",
    )

    # -------------------------------------------------------------------------
    # G. CSV spatial summaries.
    # -------------------------------------------------------------------------
    xgb_summary_h = summarize_xgb(
        delta_h,
        "Hiroshima_2018",
    )

    xgb_summary_k = summarize_xgb(
        delta_k,
        "Kyushu_2017_Asakura_Toho",
    )

    consensus_summary_h = summarize_consensus(
        consensus_h,
        "Hiroshima_2018",
    )

    consensus_summary_k = summarize_consensus(
        consensus_k,
        "Kyushu_2017_Asakura_Toho",
    )

    pd.DataFrame(
        [
            {
                "event": s["event"],
                "n_sets": s["n_sets"],
                "raw_strictpair": s["raw_strictpair"],
                "msrr_strictpair": s["msrr_strictpair"],
                "strictpair_gain": s["strictpair_gain"],
                "delta_margin_mean": s["delta_margin"]["mean"],
                "delta_margin_median": s["delta_margin"]["median"],
                "delta_margin_q25": s["delta_margin"]["q25"],
                "delta_margin_q75": s["delta_margin"]["q75"],
                "delta_positive_fraction": s["delta_margin_positive_fraction"],
                "corrected_by_msrr": s["transition_counts"]["Corrected by MSRR"],
                "retained_success": s["transition_counts"]["Retained success"],
                "lost_by_msrr": s["transition_counts"]["Lost by MSRR"],
                "still_unresolved": s["transition_counts"]["Still unresolved"],
            }
            for s in [
                xgb_summary_h,
                xgb_summary_k,
            ]
        ]
    ).to_csv(
        OUT / "127_EVENT_XGB_SPATIAL_SUMMARY.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # H. Coordinate-quantile spatial robustness audit.
    # -------------------------------------------------------------------------
    block_h = build_spatial_block_audit(
        xgb_h_gdf,
        cons_h_gdf,
        "Hiroshima_2018",
        args.spatial_blocks,
    )

    block_k = build_spatial_block_audit(
        xgb_k_gdf,
        cons_k_gdf,
        "Kyushu_2017_Asakura_Toho",
        args.spatial_blocks,
    )

    block_all = pd.concat(
        [
            block_h,
            block_k,
        ],
        ignore_index=True,
    )

    block_all.to_csv(
        OUT / "127_SPATIAL_BLOCK_AUDIT.csv",
        index=False,
        encoding="utf-8-sig",
    )

    def spatial_block_summary(df: pd.DataFrame) -> Dict[str, Any]:
        valid = df[
            df["n_sets"] >= 30
        ].copy()

        return {
            "total_blocks": int(len(df)),
            "blocks_n_ge_30": int(len(valid)),
            "blocks_xgb_positive_median_delta": int(
                (
                    valid["xgb_delta_margin_median"] > 0
                ).sum()
            ),
            "fraction_blocks_xgb_positive_median_delta": (
                float(
                    (
                        valid["xgb_delta_margin_median"] > 0
                    ).mean()
                )
                if len(valid)
                else np.nan
            ),
            "blocks_xgb_positive_strictpair_gain": int(
                (
                    valid["xgb_strictpair_gain"] > 0
                ).sum()
            ),
            "fraction_blocks_xgb_positive_strictpair_gain": (
                float(
                    (
                        valid["xgb_strictpair_gain"] > 0
                    ).mean()
                )
                if len(valid)
                else np.nan
            ),
            "blocks_consensus_positive_median_delta": int(
                (
                    valid["consensus_delta_margin_median"] > 0
                ).sum()
            ),
            "fraction_blocks_consensus_positive_median_delta": (
                float(
                    (
                        valid["consensus_delta_margin_median"] > 0
                    ).mean()
                )
                if len(valid)
                else np.nan
            ),
        }

    block_summary_h = spatial_block_summary(
        block_h
    )
    block_summary_k = spatial_block_summary(
        block_k
    )

    # -------------------------------------------------------------------------
    # I. GeoPackage outputs.
    # -------------------------------------------------------------------------
    write_event_gpkg(
        xgb_h_gdf,
        all_h_gdf,
        cons_h_gdf,
        OUT / "127_HIROSHIMA_SPATIAL_RESULTS.gpkg",
    )

    write_event_gpkg(
        xgb_k_gdf,
        all_k_gdf,
        cons_k_gdf,
        OUT / "127_KYUSHU_SPATIAL_RESULTS.gpkg",
    )

    # -------------------------------------------------------------------------
    # J. Paper candidate figures.
    # -------------------------------------------------------------------------
    save_numeric_map(
        plt,
        xgb_h_gdf,
        "delta_margin",
        (
            "Hiroshima 2018: matched-set margin improvement "
            "(XGBoost, MSRR − RAW)"
        ),
        "127_FIG_HIROSHIMA_XGB_DELTA_MARGIN_MAP",
        "Delta matched-set margin",
    )

    save_numeric_map(
        plt,
        xgb_k_gdf,
        "delta_margin",
        (
            "Kyushu 2017: matched-set margin improvement "
            "(XGBoost, MSRR − RAW)"
        ),
        "127_FIG_KYUSHU_XGB_DELTA_MARGIN_MAP",
        "Delta matched-set margin",
    )

    save_categorical_map(
        plt,
        xgb_h_gdf,
        "strict_pair_transition",
        (
            "Hiroshima 2018: strict-pair transition after MSRR "
            "(XGBoost)"
        ),
        "127_FIG_HIROSHIMA_XGB_STRICTPAIR_TRANSITION_MAP",
    )

    save_categorical_map(
        plt,
        xgb_k_gdf,
        "strict_pair_transition",
        (
            "Kyushu 2017: strict-pair transition after MSRR "
            "(XGBoost)"
        ),
        "127_FIG_KYUSHU_XGB_STRICTPAIR_TRANSITION_MAP",
    )

    save_categorical_map(
        plt,
        cons_h_gdf,
        "positive_backbones",
        (
            "Hiroshima 2018: number of learner families with "
            "positive MSRR margin gain"
        ),
        "127_FIG_HIROSHIMA_CROSS_BACKBONE_CONSENSUS_MAP",
    )

    save_categorical_map(
        plt,
        cons_k_gdf,
        "positive_backbones",
        (
            "Kyushu 2017: number of learner families with "
            "positive MSRR margin gain"
        ),
        "127_FIG_KYUSHU_CROSS_BACKBONE_CONSENSUS_MAP",
    )

    save_transition_count_bar(
        plt,
        xgb_summary_h,
        "127_FIG_HIROSHIMA_XGB_TRANSITION_COUNTS",
    )

    save_transition_count_bar(
        plt,
        xgb_summary_k,
        "127_FIG_KYUSHU_XGB_TRANSITION_COUNTS",
    )

    # -------------------------------------------------------------------------
    # K. Formal summary + guardrails.
    # -------------------------------------------------------------------------
    summary = {
        "status": "PASS_127B_SPATIAL_MATCHED_MARGIN_ANALYSIS_COMPLETE",
        "analysis_scope": (
            "spatialization of frozen matched-set predictions only"
        ),
        "scientific_guardrails": {
            "arbitrary_grid_inference": False,
            "calibrated_probability_map": False,
            "model_training_or_selection": False,
            "spatial_anchor": (
                "inventory-supported positive grid of each matched set"
            ),
            "control_semantics": (
                "matched non-inventory control; not confirmed geological absence"
            ),
            "mapped_quantity": (
                "matched-set score margin and MSRR-minus-RAW margin improvement"
            ),
        },
        "input_files": {
            "internal_oof": {
                "path": str(internal_path),
                "sha256": sha256(internal_path),
            },
            "external_full_panel": {
                "path": str(external_path),
                "sha256": sha256(external_path),
            },
            "transfer_gap_reference": {
                "path": str(TRANSFER_GAP_EXACT),
                "sha256": sha256(TRANSFER_GAP_EXACT),
            },
        },
        "geometry": {
            "Hiroshima_2018": h_geom_audit,
            "Kyushu_2017_Asakura_Toho": k_geom_audit,
        },
        "strictpair_reproduction": {
            "rows": int(len(strict_audit)),
            "all_pass_1e-6": bool(
                strict_audit["pass_tolerance_1e-6"].all()
            ),
            "max_abs_error": float(
                max(
                    strict_audit["raw_abs_error"].max(),
                    strict_audit["msrr_abs_error"].max(),
                )
            ),
        },
        "xgboost": {
            "Hiroshima_2018": xgb_summary_h,
            "Kyushu_2017_Asakura_Toho": xgb_summary_k,
        },
        "cross_backbone_consensus": {
            "Hiroshima_2018": consensus_summary_h,
            "Kyushu_2017_Asakura_Toho": consensus_summary_k,
        },
        "spatial_block_audit": {
            "block_definition": (
                f"{args.spatial_blocks}x{args.spatial_blocks} "
                "coordinate-quantile blocks based only on matched-set anchor locations"
            ),
            "minimum_n_for_summary": 30,
            "Hiroshima_2018": block_summary_h,
            "Kyushu_2017_Asakura_Toho": block_summary_k,
        },
        "recommended_paper_interpretation": (
            "Use these results to assess whether the MSRR matched-set margin gain "
            "is spatially widespread or concentrated. Do not describe the maps as "
            "landslide occurrence probability or wall-to-wall susceptibility maps."
        ),
        "next_step": (
            "Interpret 127, select final paper maps, then freeze all core experiments."
        ),
    }

    jwrite(
        OUT / "127_SPATIAL_SUMMARY.json",
        summary,
    )

    # Compact machine-readable event summary.
    paper_rows = []

    for event, xs, cs, bs in [
        (
            "Hiroshima_2018",
            xgb_summary_h,
            consensus_summary_h,
            block_summary_h,
        ),
        (
            "Kyushu_2017_Asakura_Toho",
            xgb_summary_k,
            consensus_summary_k,
            block_summary_k,
        ),
    ]:
        paper_rows.append(
            {
                "event": event,
                "n_sets": xs["n_sets"],
                "xgb_raw_strictpair": xs["raw_strictpair"],
                "xgb_msrr_strictpair": xs["msrr_strictpair"],
                "xgb_strictpair_gain": xs["strictpair_gain"],
                "xgb_delta_margin_median": xs["delta_margin"]["median"],
                "xgb_delta_positive_fraction": xs["delta_margin_positive_fraction"],
                "xgb_corrected_fraction": xs["transition_fractions"]["Corrected by MSRR"],
                "xgb_lost_fraction": xs["transition_fractions"]["Lost by MSRR"],
                "consensus_all4_positive_fraction": cs["fraction_all4_backbones_delta_positive"],
                "consensus_at_least3_positive_fraction": cs["fraction_at_least3_backbones_delta_positive"],
                "spatial_blocks_n_ge_30": bs["blocks_n_ge_30"],
                "spatial_blocks_xgb_positive_median_delta_fraction": bs[
                    "fraction_blocks_xgb_positive_median_delta"
                ],
                "spatial_blocks_xgb_positive_strictpair_gain_fraction": bs[
                    "fraction_blocks_xgb_positive_strictpair_gain"
                ],
                "spatial_blocks_consensus_positive_median_delta_fraction": bs[
                    "fraction_blocks_consensus_positive_median_delta"
                ],
            }
        )

    pd.DataFrame(
        paper_rows
    ).to_csv(
        OUT / "127_PAPER_SPATIAL_RESULT_SUMMARY.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # L. Console report.
    # -------------------------------------------------------------------------
    log("")
    log("=" * 120)
    log("PASS_127B_SPATIAL_MATCHED_MARGIN_ANALYSIS_COMPLETE")
    log(
        "STRICTPAIR_REPRODUCTION_ALL_PASS="
        f"{bool(strict_audit['pass_tolerance_1e-6'].all())}"
    )

    for event, xs, cs, bs in [
        (
            "HIROSHIMA",
            xgb_summary_h,
            consensus_summary_h,
            block_summary_h,
        ),
        (
            "KYUSHU",
            xgb_summary_k,
            consensus_summary_k,
            block_summary_k,
        ),
    ]:
        log(
            f"{event}_XGB_RAW_STRICTPAIR="
            f"{xs['raw_strictpair']:.6f}"
        )
        log(
            f"{event}_XGB_MSRR_STRICTPAIR="
            f"{xs['msrr_strictpair']:.6f}"
        )
        log(
            f"{event}_XGB_STRICTPAIR_GAIN="
            f"{xs['strictpair_gain']:+.6f}"
        )
        log(
            f"{event}_XGB_DELTA_MARGIN_MEDIAN="
            f"{xs['delta_margin']['median']:+.6f}"
        )
        log(
            f"{event}_XGB_DELTA_POSITIVE_FRACTION="
            f"{xs['delta_margin_positive_fraction']:.6f}"
        )
        log(
            f"{event}_XGB_CORRECTED_BY_MSRR="
            f"{xs['transition_counts']['Corrected by MSRR']}"
        )
        log(
            f"{event}_XGB_LOST_BY_MSRR="
            f"{xs['transition_counts']['Lost by MSRR']}"
        )
        log(
            f"{event}_CONSENSUS_ALL4_POSITIVE_FRACTION="
            f"{cs['fraction_all4_backbones_delta_positive']:.6f}"
        )
        log(
            f"{event}_CONSENSUS_AT_LEAST3_POSITIVE_FRACTION="
            f"{cs['fraction_at_least3_backbones_delta_positive']:.6f}"
        )
        log(
            f"{event}_SPATIAL_BLOCKS_XGB_POSITIVE_MEDIAN_DELTA="
            f"{bs['blocks_xgb_positive_median_delta']}/"
            f"{bs['blocks_n_ge_30']}"
        )
        log(
            f"{event}_SPATIAL_BLOCKS_CONSENSUS_POSITIVE_MEDIAN_DELTA="
            f"{bs['blocks_consensus_positive_median_delta']}/"
            f"{bs['blocks_n_ge_30']}"
        )

    log(f"OUTPUT={OUT}")
    log(
        "NEXT_STEP=SEND_127_SPATIAL_SUMMARY_JSON_AND_FINAL_SCREEN"
    )
    log(
        "IF_PASS_THEN_ALL_CORE_EXPERIMENTS_CAN_BE_FROZEN"
    )
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
            "status": "FAIL_127B_SPATIAL_MATCHED_MARGIN_ANALYSIS",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        jwrite(
            OUT / "127_FAILURE.json",
            err,
        )
        print(
            traceback.format_exc(),
            flush=True,
        )
        raise SystemExit(1)
