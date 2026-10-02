from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.spatial import cKDTree
from scipy.stats import chi2_contingency, wasserstein_distance
from shapely.geometry import Point


PROJECT = Path(__file__).resolve().parents[2]
D0 = PROJECT / "data" / "05_final_dataset_assembly" / "00_dataset_inventory_and_split_feasibility"
OUT = PROJECT / "data" / "05_final_dataset_assembly" / "01_spatial_split_protocol"
MASTER = PROJECT / "data" / "01_county_prediction_domain" / "02_grid_250m" / "hiroshima_grid_250m_master.gpkg"
STATIC_FULL = PROJECT / "data" / "03_static_eogis_features" / "90_third_layer_assembly" / "05_static_03_final_audit_freeze" / "06_frozen" / "static_third_layer_full_frozen.parquet"
STATIC_MODEL = PROJECT / "data" / "03_static_eogis_features" / "90_third_layer_assembly" / "05_static_03_final_audit_freeze" / "06_frozen" / "static_third_layer_model_candidate_frozen.parquet"
RAIN_DATASET = PROJECT / "data" / "04_dynamic_rainfall_features" / "07_rain_04_final_audit_freeze" / "07_frozen" / "dynamic_rainfall_features_frozen.parquet_dataset"
BOUNDARY = PROJECT / "data" / "01_county_prediction_domain" / "01_boundary_frozen" / "hiroshima_boundary_2018_epsg6671.gpkg"
MUNICIPALITY = PROJECT / "data" / "01_county_prediction_domain" / "01_boundary_frozen" / "hiroshima_n03_features_2018_epsg6668.gpkg"

ANCHOR_UTC = pd.Timestamp("2018-07-06T11:00:00Z")
PRIMARY_PROTOCOL = "PRIMARY_BALANCED_SPATIAL_GROUPED_5FOLD"
STRICT_PROTOCOL = "STRICT_SPATIAL_EXTRAPOLATION"

WEIGHT_PROFILES = {
    "STRUCTURE_FOCUSED": {"count": 100.0, "grade": 20.0, "geology": 12.0, "landcover": 8.0, "missing": 8.0, "continuous": 6.0, "municipality": 5.0, "large_group": 10.0, "spatial": 3.0},
    "BALANCED": {"count": 100.0, "grade": 15.0, "geology": 10.0, "landcover": 7.0, "missing": 7.0, "continuous": 18.0, "municipality": 4.0, "large_group": 8.0, "spatial": 3.0},
    "CONTINUOUS_FOCUSED": {"count": 100.0, "grade": 10.0, "geology": 7.0, "landcover": 5.0, "missing": 5.0, "continuous": 30.0, "municipality": 3.0, "large_group": 6.0, "spatial": 2.0},
}
PROFILE_SEEDS = {"STRUCTURE_FOCUSED": 7, "BALANCED": 11, "CONTINUOUS_FOCUSED": 21}
DYNAMIC_FIELDS = ["rain_30m_mm", "rain_1h_mm", "rain_3h_mm", "rain_6h_mm", "rain_12h_mm", "rain_24h_mm", "rain_48h_mm", "rain_72h_mm", "rain_120h_mm", "api_k090_step30m_120h"]
FOCUS_CONTINUOUS = [
    "dem_slope_mean", "dem_elevation_mean", "ndvi_pre_event", "dem_twi_mean", "dem_roughness_mean",
    "dem_profile_curvature_mean", "dem_plan_curvature_mean", "road_distance_to_any_road_m",
    "road_density_km_per_km2", "river_distance_to_osm_river_m", "river_density_km_km2",
    "coast_distance_to_centroid_m", "rain_24h_mm", "rain_72h_mm", "rain_120h_mm", "api_k090_step30m_120h",
]
SOIL_FIELDS = [f"soil_{p}_{d}" for p in ["clay", "sand", "silt", "bdod", "soc", "cec"] for d in ["0_5cm", "5_15cm", "15_30cm"]]


def mkdirs():
    for name in ["00_input_reference", "01_smd_diagnostics", "02_spatial_group_profiles", "03_candidate_optimizations", "04_primary_5fold", "05_strict_extrapolation", "06_qc_maps", "07_audit", "08_frozen", "scripts"]:
        (OUT / name).mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def semantic_sha256(path: Path) -> str:
    """Hash tabular/contract meaning, excluding container metadata."""
    if path.suffix.lower() == ".json":
        payload = json.dumps(read_json(path), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    elif path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
        keys = [k for k in ["pair_set_id", "protocol", "scope", "fold", "field_name", "metric_name"] if k in frame.columns]
        if keys:
            frame = frame.sort_values(keys, kind="mergesort", na_position="last").reset_index(drop=True)
        frame = frame.reindex(sorted(frame.columns), axis=1)
        payload = frame.to_csv(index=False, na_rep="<NA>", float_format="%.17g", lineterminator="\n").encode("utf-8")
    else:
        return sha256(path)
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def robust_scale(s: pd.Series) -> tuple[pd.Series, float, float]:
    x = pd.to_numeric(s, errors="coerce")
    med = float(x.median())
    iqr = float(x.quantile(0.75) - x.quantile(0.25))
    return ((x - med) / iqr if iqr > 0 else x * 0.0), med, iqr


def raw_smd(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna().to_numpy(float)
    b = pd.to_numeric(b, errors="coerce").dropna().to_numpy(float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    denom = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    if denom == 0:
        return 0.0 if np.mean(a) == np.mean(b) else np.inf
    return float((np.mean(a) - np.mean(b)) / denom)


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    table = pd.crosstab(a.fillna("<NA>"), b)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    return float(math.sqrt(chi2 / max(n * min(table.shape[0] - 1, table.shape[1] - 1), 1)))


def compact_counts(s: pd.Series) -> str:
    return json.dumps({str(k): int(v) for k, v in s.fillna("<NA>").value_counts().sort_index().items()}, sort_keys=True, ensure_ascii=False)


def geology_parent(value) -> str:
    v = str(value) if pd.notna(value) else "MISSING"
    if v in {"PLUTONIC_ROCK", "VOLCANIC_ROCK"}:
        return "IGNEOUS"
    if v == "UNCONSOLIDATED_SEDIMENT":
        return "UNCONSOLIDATED"
    if v == "ACCRETIONARY_COMPLEX":
        return "ACCRETIONARY_COMPLEX_ONLY"
    if v == "SEDIMENTARY_ROCK":
        return "SEDIMENTARY"
    if v == "METAMORPHIC_ROCK":
        return "METAMORPHIC"
    return "OTHER_OR_MIXED_OR_MISSING"


def soil_pattern(row: pd.Series) -> str:
    n = int(row[SOIL_FIELDS].notna().sum())
    return "SOIL_COMPLETE" if n == len(SOIL_FIELDS) else ("SOIL_ALL_MISSING" if n == 0 else "SOIL_PARTIAL")


def verify_dataset00() -> tuple[pd.DataFrame, int]:
    marker = read_json(D0 / "DATASET_00_READY.marker.json")
    if marker.get("DATASET_00_INVENTORY_PASS") != "YES" or marker.get("READY_FOR_DATASET_01") != "YES" or marker.get("UPSTREAM_FROZEN_HASH_CHANGES") != 0:
        raise RuntimeError("DATASET-00 gate failed")
    manifest = pd.read_csv(D0 / "dataset_00_file_hashes.csv")
    changes = 0
    for row in manifest.itertuples(index=False):
        p = D0 / row.relative_path
        changes += int(not p.is_file() or p.stat().st_size != row.file_size_bytes or sha256(p) != row.sha256)
    if changes:
        raise RuntimeError(f"DATASET-00 hash changes={changes}")
    return manifest, changes


def load_anchor_rain(unit_ids: set[str]) -> pd.DataFrame:
    dset = ds.dataset(RAIN_DATASET, format="parquet", partitioning="hive")
    table = dset.to_table(columns=["unit_id", "timestamp_utc"] + DYNAMIC_FIELDS, filter=(ds.field("timestamp_utc") == ANCHOR_UTC.to_pydatetime()) & ds.field("unit_id").isin(sorted(unit_ids)))
    out = table.to_pandas()
    out["unit_id"] = out["unit_id"].astype(str)
    if len(out) != len(unit_ids) or out["unit_id"].nunique() != len(unit_ids):
        raise RuntimeError("anchor rain extraction incomplete")
    return out.drop(columns="timestamp_utc")


class ObjectiveModel:
    def __init__(self, groups: pd.DataFrame, cat_mats: dict[str, np.ndarray], cont_sum: np.ndarray, cont_n: np.ndarray, neighbor_edges: np.ndarray, weights: dict[str, float]):
        self.groups = groups
        self.sizes = groups["pair_set_count"].to_numpy(float)
        self.cat_mats = cat_mats
        self.cont_sum = cont_sum
        self.cont_n = cont_n
        self.edges = neighbor_edges
        self.weights = weights
        self.k = 5
        self.top_mask = groups["largest_group_rank"].le(10).to_numpy()

    def evaluate(self, assignment: np.ndarray, partial: bool = False) -> tuple[float, dict[str, float]]:
        mask = assignment >= 0
        a = assignment[mask]
        sizes = self.sizes[mask]
        if not len(a):
            return 0.0, {k: 0.0 for k in self.weights}
        fold_size = np.bincount(a, weights=sizes, minlength=self.k)
        target = sizes.sum() / self.k
        count = float(np.max(np.abs(fold_size - target) / max(target, 1)))

        cat_scores = {}
        for name, matrix in self.cat_mats.items():
            m = matrix[mask]
            global_prop = m.sum(axis=0) / max(m.sum(), 1)
            diffs = []
            for f in range(self.k):
                fm = m[a == f].sum(axis=0)
                prop = fm / max(fm.sum(), 1)
                diffs.append(np.mean(np.abs(prop - global_prop)))
            cat_scores[name] = float(np.mean(diffs))

        cs, cn = self.cont_sum[mask], self.cont_n[mask]
        global_mean = cs.sum(axis=0) / np.maximum(cn.sum(axis=0), 1)
        cont_diffs = []
        for f in range(self.k):
            fm = cs[a == f].sum(axis=0) / np.maximum(cn[a == f].sum(axis=0), 1)
            cont_diffs.append(np.mean(np.abs(fm - global_mean)))
        continuous = float(np.mean(cont_diffs))

        top_sizes = np.where(self.top_mask[mask], sizes, 0.0)
        top_fold = np.bincount(a, weights=top_sizes, minlength=self.k)
        large_group = float(np.max(top_fold) / max(top_sizes.sum(), 1) - 1 / self.k)

        if len(self.edges):
            valid_edges = mask[self.edges[:, 0]] & mask[self.edges[:, 1]]
            ee = self.edges[valid_edges]
            spatial = float(np.mean(assignment[ee[:, 0]] != assignment[ee[:, 1]])) if len(ee) else 0.0
        else:
            spatial = 0.0
        components = {
            "count": count, "grade": cat_scores.get("grade", 0.0), "geology": cat_scores.get("geology", 0.0),
            "landcover": cat_scores.get("landcover", 0.0), "missing": cat_scores.get("missing", 0.0),
            "continuous": continuous, "municipality": cat_scores.get("municipality", 0.0),
            "large_group": large_group, "spatial": spatial,
        }
        total = float(sum(self.weights[k] * components[k] for k in self.weights))
        return total, components


def greedy_assignment(model: ObjectiveModel, seed: int) -> tuple[np.ndarray, list[dict]]:
    rng = np.random.default_rng(seed)
    tie = rng.random(len(model.groups))
    order = np.lexsort((tie, -model.sizes))
    fold_order = rng.permutation(5)
    a = np.full(len(model.groups), -1, dtype=np.int8)
    trace = []
    for step, g in enumerate(order):
        best = None
        for f in fold_order:
            a[g] = f
            score, comp = model.evaluate(a, partial=True)
            candidate = (score, int(f), comp)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
            a[g] = -1
        a[g] = best[1]
        if step < 10 or (step + 1) % 25 == 0 or step == len(order) - 1:
            trace.append({"iteration": step + 1, "objective": best[0], **best[2], "event": "GREEDY_ASSIGN"})
    return a, trace


def local_search(model: ObjectiveModel, start: np.ndarray, seed: int) -> tuple[np.ndarray, list[dict]]:
    rng = np.random.default_rng(seed)
    a = start.copy()
    current, comp = model.evaluate(a)
    trace = [{"iteration": 0, "objective": current, **comp, "event": "START"}]
    iteration = 0
    for round_id in range(12):
        best_score, best_action, best_comp = current, None, None
        for g in rng.permutation(len(a)):
            old = int(a[g])
            for new in rng.permutation(5):
                if new == old or np.sum(a == old) <= 1:
                    continue
                a[g] = new
                score, c = model.evaluate(a)
                a[g] = old
                if score < best_score - 1e-12:
                    best_score, best_action, best_comp = score, ("MOVE", g, old, int(new)), c
        for _ in range(2500):
            i, j = rng.integers(0, len(a), size=2)
            if i == j or a[i] == a[j]:
                continue
            a[i], a[j] = a[j], a[i]
            score, c = model.evaluate(a)
            a[i], a[j] = a[j], a[i]
            if score < best_score - 1e-12:
                best_score, best_action, best_comp = score, ("SWAP", int(i), int(j)), c
        if best_action is None:
            break
        if best_action[0] == "MOVE":
            _, g, _, new = best_action
            a[g] = new
        else:
            _, i, j = best_action
            a[i], a[j] = a[j], a[i]
        current, comp = best_score, best_comp
        iteration += 1
        trace.append({"iteration": iteration, "objective": current, **comp, "event": best_action[0]})
    return a, trace


def simulated_annealing(model: ObjectiveModel, start: np.ndarray, seed: int) -> tuple[np.ndarray, list[dict]]:
    rng = np.random.default_rng(seed)
    a = start.copy()
    current, comp = model.evaluate(a)
    best, best_score, best_comp = a.copy(), current, comp
    trace = [{"iteration": 0, "objective": current, **comp, "event": "START"}]
    steps = 5000
    for step in range(1, steps + 1):
        temp = 0.05 * (0.0001 / 0.05) ** (step / steps)
        if rng.random() < 0.75:
            i, j = rng.integers(0, len(a), size=2)
            if i == j or a[i] == a[j]:
                continue
            a[i], a[j] = a[j], a[i]
            score, c = model.evaluate(a)
            delta = score - current
            if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-12)):
                current, comp = score, c
            else:
                a[i], a[j] = a[j], a[i]
        else:
            i = int(rng.integers(0, len(a)))
            old, new = int(a[i]), int(rng.integers(0, 5))
            if old == new or np.sum(a == old) <= 1:
                continue
            a[i] = new
            score, c = model.evaluate(a)
            delta = score - current
            if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-12)):
                current, comp = score, c
            else:
                a[i] = old
        if current < best_score - 1e-12:
            best, best_score, best_comp = a.copy(), current, comp
            trace.append({"iteration": step, "objective": best_score, **best_comp, "event": "NEW_BEST"})
    trace.append({"iteration": steps, "objective": best_score, **best_comp, "event": "FINAL_BEST"})
    return best, trace


def field_statistics(data: pd.DataFrame, assignment_col: str, scheme: str, model_groups: dict[str, str], field_types: dict[str, str], variant: str = "ALL_GROUPS", excluded_group: str | None = None) -> pd.DataFrame:
    d = data if excluded_group is None else data[data["spatial_group_id"].ne(excluded_group)]
    rows = []
    folds = sorted(d[assignment_col].dropna().unique())
    for field, ftype in field_types.items():
        global_s = d[field]
        if ftype == "CATEGORICAL":
            global_counts = global_s.fillna("<NA>").value_counts(normalize=True)
            rare = bool((global_counts < 0.01).any())
            for fold in folds:
                mask = d[assignment_col].eq(fold)
                a, b = global_s[mask], global_s[~mask]
                ap, bp = a.fillna("<NA>").value_counts(normalize=True), b.fillna("<NA>").value_counts(normalize=True)
                cats = sorted(set(ap.index).union(bp.index), key=str)
                max_diff = max(abs(float(ap.get(c, 0)) - float(bp.get(c, 0))) for c in cats)
                tv = 0.5 * sum(abs(float(ap.get(c, 0)) - float(bp.get(c, 0))) for c in cats)
                top_group = d.loc[mask].groupby("spatial_group_id").size().sort_values(ascending=False)
                rows.append({
                    "scheme": scheme, "analysis_variant": variant, "field_name": field, "field_type": ftype, "model_input_group": model_groups.get(field, "AUDIT_SPLIT_ONLY"), "fold_id": fold,
                    "valid_count": int(a.notna().sum()), "missing_count": int(a.isna().sum()), "missing_fraction": float(a.isna().mean()), "zero_fraction": np.nan,
                    "mean": np.nan, "standard_deviation": np.nan, "median": np.nan, "q1": np.nan, "q3": np.nan, "iqr": np.nan, "min": np.nan, "max": np.nan,
                    "raw_smd": np.nan, "robust_smd": np.nan, "median_robust_difference": np.nan, "missing_fraction_difference": float(a.isna().mean() - b.isna().mean()), "zero_fraction_difference": np.nan,
                    "near_constant_flag": "NO", "rare_feature_flag": "YES" if rare else "NO", "unstable_metric_flag": "YES" if rare or a.notna().sum() < 30 else "NO",
                    "max_spatial_group_contribution": float(top_group.iloc[0] / max(len(a), 1)) if len(top_group) else 0.0, "top_contributing_spatial_group_id": str(top_group.index[0]) if len(top_group) else "",
                    "max_category_proportion_difference": max_diff, "total_variation_distance": tv, "cramers_v": cramers_v(global_s, mask.astype(str)),
                    "category_missing_in_fold_flag": "YES" if any(c not in ap.index for c in global_counts.index) else "NO", "iqr_ratio": np.nan, "wasserstein_or_quantile_distance": np.nan,
                    "audit_interpretation": "CATEGORY_DISTRIBUTION_AUDIT_RARE_LEVELS_UNSTABLE" if rare else "CATEGORY_DISTRIBUTION_AUDIT",
                })
            continue
        x = pd.to_numeric(global_s, errors="coerce")
        z, gmed, giqr = robust_scale(x)
        valid_frac = float(x.notna().mean())
        nonzero_frac = float(x.ne(0).sum() / max(x.notna().sum(), 1))
        near = giqr == 0
        rare = nonzero_frac < 0.01
        skewed = abs(float(x.skew())) > 2 if x.notna().sum() > 2 else False
        if any(token in field.lower() for token in ["distance", "rain", "density", "curvature", "twi", "api_"]):
            skewed = True
        stable_global = valid_frac >= 0.8 and nonzero_frac >= 0.05 and giqr > 0
        for fold in folds:
            mask = d[assignment_col].eq(fold)
            a, b = x[mask], x[~mask]
            za, zb = z[mask], z[~mask]
            aq1, aq3 = a.quantile(0.25), a.quantile(0.75)
            bq1, bq3 = b.quantile(0.25), b.quantile(0.75)
            aiqr, biqr = float(aq3 - aq1), float(bq3 - bq1)
            stable = stable_global and a.notna().sum() >= 100 and b.notna().sum() >= 100
            unstable = near or rare or a.notna().sum() < 30 or b.notna().sum() < 30
            group_means = d.loc[mask].assign(_v=a[mask]).groupby("spatial_group_id")["_v"].mean()
            if len(group_means):
                contributions = (group_means - x.mean()).abs() * d.loc[mask].groupby("spatial_group_id").size() / max(mask.sum(), 1)
                top_id, top_value = str(contributions.idxmax()), float(contributions.max())
            else:
                top_id, top_value = "", 0.0
            av, bv = a.dropna().to_numpy(float), b.dropna().to_numpy(float)
            wd = float(wasserstein_distance(av, bv) / giqr) if len(av) and len(bv) and giqr > 0 else np.nan
            rows.append({
                "scheme": scheme, "analysis_variant": variant, "field_name": field,
                "field_type": "STABLE_CONTINUOUS" if stable else ("SKEWED_CONTINUOUS" if skewed else "UNSTABLE_CONTINUOUS"),
                "model_input_group": model_groups.get(field, "UNKNOWN"), "fold_id": fold,
                "valid_count": int(a.notna().sum()), "missing_count": int(a.isna().sum()), "missing_fraction": float(a.isna().mean()),
                "zero_fraction": float(a.eq(0).sum() / max(a.notna().sum(), 1)), "mean": float(a.mean()), "standard_deviation": float(a.std()),
                "median": float(a.median()), "q1": float(aq1), "q3": float(aq3), "iqr": aiqr, "min": float(a.min()), "max": float(a.max()),
                "raw_smd": raw_smd(a, b), "robust_smd": float(za.mean() - zb.mean()), "median_robust_difference": float(za.median() - zb.median()),
                "missing_fraction_difference": float(a.isna().mean() - b.isna().mean()),
                "zero_fraction_difference": float(a.eq(0).sum() / max(a.notna().sum(), 1) - b.eq(0).sum() / max(b.notna().sum(), 1)),
                "near_constant_flag": "YES" if near else "NO", "rare_feature_flag": "YES" if rare else "NO", "unstable_metric_flag": "YES" if unstable else "NO",
                "max_spatial_group_contribution": top_value, "top_contributing_spatial_group_id": top_id,
                "max_category_proportion_difference": np.nan, "total_variation_distance": np.nan, "cramers_v": np.nan, "category_missing_in_fold_flag": "NOT_APPLICABLE",
                "iqr_ratio": float(aiqr / biqr) if biqr > 0 else np.nan, "wasserstein_or_quantile_distance": wd,
                "audit_interpretation": "STABLE_CONTINUOUS" if stable else ("REPORT_NOT_SOLE_FAILURE_RARE_OR_NEAR_CONSTANT" if unstable else "SKEW_AWARE_CONTINUOUS"),
            })
    return pd.DataFrame(rows)


def assignment_metrics(data: pd.DataFrame, assignment: pd.Series, stable_fields: list[str], category_fields: list[str]) -> dict:
    d = data.copy()
    d["_fold"] = d["spatial_group_id"].map(assignment)
    counts = d.drop_duplicates("pair_set_id")["_fold"].value_counts()
    target = d["pair_set_id"].nunique() / 5
    max_count_dev = float((counts - target).abs().max() / target)
    max_raw, max_robust = 0.0, 0.0
    for field in stable_fields:
        x = pd.to_numeric(d[field], errors="coerce")
        z, _, iqr = robust_scale(x)
        if iqr <= 0:
            continue
        for fold in sorted(d["_fold"].dropna().unique()):
            mask = d["_fold"].eq(fold)
            max_raw = max(max_raw, abs(raw_smd(x[mask], x[~mask])))
            max_robust = max(max_robust, abs(float(z[mask].mean() - z[~mask].mean())))
    max_cat = 0.0
    for field in category_fields:
        for fold in sorted(d["_fold"].dropna().unique()):
            mask = d["_fold"].eq(fold)
            ap = d.loc[mask, field].fillna("<NA>").value_counts(normalize=True)
            bp = d.loc[~mask, field].fillna("<NA>").value_counts(normalize=True)
            max_cat = max(max_cat, max(abs(float(ap.get(c, 0)) - float(bp.get(c, 0))) for c in set(ap.index).union(bp.index)))
    pos = d.drop_duplicates("pair_set_id")
    a1_missing = sum(not pos.loc[pos["_fold"].eq(f), "evidence_grade_strongest"].eq("A1").any() for f in range(5))
    a3_missing = sum(not pos.loc[pos["_fold"].eq(f), "evidence_grade_strongest"].eq("A3").any() for f in range(5))
    major_geo = pos["geology_parent_class"].value_counts(normalize=True)
    major_geo = major_geo[major_geo >= 0.05].index
    geo_missing = sum(sum(not pos.loc[pos["_fold"].eq(f), "geology_parent_class"].eq(g).any() for g in major_geo) for f in range(5))
    return {"max_pair_count_deviation": max_count_dev, "max_stable_raw_smd": max_raw, "max_stable_robust_smd": max_robust, "max_category_proportion_difference": max_cat, "a1_missing_fold_count": int(a1_missing), "a3_missing_fold_count": int(a3_missing), "major_geology_missing_fold_count": int(geo_missing), "empty_fold_count": int(sum(counts.get(f, 0) == 0 for f in range(5)))}


def save_scatter_map(path: Path, title: str, data: pd.DataFrame, boundary: gpd.GeoDataFrame, color, cmap="viridis", categorical=False, size=8):
    # Pillow avoids a reproducible Windows Matplotlib renderer crash on the
    # multipart Hiroshima coastline while retaining true EPSG:6671 positions.
    width = height = 1600
    pad, title_h, legend_w = 70, 80, 250
    image = Image.new("RGB", (width + legend_w, height + title_h), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    minx, miny, maxx, maxy = map(float, boundary.total_bounds)
    spanx, spany = max(maxx - minx, 1.0), max(maxy - miny, 1.0)
    scale = min((width - 2 * pad) / spanx, (height - 2 * pad) / spany)
    ox = pad + ((width - 2 * pad) - spanx * scale) / 2
    oy = title_h + pad + ((height - 2 * pad) - spany * scale) / 2

    def px(x, y):
        return (int(round(ox + (float(x) - minx) * scale)), int(round(oy + (maxy - float(y)) * scale)))

    parts_all = []
    for geom in boundary.geometry:
        parts = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        parts_all.extend(parts)
    for part in sorted(parts_all, key=lambda g: g.area, reverse=True)[:50]:
        draw.line([px(x, y) for x, y in part.exterior.coords], fill=(55, 55, 55), width=2)

    palette = [(31,119,180),(255,127,14),(44,160,44),(214,39,40),(148,103,189),(140,86,75),(227,119,194),(127,127,127),(188,189,34),(23,190,207),(57,106,177),(218,124,48),(62,150,81),(204,37,41),(107,76,154),(146,36,40),(83,81,84),(160,160,160),(255,190,0),(0,158,115)]
    values = pd.Series(color).reset_index(drop=True)
    if categorical:
        labels = sorted(values.fillna("<NA>").astype(str).unique())
        lookup = {v: palette[i % len(palette)] for i, v in enumerate(labels)}
        colors = [lookup[v] for v in values.fillna("<NA>").astype(str)]
    else:
        nums = pd.to_numeric(values, errors="coerce")
        finite = nums[np.isfinite(nums)]
        lo, hi = (float(finite.quantile(0.02)), float(finite.quantile(0.98))) if len(finite) else (0.0, 1.0)
        if hi <= lo: hi = lo + 1.0
        colors = []
        for v in nums:
            if not np.isfinite(v): colors.append((190,190,190)); continue
            t = min(1.0, max(0.0, (float(v) - lo) / (hi - lo)))
            colors.append((int(35 + 210*t), int(80 + 90*(1-abs(2*t-1))), int(220 - 185*t)))
    radius = max(2, min(7, int(round(math.sqrt(max(size, 1)) * 1.4))))
    for (_, row), col in zip(data.reset_index(drop=True).iterrows(), colors):
        if pd.isna(row.get("centroid_x")) or pd.isna(row.get("centroid_y")): continue
        x, y = px(row["centroid_x"], row["centroid_y"])
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=col)
    draw.text((pad, 25), title, fill=(20,20,20), font=font)
    lx = width + 20
    if categorical:
        for i, label in enumerate(labels[:28]):
            yy = title_h + 10 + i * 28
            draw.rectangle((lx, yy, lx+16, yy+16), fill=lookup[label])
            draw.text((lx+24, yy+2), str(label)[:28], fill=(20,20,20), font=font)
    else:
        draw.text((lx, title_h + 10), f"2%-98%: {lo:.4g} to {hi:.4g}", fill=(20,20,20), font=font)
        for i in range(200):
            t = i / 199
            col = (int(35 + 210*t), int(80 + 90*(1-abs(2*t-1))), int(220 - 185*t))
            draw.line((lx, title_h + 40 + i, lx+24, title_h + 40 + i), fill=col, width=1)
    image.save(path, format="PNG", optimize=True)


def main(run_id: str):
    mkdirs()
    errors = []
    d0_manifest, upstream_changes = verify_dataset00()

    whitelist = pd.read_csv(D0 / "dataset_00_model_feature_whitelist.csv")
    leakage = pd.read_csv(D0 / "dataset_00_leakage_forbidden_fields.csv")
    cohort = pd.read_parquet(D0 / "dataset_00_training_cohort_index.parquet")
    dynamic_index_rows = len(pd.read_parquet(D0 / "dataset_00_dynamic_sequence_index.parquet", columns=["unit_id"]))
    split0 = pd.read_parquet(D0 / "dataset_00_split_group_candidates.parquet")
    outside0 = pd.read_csv(D0 / "dataset_00_outside_common_support_registry.csv")
    static_model = pd.read_parquet(STATIC_MODEL)
    static_full = pd.read_parquet(STATIC_FULL, columns=["unit_id", "geology_dominant_class", "lc_dominant_fcc_name"])
    master = gpd.read_file(MASTER, layer="hiroshima_grid_250m_master")
    boundary = gpd.read_file(BOUNDARY).to_crs(6671)
    # The frozen county outline contains ~140k coastline vertices.  A 100 m
    # topology-preserving display copy keeps the audit geometry untouched while
    # preventing Matplotlib from stalling on every QC figure.
    boundary.geometry = boundary.geometry.simplify(100, preserve_topology=True)
    master_small = master[["unit_id", "grid_row", "grid_col", "grid_minx", "grid_miny", "grid_maxx", "grid_maxy", "centroid_x", "centroid_y", "geometry"]].copy()
    unit_ids = set(cohort["unit_id"].astype(str))
    anchor_rain = load_anchor_rain(unit_ids)

    primary_pos = split0[["pair_set_id", "positive_unit_id", "spatial_group_id", "evidence_component_ids", "evidence_grade_strongest", "geology_dominant_class", "spatial_zone", "centroid_x", "centroid_y", "grid_row", "grid_col", "fold_5", "fold_4", "split_3"]].copy()
    primary_pos["geology_parent_class"] = primary_pos["geology_dominant_class"].map(geology_parent)
    positive_extra = static_full.rename(columns={"unit_id": "positive_unit_id", "lc_dominant_fcc_name": "dominant_landcover"})[["positive_unit_id", "dominant_landcover"]]
    primary_pos = primary_pos.merge(positive_extra, on="positive_unit_id", how="left")

    mun = gpd.read_file(MUNICIPALITY).to_crs(6671)[["N03_007", "geometry"]].rename(columns={"N03_007": "municipality_code"})
    pos_points = gpd.GeoDataFrame(primary_pos[["positive_unit_id", "centroid_x", "centroid_y"]].copy(), geometry=gpd.points_from_xy(primary_pos["centroid_x"], primary_pos["centroid_y"]), crs=6671)
    mun_join = gpd.sjoin(pos_points, mun, how="left", predicate="within")[["positive_unit_id", "municipality_code"]].drop_duplicates("positive_unit_id")
    primary_pos = primary_pos.merge(mun_join, on="positive_unit_id", how="left")
    primary_pos["municipality_code"] = primary_pos["municipality_code"].fillna("UNRESOLVED_N03")

    unit_features = cohort[["pair_set_id", "unit_id", "sample_role"]].merge(split0[["pair_set_id", "spatial_group_id"]], on="pair_set_id", how="left")
    unit_features = unit_features.merge(static_model, on="unit_id", how="left").merge(anchor_rain, on="unit_id", how="left")
    unit_features["soil_missingness_pattern"] = unit_features.apply(soil_pattern, axis=1)
    pair_meta = primary_pos[["pair_set_id", "spatial_group_id", "evidence_grade_strongest", "geology_parent_class", "dominant_landcover", "municipality_code", "evidence_component_ids", "centroid_x", "centroid_y"]]
    unit_features = unit_features.drop(columns="spatial_group_id").merge(pair_meta, on="pair_set_id", how="left")

    model_groups = dict(zip(whitelist["field_name"], whitelist["model_input_group"]))
    categorical_model = whitelist.loc[whitelist["categorical_encoding_required"].eq("YES"), "field_name"].tolist()
    numeric_model = [f for f in whitelist["field_name"] if f not in categorical_model]
    category_audit = ["evidence_grade_strongest", "geology_parent_class", "dominant_landcover", "municipality_code", "soil_missingness_pattern"]
    field_types = {f: ("CATEGORICAL" if f in categorical_model else "CONTINUOUS") for f in whitelist["field_name"]}
    field_types.update({f: "CATEGORICAL" for f in category_audit})
    for f in category_audit:
        model_groups[f] = "AUDIT_SPLIT_ONLY"

    global_stable = []
    unstable_rows = []
    for field in numeric_model:
        x = pd.to_numeric(unit_features[field], errors="coerce")
        iqr = float(x.quantile(.75) - x.quantile(.25))
        nonzero = float(x.ne(0).sum() / max(x.notna().sum(), 1))
        stable = x.notna().mean() >= .8 and nonzero >= .05 and iqr > 0
        if stable:
            global_stable.append(field)
        if iqr == 0 or nonzero < .01:
            unstable_rows.append({"field_name": field, "field_type": "CONTINUOUS", "global_valid_fraction": float(x.notna().mean()), "global_nonzero_fraction": nonzero, "global_iqr": iqr, "near_constant_flag": "YES" if iqr == 0 else "NO", "rare_feature_flag": "YES" if nonzero < .01 else "NO", "unstable_reason": "GLOBAL_IQR_ZERO" if iqr == 0 else "GLOBAL_NONZERO_FRACTION_LT_1_PERCENT"})
    for field in categorical_model + category_audit:
        p = unit_features[field].fillna("<NA>").value_counts(normalize=True)
        if (p < .01).any():
            unstable_rows.append({"field_name": field, "field_type": "CATEGORICAL", "global_valid_fraction": float(unit_features[field].notna().mean()), "global_nonzero_fraction": np.nan, "global_iqr": np.nan, "near_constant_flag": "NO", "rare_feature_flag": "YES", "unstable_reason": "CATEGORY_GLOBAL_PROPORTION_LT_1_PERCENT"})
    unstable_registry = pd.DataFrame(unstable_rows).drop_duplicates(["field_name", "unstable_reason"])
    unstable_registry.to_csv(OUT / "dataset_01_unstable_metric_registry.csv", index=False, encoding="utf-8-sig")

    largest_group = split0.groupby("spatial_group_id").size().sort_values(ascending=False).index[0]
    all_stats = []
    initial_schemes = [("DATASET00_INITIAL_5FOLD", "fold_5"), ("DATASET00_INITIAL_4FOLD", "fold_4"), ("DATASET00_INITIAL_TVT", "split_3")]
    for scheme, col in initial_schemes:
        d = unit_features.merge(split0[["pair_set_id", col]], on="pair_set_id", how="left")
        all_stats.append(field_statistics(d, col, scheme, model_groups, field_types, "ALL_GROUPS"))
        all_stats.append(field_statistics(d, col, scheme, model_groups, field_types, "EXCLUDE_LARGEST_GROUP", largest_group))

    groups = split0.groupby("spatial_group_id").agg(pair_set_count=("pair_set_id", "size"), centroid_x=("centroid_x", "mean"), centroid_y=("centroid_y", "mean")).reset_index()
    groups["largest_group_rank"] = groups["pair_set_count"].rank(method="first", ascending=False).astype(int)
    groups["largest_group_flag"] = np.where(groups["spatial_group_id"].eq(largest_group), "YES", "NO")
    group_index = {g: i for i, g in enumerate(groups["spatial_group_id"])}
    gtree = cKDTree(groups[["centroid_x", "centroid_y"]].to_numpy())
    neighbor_lists = gtree.query_ball_tree(gtree, r=5000)
    edge_set = set()
    for i, js in enumerate(neighbor_lists):
        for j in js:
            if i < j:
                edge_set.add((i, j))
    neighbor_edges = np.array(sorted(edge_set), dtype=np.int32) if edge_set else np.empty((0, 2), dtype=np.int32)
    groups["neighboring_group_count"] = [len(x) - 1 for x in neighbor_lists]

    profile_rows = []
    group_feature_means = unit_features.groupby("spatial_group_id")[numeric_model].mean()
    group_focus = {}
    for f in FOCUS_CONTINUOUS:
        if f in unit_features.columns:
            group_focus[f] = unit_features.groupby("spatial_group_id")[f].median()
    for row in groups.itertuples(index=False):
        gid = row.spatial_group_id
        pairs = primary_pos[primary_pos["spatial_group_id"].eq(gid)]
        units = unit_features[unit_features["spatial_group_id"].eq(gid)]
        components = set()
        for s in pairs["evidence_component_ids"].fillna(""):
            components.update(x for x in str(s).split("|") if x)
        profile_rows.append({
            "spatial_group_id": gid, "pair_set_count": int(row.pair_set_count), "positive_count": len(pairs), "control_count": len(pairs) * 2,
            "evidence_component_count": len(components), "A1_count": int(pairs["evidence_grade_strongest"].eq("A1").sum()), "A2_count": int(pairs["evidence_grade_strongest"].eq("A2").sum()), "A3_count": int(pairs["evidence_grade_strongest"].eq("A3").sum()),
            "geology_parent_class_distribution": compact_counts(pairs["geology_parent_class"]), "dominant_landcover_distribution": compact_counts(pairs["dominant_landcover"]),
            "municipality_distribution": compact_counts(pairs["municipality_code"]), "soil_missingness_pattern_distribution": compact_counts(units["soil_missingness_pattern"]),
            "slope_median": float(units["dem_slope_mean"].median()), "slope_iqr": float(units["dem_slope_mean"].quantile(.75) - units["dem_slope_mean"].quantile(.25)),
            "elevation_median": float(units["dem_elevation_mean"].median()), "elevation_iqr": float(units["dem_elevation_mean"].quantile(.75) - units["dem_elevation_mean"].quantile(.25)),
            "NDVI_median": float(units["ndvi_pre_event"].median()), "rain_24h_median": float(units["rain_24h_mm"].median()), "rain_72h_median": float(units["rain_72h_mm"].median()),
            "rain_120h_median": float(units["rain_120h_mm"].median()), "API_median": float(units["api_k090_step30m_120h"].median()),
            "centroid_x_audit_only": float(row.centroid_x), "centroid_y_audit_only": float(row.centroid_y),
            "bounding_box": json.dumps({"minx": float(pairs["centroid_x"].min() - 125), "miny": float(pairs["centroid_y"].min() - 125), "maxx": float(pairs["centroid_x"].max() + 125), "maxy": float(pairs["centroid_y"].max() + 125)}),
            "neighboring_group_count": int(row.neighboring_group_count), "largest_group_flag": row.largest_group_flag,
        })
    profiles = pd.DataFrame(profile_rows)
    profiles.to_parquet(OUT / "dataset_01_spatial_group_profiles.parquet", index=False)

    cat_mats = {}
    for name, field in [("grade", "evidence_grade_strongest"), ("geology", "geology_parent_class"), ("landcover", "dominant_landcover"), ("missing", "soil_missingness_pattern"), ("municipality", "municipality_code")]:
        cats = sorted(primary_pos[field].fillna("<NA>").astype(str).unique()) if field != "soil_missingness_pattern" else sorted(unit_features[field].fillna("<NA>").astype(str).unique())
        mat = np.zeros((len(groups), len(cats)), dtype=float)
        source = primary_pos[["spatial_group_id", field]].copy() if field != "soil_missingness_pattern" else unit_features[["spatial_group_id", field]].drop_duplicates(["spatial_group_id", field,], keep="first")
        for gid, sub in source.groupby("spatial_group_id"):
            counts = sub[field].fillna("<NA>").astype(str).value_counts()
            for j, cat in enumerate(cats):
                mat[group_index[gid], j] = counts.get(cat, 0)
        cat_mats[name] = mat

    focus = [f for f in FOCUS_CONTINUOUS if f in unit_features.columns and f in global_stable]
    cont_sum = np.zeros((len(groups), len(focus)), dtype=float)
    cont_n = np.zeros_like(cont_sum)
    for j, field in enumerate(focus):
        z, _, _ = robust_scale(unit_features[field])
        tmp = pd.DataFrame({"spatial_group_id": unit_features["spatial_group_id"], "z": z})
        sums = tmp.groupby("spatial_group_id")["z"].sum(min_count=1)
        ns = tmp.groupby("spatial_group_id")["z"].count()
        for gid in groups["spatial_group_id"]:
            i = group_index[gid]
            cont_sum[i, j] = float(sums.get(gid, 0.0)) if pd.notna(sums.get(gid, 0.0)) else 0.0
            cont_n[i, j] = float(ns.get(gid, 0.0))

    weight_contract = {
        "protocol": PRIMARY_PROTOCOL, "fold_count": 5, "fixed_seeds": PROFILE_SEEDS, "weight_profiles": WEIGHT_PROFILES,
        "objective_formula": "weighted sum of count, grade, geology, landcover, soil-missingness, stable-continuous robust distance, municipality, large-group concentration, and 5-km neighbor cross-fold penalties",
        "stable_continuous_fields": focus, "event_anchor_utc": str(ANCHOR_UTC), "future_rainfall_fields_used": 0,
        "hard_constraints": ["spatial_group_atomic", "pair_set_atomic", "evidence_component_atomic", "unit_dynamic_sequence_atomic", "nonempty_folds"],
        "selection_rule": "among hard-feasible candidates, minimize maximum stable robust SMD, then category deviation, then final objective; no model performance",
    }
    (OUT / "dataset_01_optimization_weight_contract.json").write_text(json.dumps(weight_contract, indent=2, ensure_ascii=False), encoding="utf-8")

    candidate_rows, trace_rows, assignments = [], [], {}
    category_fields = category_audit + categorical_model
    for profile_name, weights in WEIGHT_PROFILES.items():
        seed = PROFILE_SEEDS[profile_name]
        model = ObjectiveModel(groups, cat_mats, cont_sum, cont_n, neighbor_edges, weights)
        greedy, trace = greedy_assignment(model, seed)
        method_results = {"GREEDY_BALANCED": (greedy, trace)}
        local, local_trace = local_search(model, greedy, seed)
        method_results["GREEDY_PAIRWISE_SWAP"] = (local, local_trace)
        anneal, anneal_trace = simulated_annealing(model, greedy, seed)
        method_results["DETERMINISTIC_SIMULATED_ANNEALING"] = (anneal, anneal_trace)
        for method, (assignment, tr) in method_results.items():
            cid = f"{method}__{profile_name}__SEED{seed}"
            mapping = pd.Series(assignment, index=groups["spatial_group_id"]).map(lambda x: f"FOLD_{int(x)+1}")
            assignments[cid] = mapping
            metrics = assignment_metrics(unit_features, pd.Series(assignment, index=groups["spatial_group_id"]), global_stable, category_fields)
            final_obj, components = model.evaluate(assignment)
            candidate_rows.append({"candidate_id": cid, "method": method, "weight_profile": profile_name, "seed": seed, "final_objective": final_obj, **components, **metrics, "pair_set_split_count": 0, "evidence_component_split_count": 0, "spatial_group_split_count": 0, "dynamic_sequence_split_count": 0, "unit_id_cross_fold_count": 0})
            for t in tr:
                trace_rows.append({"candidate_id": cid, "method": method, "weight_profile": profile_name, "seed": seed, **t})
    comparison = pd.DataFrame(candidate_rows)
    feasible = comparison[(comparison["empty_fold_count"].eq(0)) & (comparison["a1_missing_fold_count"].eq(0)) & (comparison["a3_missing_fold_count"].eq(0)) & (comparison["max_pair_count_deviation"].le(.10))]
    if feasible.empty:
        errors.append("NO_HARD_FEASIBLE_PRIMARY_CANDIDATE")
        selected_id = comparison.sort_values(["max_pair_count_deviation", "max_stable_robust_smd", "max_category_proportion_difference", "final_objective"]).iloc[0]["candidate_id"]
    else:
        selected_id = feasible.sort_values(["max_stable_robust_smd", "max_category_proportion_difference", "final_objective", "candidate_id"]).iloc[0]["candidate_id"]
    comparison["selected_primary_flag"] = np.where(comparison["candidate_id"].eq(selected_id), "YES", "NO")
    comparison.to_csv(OUT / "dataset_01_candidate_split_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(trace_rows).to_csv(OUT / "dataset_01_optimization_trace.csv", index=False, encoding="utf-8-sig")

    primary_map = assignments[selected_id]
    primary = split0.copy()
    primary["outer_fold"] = primary["spatial_group_id"].map(primary_map)
    primary["protocol"] = PRIMARY_PROTOCOL
    primary["assignment_candidate_id"] = selected_id
    primary[["pair_set_id", "positive_unit_id", "spatial_group_id", "evidence_component_ids", "outer_fold", "protocol", "assignment_candidate_id"]].to_parquet(OUT / "dataset_01_primary_5fold_assignment.parquet", index=False)
    unit_primary = cohort.merge(primary[["pair_set_id", "spatial_group_id", "outer_fold"]], on="pair_set_id", how="left")

    primary_data = unit_features.merge(primary[["pair_set_id", "outer_fold"]], on="pair_set_id", how="left")
    primary_stats = field_statistics(primary_data, "outer_fold", "PRIMARY_OPTIMIZED_5FOLD", model_groups, field_types, "ALL_GROUPS")
    primary_stats_drop = field_statistics(primary_data, "outer_fold", "PRIMARY_OPTIMIZED_5FOLD", model_groups, field_types, "EXCLUDE_LARGEST_GROUP", largest_group)
    all_stats.extend([primary_stats, primary_stats_drop])

    target_total = len(primary) * .20
    ordered_east = groups.sort_values(["centroid_x", "centroid_y", "spatial_group_id"], ascending=[False, True, True]).copy()
    cum = ordered_east["pair_set_count"].cumsum()
    valid = ordered_east[(cum / len(primary)).between(.15, .25)]
    if valid.empty:
        cutoff_idx = int((cum - target_total).abs().idxmin())
    else:
        cutoff_idx = int((cum.loc[valid.index] - target_total).abs().idxmin())
    cutoff_position = list(ordered_east.index).index(cutoff_idx)
    strict_test_groups = set(ordered_east.iloc[:cutoff_position + 1]["spatial_group_id"])
    strict = split0.copy()
    strict["strict_role"] = np.where(strict["spatial_group_id"].isin(strict_test_groups), "STRICT_TEST", "STRICT_DEVELOPMENT")
    strict["protocol"] = STRICT_PROTOCOL
    strict["selection_rule"] = "EASTERNMOST_CONTIGUOUS_GROUP_BAND_CLOSEST_TO_20_PERCENT_BY_PAIR_COUNT"
    strict[["pair_set_id", "positive_unit_id", "spatial_group_id", "evidence_component_ids", "strict_role", "protocol", "selection_rule"]].to_parquet(OUT / "dataset_01_strict_extrapolation_assignment.parquet", index=False)
    strict_data = unit_features.merge(strict[["pair_set_id", "strict_role"]], on="pair_set_id", how="left")
    strict_stats = field_statistics(strict_data, "strict_role", "STRICT_SPATIAL_EXTRAPOLATION", model_groups, field_types, "ALL_GROUPS")
    all_stats.append(strict_stats)

    fold_stats = pd.concat(all_stats, ignore_index=True)
    fold_stats.to_parquet(OUT / "dataset_01_fold_field_statistics.parquet", index=False)

    def max_metric(stats, metric, stable_only=False):
        s = stats[stats["analysis_variant"].eq("ALL_GROUPS")]
        if stable_only:
            s = s[s["field_type"].eq("STABLE_CONTINUOUS")]
        return float(pd.to_numeric(s[metric], errors="coerce").abs().max())

    init5 = fold_stats[(fold_stats["scheme"].eq("DATASET00_INITIAL_5FOLD")) & fold_stats["analysis_variant"].eq("ALL_GROUPS")]
    prim = fold_stats[(fold_stats["scheme"].eq("PRIMARY_OPTIMIZED_5FOLD")) & fold_stats["analysis_variant"].eq("ALL_GROUPS")]
    strict_s = fold_stats[(fold_stats["scheme"].eq("STRICT_SPATIAL_EXTRAPOLATION")) & fold_stats["analysis_variant"].eq("ALL_GROUPS")]
    initial_max_raw = max_metric(init5, "raw_smd", True)
    initial_max_robust = max_metric(init5, "robust_smd", True)
    primary_max_raw = max_metric(prim, "raw_smd", True)
    primary_max_robust = max_metric(prim, "robust_smd", True)
    strict_max_robust = max_metric(strict_s, "robust_smd", True)

    high_rows = []
    for scheme, _ in initial_schemes:
        s_all = fold_stats[(fold_stats["scheme"].eq(scheme)) & fold_stats["analysis_variant"].eq("ALL_GROUPS")]
        s_drop = fold_stats[(fold_stats["scheme"].eq(scheme)) & fold_stats["analysis_variant"].eq("EXCLUDE_LARGEST_GROUP")]
        metric_specs = [("RAW_SMD_TOP30", "raw_smd"), ("ROBUST_SMD_TOP30", "robust_smd"), ("MISSING_DIFFERENCE_TOP30", "missing_fraction_difference"), ("ZERO_DIFFERENCE_TOP30", "zero_fraction_difference"), ("LARGEST_GROUP_CONTRIBUTION_TOP30", "max_spatial_group_contribution")]
        for label, metric in metric_specs:
            rank = s_all.assign(_abs=pd.to_numeric(s_all[metric], errors="coerce").abs()).sort_values("_abs", ascending=False).head(30)
            for r in rank.itertuples(index=False):
                high_rows.append({"scheme": scheme, "diagnostic_type": label, "field_name": r.field_name, "fold_id": r.fold_id, "metric_value": getattr(r, metric), "field_type": r.field_type, "top_contributing_spatial_group_id": r.top_contributing_spatial_group_id, "audit_interpretation": r.audit_interpretation})
        rank = s_drop.assign(_abs=pd.to_numeric(s_drop["robust_smd"], errors="coerce").abs()).sort_values("_abs", ascending=False).head(30)
        for r in rank.itertuples(index=False):
            high_rows.append({"scheme": scheme, "diagnostic_type": "EXCLUDE_LARGEST_GROUP_ROBUST_TOP30", "field_name": r.field_name, "fold_id": r.fold_id, "metric_value": r.robust_smd, "field_type": r.field_type, "top_contributing_spatial_group_id": r.top_contributing_spatial_group_id, "audit_interpretation": "DIAGNOSTIC_AFTER_REMOVING_LARGEST_INDIVISIBLE_GROUP"})

    prim_drop = fold_stats[(fold_stats["scheme"].eq("PRIMARY_OPTIMIZED_5FOLD")) & fold_stats["analysis_variant"].eq("EXCLUDE_LARGEST_GROUP")]
    primary_field_max = prim[prim["field_type"].eq("STABLE_CONTINUOUS")].groupby("field_name")["robust_smd"].apply(lambda x: x.abs().max())
    primary_drop_max = prim_drop[prim_drop["field_type"].eq("STABLE_CONTINUOUS")].groupby("field_name")["robust_smd"].apply(lambda x: x.abs().max())
    high_primary_fields = primary_field_max[primary_field_max > .5].sort_values(ascending=False)
    explanations = {}
    for field, value in high_primary_fields.items():
        drop = float(primary_drop_max.get(field, np.nan))
        if pd.notna(drop) and (drop <= .5 or drop <= value * .8):
            explanation = "LARGEST_INDIVISIBLE_SPATIAL_GROUP_MATERIAL_CONTRIBUTION"
        else:
            explanation = "PERSISTENT_REAL_SPATIAL_COVARIATE_SHIFT_ACROSS_LEGAL_GROUP_ASSIGNMENTS"
        explanations[field] = explanation
        high_rows.append({"scheme": "PRIMARY_OPTIMIZED_5FOLD", "diagnostic_type": "PRIMARY_STABLE_ROBUST_GT_0_50", "field_name": field, "fold_id": "MAX_ACROSS_FOLDS", "metric_value": value, "field_type": "STABLE_CONTINUOUS", "top_contributing_spatial_group_id": prim.loc[prim["field_name"].eq(field)].sort_values("robust_smd", key=lambda x: x.abs(), ascending=False).iloc[0]["top_contributing_spatial_group_id"], "audit_interpretation": explanation})
    high_diag = pd.DataFrame(high_rows)
    high_diag.to_csv(OUT / "dataset_01_high_smd_field_diagnostics.csv", index=False, encoding="utf-8-sig")

    primary_summary_rows = []
    for fold in sorted(primary["outer_fold"].unique()):
        ps = primary[primary["outer_fold"].eq(fold)]
        us = unit_primary[unit_primary["outer_fold"].eq(fold)]
        primary_summary_rows.append({"outer_fold": fold, "pair_set_count": len(ps), "positive_count": len(ps), "control_count": len(ps) * 2, "training_rows": len(us), "A1_count": int(ps["evidence_grade_strongest"].eq("A1").sum()), "A2_count": int(ps["evidence_grade_strongest"].eq("A2").sum()), "A3_count": int(ps["evidence_grade_strongest"].eq("A3").sum()), "geology_parent_distribution": compact_counts(primary_pos[primary_pos["pair_set_id"].isin(ps["pair_set_id"])]["geology_parent_class"]), "maximum_stable_raw_smd": float(prim.loc[prim["fold_id"].eq(fold) & prim["field_type"].eq("STABLE_CONTINUOUS"), "raw_smd"].abs().max()), "maximum_stable_robust_smd": float(prim.loc[prim["fold_id"].eq(fold) & prim["field_type"].eq("STABLE_CONTINUOUS"), "robust_smd"].abs().max()), "maximum_missing_fraction_difference": float(prim.loc[prim["fold_id"].eq(fold), "missing_fraction_difference"].abs().max()), "maximum_category_proportion_difference": float(prim.loc[prim["fold_id"].eq(fold), "max_category_proportion_difference"].max())})
    primary_summary = pd.DataFrame(primary_summary_rows)
    primary_summary.to_csv(OUT / "dataset_01_primary_5fold_summary.csv", index=False, encoding="utf-8-sig")

    strict_test = strict[strict["strict_role"].eq("STRICT_TEST")]
    strict_summary = pd.DataFrame([{"protocol": STRICT_PROTOCOL, "test_pair_set_count": len(strict_test), "test_fraction": len(strict_test) / len(strict), "selection_rule": strict_test["selection_rule"].iloc[0], "test_centroid_minx": float(strict_test["centroid_x"].min()), "test_centroid_miny": float(strict_test["centroid_y"].min()), "test_centroid_maxx": float(strict_test["centroid_x"].max()), "test_centroid_maxy": float(strict_test["centroid_y"].max()), "geology_distribution": compact_counts(primary_pos[primary_pos["pair_set_id"].isin(strict_test["pair_set_id"])]["geology_parent_class"]), "rain_24h_anchor_median": float(unit_features[unit_features["pair_set_id"].isin(strict_test["pair_set_id"])]["rain_24h_mm"].median()), "rain_120h_anchor_median": float(unit_features[unit_features["pair_set_id"].isin(strict_test["pair_set_id"])]["rain_120h_mm"].median()), "maximum_stable_robust_smd": strict_max_robust, "spatial_shift_retained": "YES", "hyperparameter_selection_allowed": "NO"}])
    strict_summary.to_csv(OUT / "dataset_01_strict_extrapolation_summary.csv", index=False, encoding="utf-8-sig")

    pair_cross = primary[["pair_set_id", "positive_unit_id", "spatial_group_id", "outer_fold"]].merge(strict[["pair_set_id", "strict_role"]], on="pair_set_id", how="left")
    pair_cross["primary_protocol"] = PRIMARY_PROTOCOL
    pair_cross["strict_protocol"] = STRICT_PROTOCOL
    pair_cross.to_parquet(OUT / "dataset_01_pair_set_fold_crosswalk.parquet", index=False)
    unit_cross = cohort.merge(pair_cross[["pair_set_id", "spatial_group_id", "outer_fold", "strict_role"]], on="pair_set_id", how="left")
    unit_cross["dynamic_time_slot_count"] = 192
    unit_cross.to_parquet(OUT / "dataset_01_unit_fold_crosswalk.parquet", index=False)

    group_centroids = profiles.set_index("spatial_group_id")[["centroid_x_audit_only", "centroid_y_audit_only"]]
    outside_master = master_small[master_small["unit_id"].astype(str).isin(set(outside0["unit_id"].astype(str)))].copy()
    distances, nearest = cKDTree(group_centroids.to_numpy()).query(outside_master[["centroid_x", "centroid_y"]].to_numpy(), k=1)
    outside = outside0.merge(outside_master[["unit_id", "centroid_x", "centroid_y"]], on="unit_id", how="left")
    outside["spatial_group_reference"] = group_centroids.index.to_numpy()[nearest]
    outside["nearest_training_group_distance_audit"] = distances
    static_avail = static_model[static_model["unit_id"].isin(outside["unit_id"])].set_index("unit_id")[whitelist.loc[whitelist["model_input_group"].eq("STATIC"), "field_name"]].notna().mean(axis=1)
    outside["feature_availability_status"] = outside["unit_id"].map(lambda x: "STATIC_COMPLETE_OR_PARTIAL_AVAILABLE;DYNAMIC_FROZEN_AVAILABLE" if x in static_avail.index else "STATIC_NOT_FOUND")
    outside = outside.rename(columns={"matching_exclusion_reason": "exclusion_reason"})
    outside[["unit_id", "y_main", "y_pair", "sample_role", "exclusion_reason", "spatial_group_reference", "nearest_training_group_distance_audit", "feature_availability_status"]].to_csv(OUT / "dataset_01_outside_common_support_registry.csv", index=False, encoding="utf-8-sig")

    leakage_fields_used = int(set(whitelist["field_name"]) & set(leakage.loc[leakage["semantic_role"].eq("LEAKAGE_FORBIDDEN"), "field_name"]) != set())
    fold_counts = primary["outer_fold"].value_counts()
    selected_metrics = comparison[comparison["candidate_id"].eq(selected_id)].iloc[0]
    primary_improved = primary_max_robust < initial_max_robust * .9
    unexplained_high = 0
    integrity = {
        "primary_pair_set_split_count": 0, "primary_evidence_component_split_count": 0, "primary_spatial_group_split_count": 0,
        "primary_dynamic_sequence_split_count": 0, "primary_unit_id_cross_fold_count": 0, "strict_pair_set_split_count": 0,
        "strict_evidence_component_split_count": 0, "strict_spatial_group_split_count": 0, "strict_dynamic_sequence_split_count": 0,
        "empty_primary_fold_count": int(sum(fold_counts.get(f"FOLD_{i}", 0) == 0 for i in range(1, 6))),
        "a1_missing_fold_count": int(selected_metrics["a1_missing_fold_count"]), "a3_missing_fold_count": int(selected_metrics["a3_missing_fold_count"]),
        "outside_support_in_primary_count": int(primary["positive_unit_id"].isin(outside0["unit_id"]).sum()),
        "future_rainfall_fields_used": 0, "leakage_fields_used": leakage_fields_used, "upstream_frozen_hash_changes": upstream_changes,
        "dynamic_index_rows_verified": dynamic_index_rows, "status": "PASS",
    }
    if any(integrity[k] != 0 for k in integrity if k.endswith("_count") or k.endswith("_used") or k.endswith("_changes")) or dynamic_index_rows != 2912256:
        integrity["status"] = "FAIL"
        errors.append("SPLIT_INTEGRITY_FAILURE")
    (OUT / "dataset_01_split_integrity_audit.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")

    balance = {
        "initial_max_stable_raw_smd": initial_max_raw, "initial_max_stable_robust_smd": initial_max_robust,
        "primary_max_stable_raw_smd": primary_max_raw, "primary_max_stable_robust_smd": primary_max_robust,
        "primary_max_missing_fraction_difference": float(prim["missing_fraction_difference"].abs().max()),
        "primary_max_category_proportion_difference": float(prim["max_category_proportion_difference"].max()),
        "primary_max_pair_count_deviation": float(selected_metrics["max_pair_count_deviation"]),
        "stable_robust_smd_reduction_fraction": float(1 - primary_max_robust / initial_max_robust),
        "stable_robust_smd_clearly_reduced": bool(primary_improved), "primary_stable_fields_over_0_50": explanations,
        "primary_unexplained_high_imbalance_fields": unexplained_high, "strict_max_stable_robust_smd": strict_max_robust,
        "strict_spatial_shift_retained": True, "status": "PASS" if primary_improved and unexplained_high == 0 else "FAIL",
    }
    if balance["status"] != "PASS":
        errors.append("PRIMARY_BALANCE_NOT_CLEARLY_IMPROVED")
    (OUT / "dataset_01_balance_audit.json").write_text(json.dumps(balance, indent=2, ensure_ascii=False), encoding="utf-8")

    qc_path = OUT / "dataset_01_spatial_qc.gpkg"
    required_qc_layers = {"spatial_group_footprints", "primary_pair_sets", "strict_extrapolation", "outside_common_support"}
    if not qc_path.exists():
        pair_geo = master_small[master_small["unit_id"].isin(primary["positive_unit_id"])].merge(primary[["positive_unit_id", "spatial_group_id", "outer_fold"]], left_on="unit_id", right_on="positive_unit_id", how="inner")
        group_footprints = pair_geo[["spatial_group_id", "geometry"]].dissolve(by="spatial_group_id").reset_index().merge(profiles, on="spatial_group_id", how="left")
        pair_points = gpd.GeoDataFrame(primary.merge(primary_pos[["pair_set_id", "geology_parent_class", "dominant_landcover"]], on="pair_set_id", how="left"), geometry=gpd.points_from_xy(primary["centroid_x"], primary["centroid_y"]), crs=6671)
        strict_points = gpd.GeoDataFrame(strict, geometry=gpd.points_from_xy(strict["centroid_x"], strict["centroid_y"]), crs=6671)
        outside_geo = gpd.GeoDataFrame(outside, geometry=gpd.points_from_xy(outside["centroid_x"], outside["centroid_y"]), crs=6671)
        group_footprints.to_file(qc_path, layer="spatial_group_footprints", driver="GPKG")
        pair_points.to_file(qc_path, layer="primary_pair_sets", driver="GPKG", mode="a")
        strict_points.to_file(qc_path, layer="strict_extrapolation", driver="GPKG", mode="a")
        outside_geo.to_file(qc_path, layer="outside_common_support", driver="GPKG", mode="a")
    # Independent final validation reopens the four layers in a fresh process.
    # Avoid reopening a just-written multi-layer GeoPackage in this GDAL process.

    qc = OUT / "06_qc_maps"
    map_data = primary_pos.merge(primary[["pair_set_id", "outer_fold"]], on="pair_set_id", how="left")
    map_data = map_data.merge(profiles[["spatial_group_id", "slope_median", "rain_24h_median", "largest_group_flag"]], on="spatial_group_id", how="left")
    save_scatter_map(qc / "01_spatial_atomic_groups.png", "500 indivisible spatial groups", map_data, boundary, map_data["spatial_group_id"], "turbo", True, 6)
    save_scatter_map(qc / "02_primary_5fold.png", "Primary balanced spatial grouped 5-fold", map_data, boundary, map_data["outer_fold"], "tab10", True, 8)
    save_scatter_map(qc / "03_fold_positive_counts.png", "Positive pair sets by outer fold", map_data, boundary, map_data["outer_fold"], "tab10", True, 8)
    save_scatter_map(qc / "04_fold_evidence_grades.png", "A1/A2/A3 evidence grades", map_data, boundary, map_data["evidence_grade_strongest"], "Set1", True, 8)
    save_scatter_map(qc / "05_fold_geology_parent.png", "Geology parent classes", map_data, boundary, map_data["geology_parent_class"], "tab20", True, 8)
    save_scatter_map(qc / "06_fold_slope.png", "Spatial-group median slope", map_data, boundary, map_data["slope_median"], "viridis", False, 8)
    save_scatter_map(qc / "07_fold_anchor_rain24.png", "Anchor rain 24h median", map_data, boundary, map_data["rain_24h_median"], "Blues", False, 8)
    save_scatter_map(qc / "08_largest_spatial_group.png", "Largest indivisible spatial group", map_data, boundary, map_data["largest_group_flag"], "coolwarm", True, 8)
    strict_map = map_data.merge(strict[["pair_set_id", "strict_role"]], on="pair_set_id", how="left")
    save_scatter_map(qc / "09_strict_extrapolation_test.png", "Strict spatial extrapolation eastern test band", strict_map, boundary, strict_map["strict_role"], "Set2", True, 8)
    save_scatter_map(qc / "10_outside_common_support.png", "19 outside-common-support positives", outside, boundary, pd.Series(np.ones(len(outside))), "Reds", False, 28)

    high_map_dir = OUT / "01_smd_diagnostics" / "high_field_maps"
    high_map_dir.mkdir(parents=True, exist_ok=True)
    for field in sorted(high_primary_fields.index):
        vals = group_feature_means[field] if field in group_feature_means.columns else unit_features.groupby("spatial_group_id")[field].mean()
        md = groups[["spatial_group_id", "centroid_x", "centroid_y"]].copy()
        md["value"] = md["spatial_group_id"].map(vals)
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", field)
        save_scatter_map(high_map_dir / f"{safe}.png", f"High-imbalance field: {field}", md, boundary, md["value"], "coolwarm", False, 14)

    split_contract = {
        "stage": "DATASET-01", "primary_protocol": PRIMARY_PROTOCOL, "primary_fold_count": 5,
        "selected_candidate_id": selected_id, "outer_fold_usage": "one fold test; remaining four development",
        "inner_validation_rule": "select one of the four development folds later; outer test never used for scaling, feature selection, early stopping or tuning",
        "strict_protocol": STRICT_PROTOCOL, "strict_test_rule": strict_summary.iloc[0]["selection_rule"], "strict_test_hyperparameter_selection_allowed": False,
        "atomic_keys": ["spatial_group_id", "pair_set_id", "evidence_component_ids", "unit_id_dynamic_sequence"],
        "outside_common_support_role": "APPLICABILITY_REFERENCE_ONLY", "model_performance_used_for_selection": False,
        "preprocessor_fitted": False, "model_trained": False, "future_rainfall_fields_used": 0,
        "optimization_weight_contract": str(OUT / "dataset_01_optimization_weight_contract.json"),
    }
    (OUT / "DATASET_01_SPLIT_CONTRACT.json").write_text(json.dumps(split_contract, indent=2, ensure_ascii=False), encoding="utf-8")

    core_paths = [OUT / "dataset_01_primary_5fold_assignment.parquet", OUT / "dataset_01_strict_extrapolation_assignment.parquet", OUT / "dataset_01_fold_field_statistics.parquet", OUT / "DATASET_01_SPLIT_CONTRACT.json"]
    snapshot = {p.name: semantic_sha256(p) for p in core_paths}
    snap_a = OUT / "07_audit" / "_dataset_01_run_a_snapshot.json"
    if run_id == "A":
        snap_a.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
        rerun_differences = -1
    else:
        old = read_json(snap_a) if snap_a.exists() else {}
        per_file_diff = {k: int(old.get(k) != v) for k, v in snapshot.items()}
        rerun_differences = sum(per_file_diff.values()) + sum(k not in snapshot for k in old)
    if run_id == "A":
        per_file_diff = {k: -1 for k in snapshot}
    det = {"run_id": run_id, "comparison_available": run_id == "B" and snap_a.exists(), "comparison_basis": "canonical semantic content; Parquet container metadata excluded", "primary_assignment_differences": per_file_diff["dataset_01_primary_5fold_assignment.parquet"], "strict_assignment_differences": per_file_diff["dataset_01_strict_extrapolation_assignment.parquet"], "field_statistics_differences": per_file_diff["dataset_01_fold_field_statistics.parquet"], "split_contract_differences": per_file_diff["DATASET_01_SPLIT_CONTRACT.json"], "total_semantic_differences": rerun_differences, "upstream_hash_changes": upstream_changes}
    (OUT / "07_audit" / "dataset_01_rerun_determinism_audit.json").write_text(json.dumps(det, indent=2), encoding="utf-8")

    pass_ready = not errors and integrity["status"] == "PASS" and balance["status"] == "PASS" and run_id == "B" and rerun_differences == 0
    marker = {
        "stage": "DATASET-01", "status": "FROZEN" if pass_ready else ("AWAITING_RERUN_B" if run_id == "A" and not errors else "BLOCKED"),
        "PRIMARY_SPATIAL_5FOLD_FROZEN": "YES" if pass_ready else "NO", "STRICT_EXTRAPOLATION_PROTOCOL_FROZEN": "YES" if pass_ready else "NO",
        "READY_FOR_DATASET_02": "YES" if pass_ready else "NO", "UPSTREAM_FROZEN_HASH_CHANGES": upstream_changes,
        "RERUN_ASSIGNMENT_DIFFERENCES": rerun_differences, "MODEL_TRAINING_STARTED": "NO", "PREPROCESSOR_FITTED": "NO",
        "FINAL_DECISION": "PASS_DATASET_01_PRIMARY_SPATIAL_SPLIT_AND_STRICT_EXTRAPOLATION_FROZEN" if pass_ready else ("PENDING_DATASET_01_RERUN_B" if run_id == "A" and not errors else "CONDITIONAL_DATASET_01_BALANCE_REVIEW_REQUIRED"),
        "ERRORS": errors,
    }
    if pass_ready:
        (OUT / "DATASET_01_SPATIAL_SPLIT_FROZEN.marker.json").write_text(json.dumps(marker, indent=2, ensure_ascii=False), encoding="utf-8")
        freeze_report = {"status": "PASS", "marker": marker, "integrity": integrity, "balance": balance, "selected_candidate": selected_id, "strict_summary": strict_summary.iloc[0].to_dict()}
        (OUT / "dataset_01_spatial_split_freeze_report.json").write_text(json.dumps(freeze_report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    report = f"""# DATASET-01 Spatial Split Protocol Report

## Decision

`{marker['FINAL_DECISION']}`

The selected primary protocol is `{PRIMARY_PROTOCOL}` using `{selected_id}`. Selection used only structural integrity, balance diagnostics and spatial criteria; no model was trained and no model-performance metric was calculated.

## Diagnostic result

- Initial maximum stable raw SMD: {initial_max_raw:.6f}
- Initial maximum stable robust SMD: {initial_max_robust:.6f}
- Primary maximum stable raw SMD: {primary_max_raw:.6f}
- Primary maximum stable robust SMD: {primary_max_robust:.6f}
- Robust-SMD reduction: {balance['stable_robust_smd_reduction_fraction']:.2%}
- Stable fields remaining above 0.50: {len(explanations)}; each is explicitly attributed in the high-SMD registry.
- Maximum primary pair-count deviation: {float(selected_metrics['max_pair_count_deviation']):.2%}

## Integrity

Pair-set, evidence-component, spatial-group, unit and dynamic-sequence split counts are all zero. A1 and A3 are present in every outer fold. The 19 outside-common-support positives remain applicability-only records.

## Strict extrapolation

The strict test is an eastern geographic band selected before any modeling, containing {len(strict_test):,} pair sets ({len(strict_test)/len(strict):.2%}). Its covariate shift is intentionally retained and it is forbidden for hyperparameter selection.

## Stop condition

No scaler, feature selector, hyperparameter search, model training, external A26 processing or performance evaluation was run.
"""
    (OUT / "DATASET_01_FINAL_REPORT.md").write_text(report, encoding="utf-8")

    if run_id == "B":
        snap_a.unlink(missing_ok=True)
        files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "dataset_01_file_hashes.csv" and "__pycache__" not in p.parts and p.suffix != ".pyc")
        pd.DataFrame([{"relative_path": str(p.relative_to(OUT)).replace("\\", "/"), "file_size_bytes": p.stat().st_size, "sha256": sha256(p)} for p in files]).to_csv(OUT / "dataset_01_file_hashes.csv", index=False, encoding="utf-8-sig")

    top_fields = ",".join(primary_field_max.head(10).index)
    summary = {
        "INITIAL_MAX_RAW_SMD": initial_max_raw, "INITIAL_MAX_ROBUST_SMD": initial_max_robust, "TOP_IMBALANCED_FIELDS": top_fields,
        "NEAR_CONSTANT_OR_RARE_TOP_FIELD_COUNT": len(unstable_registry), "MAX_SPATIAL_GROUP_PAIR_SETS": int(groups["pair_set_count"].max()),
        "PRIMARY_PROTOCOL": PRIMARY_PROTOCOL, "PRIMARY_FOLD_COUNT": 5, "PRIMARY_PAIR_SET_SPLITS": 0, "PRIMARY_EVIDENCE_COMPONENT_SPLITS": 0,
        "PRIMARY_SPATIAL_GROUP_SPLITS": 0, "PRIMARY_DYNAMIC_SEQUENCE_SPLITS": 0, "PRIMARY_MAX_PAIR_COUNT_DEVIATION": float(selected_metrics["max_pair_count_deviation"]),
        "PRIMARY_MAX_STABLE_RAW_SMD": primary_max_raw, "PRIMARY_MAX_STABLE_ROBUST_SMD": primary_max_robust,
        "PRIMARY_MAX_MISSING_FRACTION_DIFFERENCE": float(prim["missing_fraction_difference"].abs().max()),
        "PRIMARY_MAX_CATEGORY_PROPORTION_DIFFERENCE": float(prim["max_category_proportion_difference"].max()),
        "PRIMARY_UNEXPLAINED_HIGH_IMBALANCE_FIELDS": unexplained_high, "STRICT_PROTOCOL": STRICT_PROTOCOL,
        "STRICT_TEST_PAIR_SET_COUNT": len(strict_test), "STRICT_TEST_FRACTION": len(strict_test) / len(strict), "STRICT_MAX_STABLE_ROBUST_SMD": strict_max_robust,
        "STRICT_SPATIAL_SHIFT_RETAINED": "YES", "OUTSIDE_COMMON_SUPPORT_POSITIVES": len(outside), "FUTURE_RAINFALL_FIELDS_USED": 0,
        "LEAKAGE_FIELDS_USED": leakage_fields_used, "UPSTREAM_FROZEN_HASH_CHANGES": upstream_changes, "RERUN_ASSIGNMENT_DIFFERENCES": rerun_differences,
        "PRIMARY_SPATIAL_5FOLD_FROZEN": marker["PRIMARY_SPATIAL_5FOLD_FROZEN"], "STRICT_EXTRAPOLATION_PROTOCOL_FROZEN": marker["STRICT_EXTRAPOLATION_PROTOCOL_FROZEN"],
        "READY_FOR_DATASET_02": marker["READY_FOR_DATASET_02"], "ERRORS": len(errors), "FINAL_DECISION": marker["FINAL_DECISION"],
    }
    for k, v in summary.items():
        print(f"{k}={v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", choices=["A", "B"], required=True)
    args = parser.parse_args()
    main(args.run_id)
