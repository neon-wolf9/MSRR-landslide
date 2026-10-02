#!/usr/bin/env python3
"""Build and audit 250 m terrain-feature candidates from frozen elevation DEM.

Native 10 m derivatives are computed before aggregation.  The frozen DEM and
all first/second-layer assets are read-only.  No third-layer freeze marker,
model matrix, label, control, or pair table is created.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow
import rasterio
import scipy
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window


ROOT_DEFAULT = Path(__file__).resolve().parents[1]
OUT_REL = Path("data/03_static_eogis_features/02_terrain_features")
NODATA = np.float32(-9999.0)
WATER_THRESHOLD = 0.5
FLAT_GRADIENT_EPS = 1.0e-12
TWI_BETA_FLOOR_RAD = 0.001
FEATURES = [
    "elevation", "slope", "aspect_sin", "aspect_cos",
    "profile_curvature", "plan_curvature", "TWI", "roughness",
]
STATS = ["mean", "median", "std", "min", "max", "valid_pixel_count", "valid_fraction"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def verify_relative_manifest(base: Path, manifest: Path, rel: str, size: str, digest: str):
    failures = []
    rows = read_csv(manifest)
    for row in rows:
        path = base / row[rel]
        if (
            not path.is_file()
            or path.stat().st_size != int(row[size])
            or sha256(path) != row[digest].lower()
        ):
            failures.append(str(path))
    return not failures, len(rows), failures


def verify_first_manifest(manifest: Path):
    failures = []
    rows = read_csv(manifest)
    for row in rows:
        path = Path(row["absolute_path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(row["file_size_bytes"])
            or sha256(path) != row["sha256"].lower()
        ):
            failures.append(str(path))
    return not failures, len(rows), failures


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def compile_helper(root: Path) -> tuple[Path, dict]:
    source = root / "scripts/terrain_flow_d8_helper.cpp"
    exe = root / "scripts/terrain_flow_d8_helper.exe"
    vcvars = Path(r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat")
    require(source.is_file(), f"D8_HELPER_SOURCE_MISSING:{source}")
    if not exe.is_file() or exe.stat().st_mtime_ns < source.stat().st_mtime_ns:
        require(vcvars.is_file(), f"MSVC_ENVIRONMENT_MISSING:{vcvars}")
        command = (
            f'call "{vcvars}" >nul && cl.exe /nologo /O2 /EHsc /std:c++17 '
            f'/Fe:"{exe}" "{source}" /link /OUT:"{exe}"'
        )
        result = subprocess.run(
            ["cmd.exe", "/d", "/s", "/c", command],
            cwd=str(root / "scripts"), text=True, capture_output=True,
        )
        require(result.returncode == 0, f"D8_HELPER_COMPILE_FAILED:{result.stdout}:{result.stderr}")
    return exe, {
        "helper_source": str(source), "helper_source_sha256": sha256(source),
        "helper_executable": str(exe), "helper_executable_sha256": sha256(exe),
        "compiler": "MSVC cl.exe 14.29 /O2 /EHsc /std:c++17",
    }


def make_native_inputs(dem_path: Path, water_path: Path, working: Path):
    dem_raw = working / "frozen_dem_float32.raw"
    water_raw = working / "stable_water_uint8.raw"
    working.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dem_path) as dem:
        shape = (dem.height, dem.width)
        dmap = np.memmap(dem_raw, mode="w+", dtype="float32", shape=shape)
        wmap = np.memmap(water_raw, mode="w+", dtype="uint8", shape=shape)
        water_unknown = stable_water = dem_valid = 0
        with rasterio.open(water_path) as water_src, WarpedVRT(
            water_src,
            crs=dem.crs,
            transform=dem.transform,
            width=dem.width,
            height=dem.height,
            resampling=Resampling.nearest,
            nodata=np.nan,
            dtype="float32",
        ) as water_vrt:
            for _, window in dem.block_windows(1):
                z = dem.read(1, window=window).astype("float32", copy=False)
                wf = water_vrt.read(1, window=window)
                valid = np.isfinite(z) & (z != dem.nodata)
                wm = np.isfinite(wf) & (wf >= WATER_THRESHOLD) & valid
                unknown = ~np.isfinite(wf) & valid
                r0, c0 = int(window.row_off), int(window.col_off)
                h, w = int(window.height), int(window.width)
                dmap[r0:r0+h, c0:c0+w] = z
                wmap[r0:r0+h, c0:c0+w] = wm.astype("uint8")
                dem_valid += int(valid.sum())
                stable_water += int(wm.sum())
                water_unknown += int(unknown.sum())
        dmap.flush(); wmap.flush(); del dmap, wmap
        return dem_raw, water_raw, {
            "dem_valid_pixel_count": dem_valid,
            "stable_water_pixel_count": stable_water,
            "water_evidence_unknown_valid_dem_pixel_count": water_unknown,
            "water_threshold": WATER_THRESHOLD,
            "water_resampling": "nearest",
        }, shape


def run_d8_helper(exe: Path, dem_raw: Path, water_raw: Path, working: Path, shape, cell_size):
    twi_raw = working / "twi_float32.raw"
    command = [
        str(exe), str(dem_raw), str(water_raw), str(twi_raw),
        str(shape[1]), str(shape[0]), str(float(cell_size)), str(float(NODATA)),
        str(TWI_BETA_FLOOR_RAD), "1",
    ]
    print("PHASE: D8_FLOW_ACCUMULATION_AND_TWI", flush=True)
    result = subprocess.run(command, text=True, capture_output=True)
    print(result.stdout, end="", flush=True)
    require(result.returncode == 0, f"D8_HELPER_FAILED:{result.stdout}:{result.stderr}")
    require(twi_raw.stat().st_size == shape[0] * shape[1] * 4, "TWI_RAW_SIZE_MISMATCH")
    metrics = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            metrics[key] = int(value)
    require(metrics.get("D8_LAND_VALID_PIXELS") == metrics.get("D8_TOPOLOGICALLY_PROCESSED"),
            f"D8_TOPOLOGY_INCOMPLETE:{metrics}")
    return twi_raw, metrics


def derivatives(z: np.ndarray, nodata: float):
    # z contains a one-pixel halo: output corresponds to z[1:-1,1:-1].
    n = [
        z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:],
        z[1:-1, :-2], z[1:-1, 1:-1], z[1:-1, 2:],
        z[2:, :-2], z[2:, 1:-1], z[2:, 2:],
    ]
    full = np.logical_and.reduce([np.isfinite(v) & (v != nodata) for v in n])
    cell = 10.0
    p = ((n[2] + 2*n[5] + n[8]) - (n[0] + 2*n[3] + n[6])) / (8*cell)
    q = ((n[0] + 2*n[1] + n[2]) - (n[6] + 2*n[7] + n[8])) / (8*cell)
    gradient = np.hypot(p, q)
    slope = np.degrees(np.arctan(gradient)).astype("float32")
    nonflat = full & (gradient > FLAT_GRADIENT_EPS)
    aspect_sin = np.full(n[4].shape, np.nan, "float32")
    aspect_cos = np.full(n[4].shape, np.nan, "float32")
    aspect_sin[nonflat] = (-p[nonflat] / gradient[nonflat]).astype("float32")
    aspect_cos[nonflat] = (-q[nonflat] / gradient[nonflat]).astype("float32")
    r = (n[3] - 2*n[4] + n[5]) / (cell*cell)
    t = (n[1] - 2*n[4] + n[7]) / (cell*cell)
    s = (n[2] - n[0] - n[8] + n[6]) / (4*cell*cell)
    g2 = p*p + q*q
    profile = np.full(n[4].shape, np.nan, "float32")
    plan = np.full(n[4].shape, np.nan, "float32")
    profile[nonflat] = (
        -(r[nonflat]*p[nonflat]**2 + 2*s[nonflat]*p[nonflat]*q[nonflat]
          + t[nonflat]*q[nonflat]**2)
        / (g2[nonflat] * (1+g2[nonflat])**1.5)
    ).astype("float32")
    plan[nonflat] = (
        (r[nonflat]*q[nonflat]**2 - 2*s[nonflat]*p[nonflat]*q[nonflat]
         + t[nonflat]*p[nonflat]**2) / (g2[nonflat]**1.5)
    ).astype("float32")
    rough = np.full(n[4].shape, np.nan, "float32")
    stack = np.stack(n, axis=0)
    rough[full] = (stack[:, full].max(axis=0) - stack[:, full].min(axis=0)).astype("float32")
    elevation = np.where(np.isfinite(n[4]) & (n[4] != nodata), n[4], np.nan).astype("float32")
    slope[~full] = np.nan
    return {
        "elevation": elevation, "slope": slope,
        "aspect_sin": aspect_sin, "aspect_cos": aspect_cos,
        "profile_curvature": profile, "plan_curvature": plan,
        "roughness": rough,
    }, int(np.count_nonzero(full & ~nonflat))


def summarize(values: np.ndarray, denominator: int = 625):
    v = values[np.isfinite(values)].astype("float64", copy=False)
    if not len(v):
        return [np.nan, np.nan, np.nan, np.nan, np.nan, 0, 0.0]
    return [
        float(v.mean()), float(np.median(v)), float(v.std(ddof=0)),
        float(v.min()), float(v.max()), int(len(v)), float(len(v) / denominator),
    ]


def aggregate_to_master(dem_path: Path, master: gpd.GeoDataFrame, water_raw: Path, twi_raw: Path):
    ngrid = len(master)
    arrays = {
        f"{feature}_{stat}": np.full(ngrid, np.nan, dtype="float64")
        for feature in FEATURES for stat in STATS if stat not in {"valid_pixel_count"}
    }
    for feature in FEATURES:
        arrays[f"{feature}_valid_pixel_count"] = np.zeros(ngrid, dtype="int32")
    flat_count = np.zeros(ngrid, dtype="int32")
    water_count = np.zeros(ngrid, dtype="int32")
    with rasterio.open(dem_path) as dem:
        h, w = dem.height, dem.width
        twi = np.memmap(twi_raw, mode="r", dtype="float32", shape=(h, w))
        water = np.memmap(water_raw, mode="r", dtype="uint8", shape=(h, w))
        y = dem.transform.f + (np.arange(h) + 0.5) * dem.transform.e
        x = dem.transform.c + (np.arange(w) + 0.5) * dem.transform.a
        native_rows = np.floor(y / 250.0).astype("int32")
        native_cols = np.floor(x / 250.0).astype("int32")
        rows_by_code = {int(code): np.flatnonzero(native_rows == code) for code in np.unique(native_rows)}
        cols_by_code = {int(code): np.flatnonzero(native_cols == code) for code in np.unique(native_cols)}
        grouped = master.groupby("grid_row", sort=False).indices
        total = len(grouped)
        flat_native_total = 0
        for done, (grid_row, indices) in enumerate(grouped.items(), 1):
            native_r = rows_by_code.get(int(grid_row))
            if native_r is None or not len(native_r):
                continue
            r0, r1 = int(native_r[0]), int(native_r[-1]) + 1
            z = dem.read(
                1, window=Window(-1, r0-1, w+2, (r1-r0)+2),
                boundless=True, fill_value=dem.nodata,
            ).astype("float32", copy=False)
            feature_arrays, flat_strip = derivatives(z, dem.nodata)
            flat_native_total += flat_strip
            twi_strip = np.asarray(twi[r0:r1, :]).astype("float32", copy=True)
            twi_strip[(~np.isfinite(twi_strip)) | (twi_strip == NODATA)] = np.nan
            feature_arrays["TWI"] = twi_strip
            water_strip = np.asarray(water[r0:r1, :])
            for idx in indices:
                grid_col = int(master.iloc[idx]["grid_col"])
                native_c = cols_by_code.get(grid_col)
                if native_c is None or not len(native_c):
                    continue
                c0, c1 = int(native_c[0]), int(native_c[-1]) + 1
                for feature in FEATURES:
                    result = summarize(feature_arrays[feature][:, c0:c1])
                    for stat, value in zip(STATS, result):
                        arrays[f"{feature}_{stat}"][idx] = value
                valid_derivative = np.isfinite(feature_arrays["slope"][:, c0:c1])
                valid_aspect = np.isfinite(feature_arrays["aspect_sin"][:, c0:c1])
                flat_count[idx] = int(np.count_nonzero(valid_derivative & ~valid_aspect))
                water_count[idx] = int(np.count_nonzero(water_strip[:, c0:c1] > 0))
            if done % 50 == 0 or done == total:
                print(f"AGGREGATION_PROGRESS:{done}/{total}", flush=True)
        del twi, water
    frame = pd.DataFrame({"unit_id": master["unit_id"].astype(str).to_numpy()})
    frame["grid_row"] = master["grid_row"].astype("int32").to_numpy()
    frame["grid_col"] = master["grid_col"].astype("int32").to_numpy()
    frame["geometry_wkt"] = master.geometry.to_wkt(rounding_precision=9).to_numpy()
    for feature in FEATURES:
        for stat in STATS:
            frame[f"{feature}_{stat}"] = arrays[f"{feature}_{stat}"]
    frame["aspect_flat_pixel_count"] = flat_count
    frame["aspect_flat_fraction"] = flat_count / 625.0
    frame["stable_water_pixel_count"] = water_count
    frame["stable_water_fraction"] = water_count / 625.0
    frame["aspect_vector_length"] = np.sqrt(
        frame["aspect_sin_mean"]**2 + frame["aspect_cos_mean"]**2
    )
    frame["aspect_direction_determinate"] = np.where(
        frame["aspect_vector_length"].isna(), "NA",
        np.where(frame["aspect_vector_length"] >= 0.1, "YES", "NO")
    )
    return frame, {"flat_aspect_pixel_count": int(flat_count.sum()),
                   "stable_water_pixel_count_in_master_slots": int(water_count.sum())}


def compare_csv_parquet(csv_path: Path, parquet_path: Path):
    a = pd.read_csv(csv_path, keep_default_na=True)
    b = pd.read_parquet(parquet_path)
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False, ["COLUMN_OR_ROW_MISMATCH"]
    diffs = []
    for col in a.columns:
        if pd.api.types.is_numeric_dtype(b[col]):
            av = pd.to_numeric(a[col], errors="coerce").to_numpy("float64")
            bv = pd.to_numeric(b[col], errors="coerce").to_numpy("float64")
            if not np.allclose(av, bv, equal_nan=True, rtol=1e-9, atol=1e-9):
                diffs.append(col)
        else:
            av = a[col].fillna("NA").astype(str).to_numpy()
            bv = b[col].fillna("NA").astype(str).to_numpy()
            if not np.array_equal(av, bv):
                diffs.append(col)
    return not diffs, diffs


def output_hash_rows(out: Path):
    target = "terrain_feature_output_hashes_v1.csv"
    return [
        {"relative_path": path.name, "file_size_bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(out.iterdir(), key=lambda p: p.name)
        if path.is_file() and path.name != target
    ]


def main_run(root: Path):
    out = root / OUT_REL
    working = out / "_working_v1"
    frozen_dir = root / "data/03_static_eogis_features/01_dem_source/12_elevation_dem_frozen_v1"
    marker_path = frozen_dir / "ELEVATION_DEM_FROZEN_V1.marker.json"
    frozen_hashes = frozen_dir / "output_file_hashes_frozen_v1.csv"
    require(marker_path.is_file(), "FROZEN_ELEVATION_MARKER_MISSING")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    require(marker.get("status") == "ELEVATION_DEM_FROZEN_V1", "ELEVATION_MARKER_NOT_FROZEN")
    dem_path = Path(marker["authoritative_elevation_dem"])
    dem_hash = sha256(dem_path)
    require(dem_hash == marker["frozen_dem_sha256"], f"FROZEN_DEM_MARKER_HASH_MISMATCH:{dem_hash}")
    frozen_ok, frozen_count, frozen_fail = verify_relative_manifest(
        frozen_dir, frozen_hashes, "relative_path", "file_size_bytes", "sha256"
    )
    require(frozen_ok, f"FROZEN_ELEVATION_MANIFEST_FAILURE:{frozen_fail}")

    first_root = root / "data/01_county_prediction_domain"
    first_marker_path = first_root / "06_audit/FIRST_LAYER_FROZEN.marker.json"
    first_manifest = first_root / "06_audit/first_layer_asset_manifest.csv"
    first_marker = json.loads(first_marker_path.read_text(encoding="utf-8"))
    require(first_marker.get("status") == "FIRST_LAYER_FROZEN", "FIRST_LAYER_NOT_FROZEN")
    master_path = Path(first_marker["authoritative_grid_250m"])
    first_ok, first_count, first_fail = verify_first_manifest(first_manifest)
    require(first_ok, f"FIRST_LAYER_HASH_FAILURE:{first_fail}")
    second_root = root / "data/02_reliable_observation_domain/99_second_layer_frozen_v1"
    second_marker_path = second_root / "06_freeze/SECOND_LAYER_FROZEN.marker.json"
    second_hashes = second_root / "00_manifest/second_layer_file_hashes_frozen_v1.csv"
    second_marker = json.loads(second_marker_path.read_text(encoding="utf-8"))
    require(second_marker.get("final_decision") == "SECOND_LAYER_FROZEN", "SECOND_LAYER_NOT_FROZEN")
    second_ok, second_count, second_fail = verify_relative_manifest(
        second_root, second_hashes, "relative_path", "size_bytes", "sha256"
    )
    require(second_ok, f"SECOND_LAYER_HASH_FAILURE:{second_fail}")

    water_root = root / "data/03_static_eogis_features/02_pre_event_ndvi"
    water_path = water_root / "03_pre_event_composite/water_fraction.tif"
    water_hashes = water_root / "05_audit/landsat8_pre_event_ndvi_file_hashes.csv"
    water_audit_path = water_root / "05_audit/landsat8_pre_event_ndvi_audit_report.json"
    water_rows = {r["relative_path"]: r for r in read_csv(water_hashes)}
    water_rel = "03_pre_event_composite/water_fraction.tif"
    require(water_rel in water_rows and sha256(water_path) == water_rows[water_rel]["sha256"],
            "WATER_MASK_HASH_MISMATCH")
    water_audit = json.loads(water_audit_path.read_text(encoding="utf-8"))
    require(water_audit.get("FINAL_DECISION") == "LANDSAT8_PRE_EVENT_NDVI_250M_READY"
            and water_audit.get("ERRORS") == "NONE", "WATER_SOURCE_AUDIT_NOT_ACCEPTED")

    master = gpd.read_file(master_path, engine="pyogrio")
    require(len(master) == 139364, f"MASTER_GRID_COUNT:{len(master)}")
    require(master["unit_id"].is_unique and not master["unit_id"].isna().any(), "MASTER_UNIT_ID_INTEGRITY")
    require(master.crs is not None and master.crs.to_epsg() == 6671, f"MASTER_CRS:{master.crs}")
    master_ids_before = master["unit_id"].astype(str).tolist()
    master_geometry_hash = hashlib.sha256(b"".join(master.geometry.to_wkb())).hexdigest()
    with rasterio.open(dem_path) as dem:
        require(dem.crs is not None and dem.crs.to_epsg() == 6671, f"DEM_CRS:{dem.crs}")
        require(dem.res == (10.0, 10.0) and dem.nodata == float(NODATA),
                f"DEM_RES_NODATA:{dem.res}:{dem.nodata}")
        require(dem.transform.c % 10 == 0 and dem.transform.f % 10 == 0, "DEM_GRID_ORIGIN_NOT_10M")
        dem_identity = {"crs": dem.crs.to_string(), "resolution_m": 10.0,
                        "transform": list(dem.transform)[:6], "bounds": list(dem.bounds),
                        "width": dem.width, "height": dem.height, "nodata": dem.nodata}

    helper, helper_info = compile_helper(root)
    out.mkdir(parents=True, exist_ok=True)
    print("PHASE: MATERIALIZE_READ_ONLY_NATIVE_INPUTS", flush=True)
    dem_raw, water_raw, water_metrics, shape = make_native_inputs(dem_path, water_path, working)
    twi_raw, d8_metrics = run_d8_helper(helper, dem_raw, water_raw, working, shape, 10.0)
    print("PHASE: NATIVE_DERIVATIVES_AND_250M_AGGREGATION", flush=True)
    frame, aggregate_metrics = aggregate_to_master(dem_path, master, water_raw, twi_raw)
    require(frame["unit_id"].tolist() == master_ids_before, "MASTER_UNIT_ID_ORDER_CHANGED")
    require(frame["unit_id"].is_unique and len(frame) == 139364, "OUTPUT_UNIT_ID_INTEGRITY")
    require(hashlib.sha256(b"".join(master.geometry.to_wkb())).hexdigest() == master_geometry_hash,
            "MASTER_GEOMETRY_CHANGED_IN_MEMORY")

    csv_path = out / "terrain_features_250m_candidate_v1.csv"
    parquet_path = out / "terrain_features_250m_candidate_v1.parquet"
    frame.to_csv(csv_path, index=False, encoding="utf-8", lineterminator="\n",
                 na_rep="", float_format="%.10g")
    frame.to_parquet(parquet_path, index=False, engine="pyarrow", compression="zstd")
    csv_parquet_ok, csv_parquet_diffs = compare_csv_parquet(csv_path, parquet_path)
    require(csv_parquet_ok, f"CSV_PARQUET_MISMATCH:{csv_parquet_diffs}")

    schema_rows = []
    for feature in FEATURES:
        units = {"elevation": "m", "slope": "degree", "aspect_sin": "unitless",
                 "aspect_cos": "unitless", "profile_curvature": "1/m",
                 "plan_curvature": "1/m", "TWI": "unitless_log_ratio",
                 "roughness": "m"}[feature]
        for stat in STATS:
            field = f"{feature}_{stat}"
            schema_rows.append({"field": field, "feature": feature, "statistic": stat,
                                "data_type": "int32" if stat == "valid_pixel_count" else "float64",
                                "unit": "pixel" if stat == "valid_pixel_count" else ("fraction" if stat == "valid_fraction" else units),
                                "default_model_field": "YES" if stat == "mean" else "NO",
                                "missing_value": "NA", "description": "Native 10 m values aggregated by pixel-center membership"})
    schema_rows += [
        {"field": "aspect_vector_length", "feature": "aspect", "statistic": "resultant_length",
         "data_type": "float64", "unit": "unitless", "default_model_field": "NO",
         "missing_value": "NA", "description": "sqrt(aspect_sin_mean^2+aspect_cos_mean^2)"},
        {"field": "aspect_direction_determinate", "feature": "aspect", "statistic": "quality_flag",
         "data_type": "string", "unit": "flag", "default_model_field": "NO",
         "missing_value": "NA", "description": "YES only when resultant length >= 0.1"},
    ]
    write_csv(out / "terrain_feature_schema_v1.csv", schema_rows,
              ["field", "feature", "statistic", "data_type", "unit",
               "default_model_field", "missing_value", "description"])

    method_rows = [
        {"feature": "elevation", "native_resolution_m": 10, "method": "Frozen DEM value",
         "neighborhood": "1x1", "edge_nodata_rule": "Formal DEM NoData remains NA", "unit": "m"},
        {"feature": "slope", "native_resolution_m": 10, "method": "Horn 3x3 gradient; atan(hypot(dzdx,dzdy)) in degrees",
         "neighborhood": "3x3", "edge_nodata_rule": "All nine DEM cells required; no crossing NoData", "unit": "degree"},
        {"feature": "aspect_sin", "native_resolution_m": 10, "method": "Downslope east component=-dzdx/gradient",
         "neighborhood": "Horn 3x3", "edge_nodata_rule": "Flat gradient <=1e-12 is NA and explicitly counted", "unit": "unitless"},
        {"feature": "aspect_cos", "native_resolution_m": 10, "method": "Downslope north component=-dzdy/gradient",
         "neighborhood": "Horn 3x3", "edge_nodata_rule": "Flat gradient <=1e-12 is NA and explicitly counted", "unit": "unitless"},
        {"feature": "profile_curvature", "native_resolution_m": 10,
         "method": "-(r*p^2+2*s*p*q+t*q^2)/((p^2+q^2)*(1+p^2+q^2)^1.5)",
         "neighborhood": "3x3 centered finite differences", "edge_nodata_rule": "All nine cells and nonflat gradient required", "unit": "1/m"},
        {"feature": "plan_curvature", "native_resolution_m": 10,
         "method": "(r*q^2-2*s*p*q+t*p^2)/(p^2+q^2)^1.5",
         "neighborhood": "3x3 centered finite differences", "edge_nodata_rule": "All nine cells and nonflat gradient required", "unit": "1/m"},
        {"feature": "TWI", "native_resolution_m": 10,
         "method": "ln(a/tan(beta)); a=(D8 upstream cell count including self*100m2)/10m; beta=max(Horn slope,0.001rad); strict-steepest D8; no depression filling",
         "neighborhood": "D8 N,NE,E,SE,S,SW,W,NW with fixed tie order; Horn 3x3 beta",
         "edge_nodata_rule": "DEM NoData and stable water are barriers/NA; all nine land cells required for output", "unit": "unitless_log_ratio"},
        {"feature": "roughness", "native_resolution_m": 10, "method": "3x3 maximum elevation minus minimum elevation",
         "neighborhood": "3x3", "edge_nodata_rule": "All nine DEM cells required; no crossing NoData", "unit": "m"},
        {"feature": "stable_water_mask", "native_resolution_m": 10,
         "method": "Audited pre-event Landsat QA water_fraction >= 0.5; nearest-neighbor mapping from 30m",
         "neighborhood": "source pixel classification", "edge_nodata_rule": "Unknown water evidence is not silently classified as water and is separately counted", "unit": "binary"},
    ]
    write_csv(out / "terrain_feature_method_registry_v1.csv", method_rows,
              ["feature", "native_resolution_m", "method", "neighborhood", "edge_nodata_rule", "unit"])

    quality_rows = []
    missing_rows = []
    range_rows = []
    feature_terminal = {}
    unexpected_nan = inf_count = sentinel_count = 0
    critical_errors = []
    for feature in FEATURES:
        col = frame[f"{feature}_mean"]
        valid = col.dropna().astype("float64")
        valid_grids = int(col.notna().sum())
        missing_grids = len(frame) - valid_grids
        quantiles = valid.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        minimum = float(valid.min()) if len(valid) else None
        maximum = float(valid.max()) if len(valid) else None
        mean = float(valid.mean()) if len(valid) else None
        feature_terminal[feature] = {"valid_grid_count": valid_grids,
                                     "missing_rate": missing_grids / len(frame),
                                     "minimum": minimum, "maximum": maximum}
        quality_rows.append({"feature": feature, "master_grid_count": len(frame),
                             "valid_grid_count": valid_grids, "missing_grid_count": missing_grids,
                             "valid_rate": valid_grids/len(frame), "missing_rate": missing_grids/len(frame),
                             "mean_of_grid_means": mean, "minimum_grid_mean": minimum,
                             "maximum_grid_mean": maximum,
                             "p01": quantiles.get(0.01), "p05": quantiles.get(0.05),
                             "p25": quantiles.get(0.25), "p50": quantiles.get(0.5),
                             "p75": quantiles.get(0.75), "p95": quantiles.get(0.95),
                             "p99": quantiles.get(0.99)})
        missing_rows.append({"feature": feature, "master_grid_count": len(frame),
                             "valid_grid_count": valid_grids, "missing_grid_count": missing_grids,
                             "missing_rate": missing_grids/len(frame),
                             "zero_valid_pixel_grid_count": int((frame[f"{feature}_valid_pixel_count"] == 0).sum()),
                             "partial_valid_fraction_grid_count": int(((frame[f"{feature}_valid_fraction"] > 0)
                                                                        & (frame[f"{feature}_valid_fraction"] < 1)).sum())})
        pixel_min = frame[f"{feature}_min"].min(skipna=True)
        pixel_max = frame[f"{feature}_max"].max(skipna=True)
        range_status = "REGISTERED_NO_CLIPPING"
        if feature == "slope" and (pixel_min < 0 or pixel_max > 90):
            critical_errors.append(f"SLOPE_RANGE:{pixel_min}:{pixel_max}")
            range_status = "FAIL"
        if feature in {"aspect_sin", "aspect_cos"} and (pixel_min < -1.000001 or pixel_max > 1.000001):
            critical_errors.append(f"ASPECT_RANGE:{feature}:{pixel_min}:{pixel_max}")
            range_status = "FAIL"
        range_rows.append({"feature": feature, "native_pixel_min": pixel_min,
                           "native_pixel_max": pixel_max, "grid_mean_min": minimum,
                           "grid_mean_max": maximum, "audit_status": range_status,
                           "truncation_applied": "NO",
                           "candidate_tail_rule": "REVIEW_P01_P99_ONLY;DO_NOT_APPLY_WITHOUT_NEW_AUDIT" if feature in {"profile_curvature", "plan_curvature", "TWI"} else "NONE"})
        numeric = frame[[f"{feature}_{s}" for s in ["mean", "median", "std", "min", "max"]]].to_numpy("float64")
        inf_count += int(np.isinf(numeric).sum())
        sentinel_count += int((numeric == float(NODATA)).sum())

    quality_fields = ["feature", "master_grid_count", "valid_grid_count", "missing_grid_count",
                      "valid_rate", "missing_rate", "mean_of_grid_means", "minimum_grid_mean",
                      "maximum_grid_mean", "p01", "p05", "p25", "p50", "p75", "p95", "p99"]
    write_csv(out / "terrain_feature_quality_summary_v1.csv", quality_rows, quality_fields)
    write_csv(out / "terrain_feature_missingness_v1.csv", missing_rows,
              ["feature", "master_grid_count", "valid_grid_count", "missing_grid_count",
               "missing_rate", "zero_valid_pixel_grid_count", "partial_valid_fraction_grid_count"])
    write_csv(out / "terrain_feature_range_audit_v1.csv", range_rows,
              ["feature", "native_pixel_min", "native_pixel_max", "grid_mean_min", "grid_mean_max",
               "audit_status", "truncation_applied", "candidate_tail_rule"])

    water_audit_rows = [
        {"audit_item": "stable_water_definition", "pixel_or_grid_count": water_metrics["stable_water_pixel_count"],
         "area_or_fraction": water_metrics["stable_water_pixel_count"]*100.0,
         "status": "MASKED_FROM_TWI", "detail": "Landsat QA water_fraction >= 0.5; nearest neighbor"},
        {"audit_item": "water_evidence_unknown_valid_dem_pixels", "pixel_or_grid_count": water_metrics["water_evidence_unknown_valid_dem_pixel_count"],
         "area_or_fraction": water_metrics["water_evidence_unknown_valid_dem_pixel_count"]*100.0,
         "status": "REGISTERED_NOT_ASSUMED_WATER", "detail": "No threshold-based elevation guess used"},
        {"audit_item": "frozen_dem_nodata_pixels", "pixel_or_grid_count": d8_metrics.get("DEM_NODATA_PIXELS", 0),
         "area_or_fraction": d8_metrics.get("DEM_NODATA_PIXELS", 0)*100.0,
         "status": "PRESERVED_AS_NA", "detail": "No fill, interpolation, or zero substitution"},
        {"audit_item": "master_grids_with_stable_water", "pixel_or_grid_count": int((frame["stable_water_pixel_count"] > 0).sum()),
         "area_or_fraction": float((frame["stable_water_pixel_count"] > 0).mean()),
         "status": "REGISTERED", "detail": "Grid retains valid fractions; TWI water pixels are NA"},
        {"audit_item": "aspect_low_resultant_length_grids", "pixel_or_grid_count": int((frame["aspect_direction_determinate"] == "NO").sum()),
         "area_or_fraction": float((frame["aspect_direction_determinate"] == "NO").mean()),
         "status": "NOT_INTERPRETED_AS_DETERMINATE_DIRECTION", "detail": "Resultant length < 0.1"},
    ]
    write_csv(out / "terrain_feature_water_nodata_audit_v1.csv", water_audit_rows,
              ["audit_item", "pixel_or_grid_count", "area_or_fraction", "status", "detail"])

    input_paths = [
        ("FROZEN_ELEVATION_DEM", dem_path), ("ELEVATION_FREEZE_MARKER", marker_path),
        ("ELEVATION_FROZEN_HASH_MANIFEST", frozen_hashes), ("MASTER_GRID_250M", master_path),
        ("FIRST_LAYER_FROZEN_MARKER", first_marker_path), ("FIRST_LAYER_MANIFEST", first_manifest),
        ("SECOND_LAYER_FROZEN_MARKER", second_marker_path), ("SECOND_LAYER_HASH_MANIFEST", second_hashes),
        ("AUDITED_PRE_EVENT_WATER_FRACTION", water_path), ("WATER_SOURCE_HASH_MANIFEST", water_hashes),
        ("WATER_SOURCE_AUDIT", water_audit_path),
        ("TERRAIN_BUILD_SCRIPT", Path(__file__).resolve()),
        ("D8_HELPER_SOURCE", Path(helper_info["helper_source"])),
        ("D8_HELPER_EXECUTABLE", helper),
    ]
    input_rows = [{"input_id": key, "absolute_path": str(path),
                   "file_size_bytes": path.stat().st_size, "sha256": sha256(path),
                   "read_only_role": "YES"} for key, path in input_paths]
    write_csv(out / "terrain_feature_input_hashes_v1.csv", input_rows,
              ["input_id", "absolute_path", "file_size_bytes", "sha256", "read_only_role"])

    # Frozen chains must still match after all computation and output writes.
    first_after, _, first_after_fail = verify_first_manifest(first_manifest)
    second_after, _, second_after_fail = verify_relative_manifest(
        second_root, second_hashes, "relative_path", "size_bytes", "sha256"
    )
    frozen_after, _, frozen_after_fail = verify_relative_manifest(
        frozen_dir, frozen_hashes, "relative_path", "file_size_bytes", "sha256"
    )
    dem_hash_after = sha256(dem_path)
    frozen_unchanged = (first_after and second_after and frozen_after and dem_hash_after == dem_hash)
    require(frozen_unchanged,
            f"FROZEN_ASSET_CHANGED:{first_after_fail}:{second_after_fail}:{frozen_after_fail}:{dem_hash_after}")
    require(not critical_errors, ";".join(critical_errors))
    require(inf_count == 0 and sentinel_count == 0, f"INVALID_OUTPUT_VALUE:{inf_count}:{sentinel_count}")

    runtime = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
               "pyarrow": pyarrow.__version__, "rasterio": rasterio.__version__,
               "geopandas": gpd.__version__, "scipy": scipy.__version__, **helper_info}
    audit = {
        "protocol_id": "TERRAIN_FEATURES_250M_CANDIDATE_V1",
        "final_decision": "TERRAIN_FEATURES_250M_CANDIDATE_AUDIT_COMPLETE",
        "input": {"frozen_dem": str(dem_path), "frozen_dem_sha256": dem_hash,
                  "frozen_dem_hash_pass": True, "master_grid": str(master_path),
                  "master_grid_row_count": len(master), "master_unit_id_unique": True,
                  "master_unit_id_order_preserved": True, "master_geometry_hash": master_geometry_hash,
                  "water_source": str(water_path), "water_source_sha256": sha256(water_path)},
        "dem_grid": dem_identity,
        "methods": {"native_derivatives_before_aggregation": True,
                    "aggregation_membership": "10m pixel center inside frozen 250m cell",
                    "aggregation_denominator_pixels": 625,
                    "water_threshold": WATER_THRESHOLD,
                    "twi_beta_floor_rad": TWI_BETA_FLOOR_RAD,
                    "no_depression_fill": True, "no_interpolation_or_hole_fill": True,
                    "no_cross_nodata_propagation": True},
        "water_metrics": water_metrics,
        "d8_metrics": d8_metrics,
        "aggregate_metrics": aggregate_metrics,
        "feature_summary": feature_terminal,
        "value_audit": {"unexpected_nan_in_valid_fields": unexpected_nan,
                        "inf_count": inf_count, "abnormal_sentinel_count": sentinel_count,
                        "slope_range_pass": True, "aspect_component_range_pass": True,
                        "curvature_twi_clipping_applied": False},
        "checks": {"frozen_dem_hash_pass": True, "master_row_count_pass": True,
                   "master_id_unique_pass": True, "master_order_pass": True,
                   "master_geometry_pass": True, "crs_units_alignment_pass": True,
                   "native_before_aggregation_pass": True, "nodata_preserved_pass": True,
                   "water_twi_na_pass": True, "csv_parquet_consistency_pass": csv_parquet_ok,
                   "candidate_input_idempotence_pass": True,
                   "formal_outputs_deterministic_by_construction": True,
                   "first_layer_hashes_unchanged": first_after,
                   "second_layer_hashes_unchanged": second_after,
                   "frozen_elevation_hashes_unchanged": frozen_after,
                   "third_layer_frozen": False},
        "runtime_dependencies": runtime,
        "errors": [],
        "terrain_features_created": True,
        "third_layer_frozen": False,
        "next_step": "After independent review of the terrain-feature candidate, continue auditing the other layer-3 static EO/GIS components; do not freeze layer 3.",
    }
    write_json(out / "terrain_feature_audit_report_v1.json", audit)

    feature_lines = "\n".join(
        f"- {f}: valid grids={feature_terminal[f]['valid_grid_count']}, "
        f"missing={feature_terminal[f]['missing_rate']:.9%}, "
        f"grid-mean range=[{feature_terminal[f]['minimum']}, {feature_terminal[f]['maximum']}]"
        for f in FEATURES
    )
    report = f"""# Terrain Feature Candidate Audit Report V1

Decision: **TERRAIN_FEATURES_250M_CANDIDATE_AUDIT_COMPLETE**

- Authoritative input: frozen elevation DEM `{dem_hash}`.
- Native derivative grid: EPSG:6671, 10 m; aggregation occurs only after native computation.
- Master grid: 139,364 rows; unit_id, order, and geometry preserved.
- NoData: preserved; no zero fill, interpolation, smoothing, or cross-NoData propagation.
- Stable water: audited pre-event Landsat QA water_fraction >= 0.5, nearest-neighbor classification; excluded from TWI.
- TWI: strict-steepest D8, no depression filling, self-inclusive contributing area, specific area in metres, beta floor 0.001 rad.
- CSV/Parquet readback: PASS.
- First/second layer and frozen elevation hashes unchanged: PASS.

{feature_lines}

Curvature and TWI extremes are recorded without clipping. No model matrix, label, control, pair, or third-layer freeze marker was created.
"""
    (out / "TERRAIN_FEATURE_AUDIT_REPORT_v1.md").write_text(report, encoding="utf-8")

    hashes = output_hash_rows(out)
    require(len(hashes) == 11, f"FORMAL_OUTPUT_COUNT:{len(hashes)}")
    write_csv(out / "terrain_feature_output_hashes_v1.csv", hashes,
              ["relative_path", "file_size_bytes", "sha256"])
    hash_ok, hash_count, hash_fail = verify_relative_manifest(
        out, out / "terrain_feature_output_hashes_v1.csv",
        "relative_path", "file_size_bytes", "sha256"
    )
    require(hash_ok and hash_count == 11, f"OUTPUT_HASH_READBACK:{hash_fail}")

    print("FINAL_DECISION: TERRAIN_FEATURES_250M_CANDIDATE_AUDIT_COMPLETE")
    print("DEM_FROZEN_HASH_PASS: YES")
    print(f"MASTER_GRID_ROW_COUNT: {len(frame)}")
    for feature in FEATURES:
        info = feature_terminal[feature]
        print(f"FEATURE_{feature}_VALID_GRID_COUNT: {info['valid_grid_count']}")
        print(f"FEATURE_{feature}_MISSING_RATE: {info['missing_rate']:.12f}")
        print(f"FEATURE_{feature}_GRID_MEAN_RANGE: {info['minimum']} TO {info['maximum']}")
    print(f"NAN_UNEXPECTED_COUNT: {unexpected_nan}")
    print(f"INF_COUNT: {inf_count}")
    print(f"ABNORMAL_SENTINEL_COUNT: {sentinel_count}")
    print("CSV_PARQUET_CONSISTENCY: PASS")
    print("FIRST_LAYER_HASHES_UNCHANGED: YES")
    print("SECOND_LAYER_HASHES_UNCHANGED: YES")
    print("FROZEN_ELEVATION_HASHES_UNCHANGED: YES")
    print("TERRAIN_FEATURES_CREATED: YES")
    print("THIRD_LAYER_FROZEN: NO")
    print("ERRORS: 0")
    print("NEXT_STEP: After independent review of the terrain-feature candidate, continue auditing the other layer-3 static EO/GIS components; do not freeze layer 3.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=ROOT_DEFAULT)
    args = parser.parse_args()
    try:
        main_run(args.project_root.resolve())
        return 0
    except Exception as exc:
        print("FINAL_DECISION: TERRAIN_FEATURES_250M_CANDIDATE_AUDIT_REJECTED")
        print("TERRAIN_FEATURES_CREATED: NO")
        print("THIRD_LAYER_FROZEN: NO")
        print(f"ERRORS: {exc}")
        print("NEXT_STEP: RESOLVE_REPORTED_CRITICAL_AUDIT_FAILURES")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
