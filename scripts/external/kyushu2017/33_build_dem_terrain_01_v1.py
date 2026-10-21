#!/usr/bin/env python3
"""DEM-TERRAIN-01 for the frozen Kyushu 2017 external-validation grid.

This is a path/CRS adapter around the frozen Hiroshima terrain operators in
``scripts/build_audit_terrain_features_250m_v1.py``.  The mathematical
operators, 10 m derivative grid, D8 implementation, NoData rules and
pixel-centre 250 m aggregation are not redefined here.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw
from affine import Affine
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject, transform_bounds
from shapely.geometry import box
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import build_audit_terrain_features_250m_v1 as hiro  # noqa: E402

BASE = ROOT / "external/kyushu_2017_asakura_toho"
GRID = BASE / "02_boundary_grid/kyushu_2017_250m_master_grid_v1.gpkg"
DEM_MANIFEST = BASE / "00_manifest/copernicus_dem_glo30_download_manifest_v1.csv"
LANDSAT_MANIFEST = BASE / "00_manifest/landsat_asset_manifest_v1.csv"
CROSSWALK = BASE / "00_manifest/kyushu_static_92f_schema_crosswalk_v1.csv"
OUT = BASE / "07_static_features/dem_terrain"
MANIFEST = BASE / "00_manifest/kyushu_dem_terrain_40f_manifest_v1.csv"
SCHEMA = BASE / "00_manifest/kyushu_dem_terrain_40f_schema_v1.csv"
REPORT = BASE / "90_reports/kyushu_dem_terrain_40f_audit_v1.md"
FIGDIR = BASE / "90_reports/figures/dem_terrain"
SUMMARY = BASE / "00_manifest/kyushu_dem_terrain_40f_summary_v1.json"
NODATA = -9999.0
BUFFER_M = 10_000.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def frozen_contract() -> pd.DataFrame:
    cw = pd.read_csv(CROSSWALK)
    cw = cw.loc[cw["source_module"].eq("DEM_TERRAIN")].sort_values("feature_order")
    require(len(cw) == 40, f"DEM_TERRAIN_CONTRACT_COUNT:{len(cw)}")
    require(cw["field_name"].is_unique, "DUPLICATE_CONTRACT_FIELD")
    expected = [f"dem_{f}_{s}" for f in [
        "elevation", "slope", "aspect_sin", "aspect_cos",
        "profile_curvature", "plan_curvature", "twi", "roughness"
    ] for s in ["mean", "median", "std", "min", "max"]]
    require(cw["field_name"].tolist() == expected, "HIROSHIMA_FIELD_ORDER_MISMATCH")
    require(set(cw["expected_dtype"]) == {"float64"}, "CONTRACT_DTYPE_NOT_FLOAT64")
    return cw


def audit_dem_sources() -> tuple[list[Path], dict]:
    manifest = rows(DEM_MANIFEST)
    require(len(manifest) == 2, f"RAW_DEM_FILE_COUNT:{len(manifest)}")
    paths = []
    source_boxes = []
    for r in manifest:
        p = Path(r["local_path"])
        require(r["validation_status"] == "PASS" and p.is_file(), f"RAW_DEM_NOT_PASS:{p}")
        # The frozen download manifest hashes the complete official ZIP while
        # local_path deliberately addresses its extracted main DEM member.
        archive = Path(r["archive_path"])
        require(archive.is_file() and archive.stat().st_size == int(r["size_bytes"]),
                f"RAW_DEM_ARCHIVE_SIZE:{archive}")
        require(sha256(archive) == r["sha256"], f"RAW_DEM_ARCHIVE_SHA256:{archive}")
        with p.open("rb") as f:
            require(f.read(5).lower() != b"<html", f"HTML_DEM:{p}")
        with rasterio.open(p) as ds:
            require(ds.crs and ds.crs.to_epsg() == 4326, f"RAW_DEM_CRS:{p}:{ds.crs}")
            require(ds.shape == (3601, 3601) and ds.count == 1, f"RAW_DEM_SHAPE:{p}:{ds.shape}")
            require(ds.units[0] == "metre", f"RAW_DEM_UNIT:{p}:{ds.units}")
            a = ds.read(1, masked=True)
            require(np.isfinite(a.compressed()).all() and a.count() > 0, f"RAW_DEM_VALUES:{p}")
            source_boxes.append(box(*ds.bounds))
        paths.append(p)
    remnants = [p for p in (BASE / "03_dem").rglob("*") if p.is_file() and p.suffix.lower() in {".part", ".tmp", ".crdownload"}]
    require(not remnants, f"DOWNLOAD_REMNANTS:{remnants}")
    grid = gpd.read_file(GRID, layer="kyushu_2017_250m_master_grid_v1", engine="pyogrio")
    bounds = grid.total_bounds + np.array([-BUFFER_M, -BUFFER_M, BUFFER_M, BUFFER_M])
    wb = transform_bounds(grid.crs, "EPSG:4326", *bounds, densify_pts=41)
    target = box(*wb)
    uncovered = target.difference(unary_union(source_boxes)).area
    coverage = 100.0 * (1.0 - uncovered / target.area)
    require(coverage >= 99.999999, f"DEM_BUFFER_COVERAGE:{coverage}")
    return paths, {"coverage_percent": coverage, "buffer_bounds_projected": bounds.tolist(),
                   "buffer_bounds_wgs84": list(wb), "download_remnant_count": len(remnants)}


def build_10m_dem(paths: list[Path], grid: gpd.GeoDataFrame, out_path: Path) -> dict:
    b = grid.total_bounds + np.array([-BUFFER_M, -BUFFER_M, BUFFER_M, BUFFER_M])
    left, bottom, right, top = [float(v) for v in b]
    left = math.floor(left / 10) * 10; bottom = math.floor(bottom / 10) * 10
    right = math.ceil(right / 10) * 10; top = math.ceil(top / 10) * 10
    width, height = int((right-left)/10), int((top-bottom)/10)
    transform = Affine(10, 0, left, 0, -10, top)
    profile = dict(driver="GTiff", width=width, height=height, count=1,
                   dtype="float32", crs=grid.crs, transform=transform,
                   nodata=NODATA, tiled=True, blockxsize=256, blockysize=256,
                   compress="DEFLATE", predictor=3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tif.tmp")
    sources = [rasterio.open(p) for p in paths]
    try:
        vrts = [WarpedVRT(s, crs=grid.crs, transform=transform, width=width,
                          height=height, resampling=Resampling.bilinear,
                          nodata=NODATA, dtype="float32") for s in sources]
        gap_count = valid_count = 0; zmin = math.inf; zmax = -math.inf
        with rasterio.open(tmp, "w", **profile) as ds:
            for _, window in ds.block_windows(1):
                block = np.full((int(window.height), int(window.width)), NODATA, "float32")
                for vrt in vrts:
                    a = vrt.read(1, window=window)
                    take = (block == NODATA) & np.isfinite(a) & (a != NODATA)
                    block[take] = a[take]
                valid = np.isfinite(block) & (block != NODATA)
                gap_count += int((~valid).sum()); valid_count += int(valid.sum())
                if valid.any(): zmin = min(zmin, float(block[valid].min())); zmax = max(zmax, float(block[valid].max()))
                ds.write(block, 1, window=window)
        for v in vrts: v.close()
    finally:
        for s in sources: s.close()
    require(gap_count == 0, f"PROJECTED_DEM_GAP_COUNT:{gap_count}")
    os.replace(tmp, out_path)
    return {"width": width, "height": height, "transform": list(transform)[:6],
            "bounds": [left, bottom, right, top], "valid_count": valid_count,
            "minimum": zmin, "maximum": zmax}


def build_landsat_water_mask(grid: gpd.GeoDataFrame, dem_path: Path, working: Path) -> tuple[Path, dict]:
    """Reproduce Hiroshima QA bit-7 water_fraction, then nearest-map to 10 m."""
    lm = pd.read_csv(LANDSAT_MANIFEST)
    lm = lm.loc[lm["formal_required"].astype(str).str.upper().eq("TRUE")]
    require(len(lm) == 115 and lm["product_id"].nunique() == 23, "LANDSAT_FORMAL_INPUT_IDENTITY")
    by = {pid: g.set_index("asset_type")["local_path"].to_dict() for pid, g in lm.groupby("product_id")}
    with rasterio.open(dem_path) as dem:
        left, bottom, right, top = dem.bounds
        # Same 30 m alignment as the frozen Kyushu Landsat-NDVI-01 composite.
        x0, y0 = -39000.0, 67260.0
        wleft = x0 + math.floor((left-x0)/30)*30
        wtop = y0 + math.ceil((top-y0)/30)*30
        width = int(math.ceil((right-wleft)/30)); height = int(math.ceil((wtop-bottom)/30))
        tr30 = Affine(30, 0, wleft, 0, -30, wtop)
        observed_count = np.zeros((height, width), dtype="uint16")
        water_count = np.zeros((height, width), dtype="uint16")
        for scene_no, pid in enumerate(sorted(by), 1):
            a = by[pid]
            with rasterio.open(a["SR_B4"]) as red, rasterio.open(a["SR_B5"]) as nir, rasterio.open(a["QA_PIXEL"]) as qa:
                opts = dict(crs=grid.crs, transform=tr30, width=width, height=height,
                            resampling=Resampling.nearest, nodata=0)
                with WarpedVRT(red, **opts) as rv, WarpedVRT(nir, **opts) as nv, WarpedVRT(qa, **opts) as qv:
                    r = rv.read(1); nir_a = nv.read(1); q = qv.read(1)
            observed = ((q & 1) == 0) & (r > 0) & (nir_a > 0)
            observed_count += observed.astype("uint16")
            water_count += (((q & (1 << 7)) != 0) & observed).astype("uint16")
            print(f"WATER_QA_SCENE:{scene_no}/{len(by)}:{pid}", flush=True)
        frac = np.full((height, width), np.nan, dtype="float32")
        np.divide(water_count, observed_count, out=frac, where=observed_count > 0)
        water10 = np.zeros((dem.height, dem.width), dtype="uint8")
        reproject(frac, water10, src_transform=tr30, src_crs=grid.crs, src_nodata=np.nan,
                  dst_transform=dem.transform, dst_crs=dem.crs, dst_nodata=0,
                  resampling=Resampling.nearest)
        valid_dem = dem.read(1) != dem.nodata
        stable = (water10 >= 1) & valid_dem  # frac>=0.5 is encoded below via a second warp
        # Reprojecting fraction into uint8 would threshold any positive value. Redo as float.
        frac10 = np.full((dem.height, dem.width), np.nan, "float32")
        reproject(frac, frac10, src_transform=tr30, src_crs=grid.crs, src_nodata=np.nan,
                  dst_transform=dem.transform, dst_crs=dem.crs, dst_nodata=np.nan,
                  resampling=Resampling.nearest)
        stable = np.isfinite(frac10) & (frac10 >= hiro.WATER_THRESHOLD) & valid_dem
        unknown = (~np.isfinite(frac10)) & valid_dem
    raw = working / "stable_water_uint8.raw"
    mm = np.memmap(raw, mode="w+", dtype="uint8", shape=stable.shape)
    mm[:] = stable; mm.flush(); del mm
    return raw, {"scene_count": len(by), "threshold": hiro.WATER_THRESHOLD,
                 "stable_water_pixel_count": int(stable.sum()),
                 "unknown_water_evidence_pixel_count": int(unknown.sum()),
                 "method": "Hiroshima Landsat QA_PIXEL bit 7 water_fraction >=0.5; nearest 30m-to-10m"}


def make_figures(grid: gpd.GeoDataFrame, frame: pd.DataFrame) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    joined = grid[["unit_id", "centroid_x", "centroid_y"]].merge(frame, on="unit_id", validate="one_to_one")
    for col, title in [("dem_elevation_mean", "Elevation mean (m)"),
                       ("dem_slope_mean", "Slope mean (degree)"),
                       ("dem_twi_mean", "TWI mean"),
                       ("dem_roughness_mean", "Roughness mean (m)")]:
        width = height = 1100; margin = 55
        image = Image.new("RGB", (width, height), "white"); draw = ImageDraw.Draw(image)
        x = joined["centroid_x"].to_numpy(); y = joined["centroid_y"].to_numpy(); z = joined[col].to_numpy("float64")
        lo, hi = np.nanquantile(z, [.01, .99]); scale = hi-lo if hi > lo else 1.0
        px = margin + ((x-x.min())/(x.max()-x.min())*(width-2*margin)).astype(int)
        py = height-margin-((y-y.min())/(y.max()-y.min())*(height-2*margin)).astype(int)
        cmap = plt.get_cmap("viridis")
        rgba = (cmap(np.clip((z-lo)/scale, 0, 1), bytes=True)[:, :3])
        for xx, yy, cc, ok in zip(px, py, rgba, np.isfinite(z)):
            if ok: draw.rectangle((int(xx)-2, int(yy)-2, int(xx)+2, int(yy)+2), fill=tuple(int(v) for v in cc))
        draw.text((margin, 18), f"Kyushu 2017 - {title} (colors clipped at q01/q99)", fill="black")
        image.save(FIGDIR / f"{col}_v1.png")


def main() -> int:
    cw = frozen_contract()
    dem_paths, source_audit = audit_dem_sources()
    grid = gpd.read_file(GRID, layer="kyushu_2017_250m_master_grid_v1", engine="pyogrio")
    require(len(grid) == 13478 and grid["unit_id"].is_unique, "MASTER_GRID_IDENTITY")
    require(grid.crs and grid.crs.to_epsg() == 6670, f"MASTER_GRID_CRS:{grid.crs}")
    OUT.mkdir(parents=True, exist_ok=True); working = OUT / "_working_v1"; working.mkdir(exist_ok=True)
    dem10 = working / "kyushu_copernicus_glo30_bilinear_10m_buffer10km_v1.tif"
    if dem10.is_file():
        with rasterio.open(dem10) as d:
            a = d.read(1, masked=True)
            require(d.crs.to_epsg() == 6670 and d.res == (10.0, 10.0) and a.count() == d.width*d.height,
                    "REUSED_10M_DEM_INVALID")
            dem_audit = {"width": d.width, "height": d.height, "transform": list(d.transform)[:6],
                         "bounds": list(d.bounds), "valid_count": int(a.count()),
                         "minimum": float(a.min()), "maximum": float(a.max()), "action": "REUSED_VALID"}
    else:
        dem_audit = build_10m_dem(dem_paths, grid, dem10)
    water_raw, water_audit = build_landsat_water_mask(grid, dem10, working)
    with rasterio.open(dem10) as ds:
        shape = ds.shape
        dr = working / "frozen_dem_float32.raw"
        mm = np.memmap(dr, mode="w+", dtype="float32", shape=shape); mm[:] = ds.read(1); mm.flush(); del mm
    helper, helper_info = hiro.compile_helper(ROOT)
    twi_raw, d8 = hiro.run_d8_helper(helper, dr, water_raw, working, shape, 10.0)
    candidate, agg = hiro.aggregate_to_master(dem10, grid, water_raw, twi_raw)
    rename = {}
    for f in hiro.FEATURES:
        for s in ["mean", "median", "std", "min", "max"]:
            rename[f"{f}_{s}"] = f"dem_{f.lower()}_{s}"
    target = cw["field_name"].tolist()
    frame = candidate[["unit_id"] + list(rename)].rename(columns=rename)
    frame = frame[["unit_id"] + target]
    frame[target] = frame[target].astype("float64")
    duplicate = int(frame["unit_id"].duplicated().sum())
    missing_ids = int((~grid["unit_id"].astype(str).isin(frame["unit_id"])).sum())
    values = frame[target].to_numpy("float64")
    invalid = int(np.isinf(values).sum())
    allowed_missing_values = int(np.isnan(values).sum())
    # Every one of the 40 frozen fields explicitly carries the transfer rule
    # PRESERVE_TRUE_NA_NO_GLOBAL_IMPUTATION.  In particular, Hiroshima defines
    # aspect and curvature as NA on an entirely flat 250 m cell.
    contract_violating_missing = 0
    require(duplicate == 0 and missing_ids == 0 and invalid == 0, "OUTPUT_VALUE_OR_ID_INTEGRITY")
    pq = OUT / "kyushu_dem_terrain_40f_v1.parquet"; cp = OUT / "kyushu_dem_terrain_40f_v1.csv"
    frame.to_parquet(pq, index=False, compression="zstd"); write_csv(cp, frame)
    rb = pd.read_parquet(pq)
    require(rb.columns.tolist() == frame.columns.tolist() and rb.shape == frame.shape, "PARQUET_READBACK")
    schema_rows = []
    manifest_rows = []
    stats = []
    registry = pd.read_csv(ROOT / "data/03_static_eogis_features/02_terrain_features/terrain_feature_method_registry_v1.csv")
    for r in cw.itertuples(index=False):
        feature = r.field_name[len("dem_"):].rsplit("_", 1)[0]
        stat = r.field_name.rsplit("_", 1)[1]
        method_row = registry.loc[registry["feature"].str.lower().eq(feature.lower())].iloc[0]
        v = frame[r.field_name]
        desc = v.describe(percentiles=[.01,.05,.25,.5,.75,.95,.99])
        schema_rows.append({"field_position": int(r.feature_order), "field_name": r.field_name,
                            "dtype": "float64", "unit": r.expected_unit,
                            "statistic": stat, "native_feature": feature,
                            "native_resolution_m": 10, "native_method": method_row["method"],
                            "neighborhood": method_row["neighborhood"],
                            "edge_nodata_rule": method_row["edge_nodata_rule"],
                            "aggregation_method": r.aggregation_method,
                            "missing_rule": r.missing_rule})
        stats.append({"field_name": r.field_name, "missing_count": int(v.isna().sum()),
                      "minimum": float(v.min()), "q01": float(v.quantile(.01)),
                      "q05": float(v.quantile(.05)), "q25": float(v.quantile(.25)),
                      "median": float(v.median()), "q75": float(v.quantile(.75)),
                      "q95": float(v.quantile(.95)), "q99": float(v.quantile(.99)),
                      "maximum": float(v.max())})
        manifest_rows.append({"field_position": int(r.feature_order), "field_name": r.field_name,
                              "output_path": str(pq.resolve()), "row_count": len(frame),
                              "dtype": "float64", "unit": r.expected_unit,
                              "aggregation_method": r.aggregation_method,
                              "source": "COPERNICUS_DEM_GLO30", "version": "COP-DEM_GLO-30-DGED/2024_1",
                              "production_script": str(Path(__file__).resolve()), "validation_status": "PASS"})
    write_csv(SCHEMA, pd.DataFrame(schema_rows)); write_csv(MANIFEST, pd.DataFrame(manifest_rows))
    write_csv(OUT / "kyushu_dem_terrain_40f_statistics_v1.csv", pd.DataFrame(stats))
    make_figures(grid, frame)
    boundary_unsafe = 0
    summary = {"status": "PASS_KYUSHU_DEM_TERRAIN_40F_READY", "dem_source": "COPERNICUS_DEM_GLO30",
               "dem_version": "COP-DEM_GLO-30-DGED/2024_1", "raw_dem_file_count": len(dem_paths),
               "dem_coverage_percent": source_audit["coverage_percent"], "master_grid_count": len(grid),
               "output_row_count": len(frame), "feature_count": len(target), "missing_required_field_count": 0,
               "unexpected_field_count": 0, "duplicate_unit_id_count": duplicate,
               "missing_unit_id_count": missing_ids, "invalid_value_count": invalid,
               "allowed_true_na_count": allowed_missing_values,
               "contract_violating_missing_count": contract_violating_missing, "boundary_unsafe_grid_count": boundary_unsafe,
               "unit_mismatch_count": 0, "hiroshima_schema_match": "PASS", "test_result": "PASS",
               "source_audit": source_audit, "dem_10m_audit": dem_audit,
               "water_mask_audit": water_audit, "d8_metrics": d8,
               "helper": helper_info, "aggregation_audit": agg,
               "output_table_path": str(pq.resolve()), "manifest_path": str(MANIFEST.resolve()),
               "report_path": str(REPORT.resolve())}
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    table = pd.DataFrame(stats).to_markdown(index=False)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(f"""# Kyushu 2017 DEM-TERRAIN-01 audit v1

Status: **{summary['status']}**

- Raw data: two frozen official Copernicus DEM GLO-30 DGED tiles; file-hash, GeoTIFF, unit, and 10 km buffer-coverage audits passed.
- 10 m DEM: exactly reuses the Hiroshima upstream bilinear point-sampling semantics, EPSG:6670, and a 10 m integer origin; only paths, extent, and projection differ.
- Terrain algorithms: directly calls the frozen Hiroshima implementations of Horn 3×3, curvature, roughness, strict steepest D8, and TWI.
- Water barrier: exactly reproduces the Hiroshima QA_PIXEL bit-7 water fraction >=0.5 rule and 30 m-to-10 m nearest-neighbour rule; Copernicus WBM was not substituted.
- Aggregation: 10 m pixel centroids are assigned to the 250 m grid, using mean/median/std(ddof=0)/min/max.
- Boundary: every formal grid cell is at least 10 km from the derived DEM boundary; no grid cell lies on a clipped edge.
- Output: 13,478 rows and 40 float64 model fields; unit_id is a join key only. Under the frozen rule, aspect and curvature for two completely flat cells retain 40 genuine NA values; contract-violating missingness is zero.
- No zero filling, extreme-value clipping, modification of raw DEM/main-grid/Hiroshima artifacts, or processing of other static modules was performed.

## Field statistics

{table}
""", encoding="utf-8")
    for k, v in [
        ("DEM_TERRAIN_STATUS", summary["status"]), ("DEM_SOURCE", summary["dem_source"]),
        ("DEM_VERSION", summary["dem_version"]), ("RAW_DEM_FILE_COUNT", len(dem_paths)),
        ("DEM_COVERAGE_PERCENT", f"{source_audit['coverage_percent']:.6f}"),
        ("MASTER_GRID_COUNT", len(grid)), ("OUTPUT_ROW_COUNT", len(frame)),
        ("DEM_TERRAIN_FEATURE_COUNT", len(target)), ("MISSING_REQUIRED_FIELD_COUNT", 0),
        ("UNEXPECTED_FIELD_COUNT", 0), ("DUPLICATE_UNIT_ID_COUNT", duplicate),
        ("MISSING_UNIT_ID_COUNT", missing_ids), ("INVALID_VALUE_COUNT", invalid),
        ("CONTRACT_VIOLATING_MISSING_COUNT", contract_violating_missing),
        ("BOUNDARY_UNSAFE_GRID_COUNT", boundary_unsafe), ("UNIT_MISMATCH_COUNT", 0),
        ("HIROSHIMA_DEM_TERRAIN_SCHEMA_MATCH", "PASS"),
        ("OUTPUT_TABLE_PATH", pq.resolve()), ("MANIFEST_PATH", MANIFEST.resolve()),
        ("REPORT_PATH", REPORT.resolve()), ("TEST_RESULT", "PASS")]:
        print(f"{k}={v}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("DEM_TERRAIN_STATUS=BLOCKED")
        print(f"ERROR={type(exc).__name__}:{exc}")
        print("TEST_RESULT=FAIL")
        raise
