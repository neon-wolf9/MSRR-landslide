#!/usr/bin/env python
"""Verify package metadata and report the restricted inputs needed for full reruns."""
from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PACKAGE_REQUIRED = [
    "FILE_MANIFEST.csv",
    "RESULT_MANIFEST.csv",
    "data/manifests/99_FINAL_DATASET_DECISION.json",
    "data/schemas/06A_STATIC_92_FIELD_SCHEMA.csv",
    "data/schemas/07B_DYNAMIC_10_FIELD_ORDER.csv",
    "configs/matching/MATCHING_VARIABLE_DICTIONARY.csv",
    "configs/external/kyushu_external_model_feature_whitelist_v1.csv",
]

FULL_RERUN_REQUIRED = [
    "data/07_FINAL_REPAIRED_DATASET_V2/04_FINAL_SAMPLE_INDEX.parquet",
    "data/07_FINAL_REPAIRED_DATASET_V2/06_STATIC_92_RAW_BY_SAMPLE.parquet",
    "data/07_FINAL_REPAIRED_DATASET_V2/07_DYNAMIC_70x10_FLOAT32.npy",
    "data/07_FINAL_REPAIRED_DATASET_V2/08_PAIR_TENSOR_INDEX_INT64.npy",
    "external/kyushu_2017_asakura_toho/99_frozen_dataset/06_matched_index/kyushu_external_matched_triplet_index_v1.parquet",
    "external/kyushu_2017_asakura_toho/99_frozen_dataset/07_matched_static/kyushu_external_matched_static_92f_v1.parquet",
    "external/kyushu_2017_asakura_toho/99_frozen_dataset/09_matched_dynamic_70_view/kyushu_external_matched_dynamic_10f_70slot_view_v1.parquet",
]


def missing(paths: list[str]) -> list[str]:
    return [p for p in paths if not (ROOT / p).is_file()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="require non-redistributed scientific inputs")
    args = parser.parse_args()
    package_missing = missing(PACKAGE_REQUIRED)
    if package_missing:
        print("FAIL package metadata missing:")
        print("\n".join(f"  - {p}" for p in package_missing))
        return 1
    print("PASS package metadata present")
    full_missing = missing(FULL_RERUN_REQUIRED)
    if full_missing:
        print("INFO full-rerun inputs not bundled:")
        print("\n".join(f"  - {p}" for p in full_missing))
        if args.full:
            return 2
    else:
        print("PASS full-rerun inputs present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

