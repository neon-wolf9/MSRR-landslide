#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
124D_RUN_KYUSHU2017_FROZEN_EXTERNAL_VALIDATION.py

Formal untouched external-event validation:
    Hiroshima 2018 development benchmark -> Kyushu 2017 Asakura-Toho

Why 124D exists
---------------
The first 124 technical attempt stopped BEFORE any Kyushu prediction because it
correctly detected that the formal Hiroshima model-ready arrays differ across
folds. Audit of the authority scripts established that this is intentional:
preprocessing is fitted on training folds only. 124D therefore implements the
standard external-deployment analogue without changing MSRR or using Kyushu for
selection:

    1) use existing Hiroshima 120B validation results to select one deployment
       XGBoost configuration separately for RAW and MSRR;
    2) fit ONE final preprocessor on ALL Hiroshima development samples;
    3) transform all Hiroshima development samples with that preprocessor;
    4) apply the SAME Hiroshima-fitted preprocessing parameters to raw Kyushu
       static/rainfall inputs;
    5) fit final RAW and MSRR XGBoost models on Hiroshima only;
    6) generate and freeze all Kyushu scores;
    7) only then parse Kyushu outcome/role metadata and calculate external
       AUROC/AUPRC/StrictPair/Edge and paired matched-set bootstrap CIs.

Frozen representation
---------------------
RAW  = 92 + 70*10 = 792 dims
MSRR = 92 raw static + 92 signed static residual + 92 abs static residual
     + 700 raw dynamic + 700 signed dynamic residual + 700 abs dynamic residual
     + 70 residual temporal summaries = 2446 dims

No Kyushu information is used for feature selection, preprocessing fitting,
hyperparameter/model selection, threshold selection, or post-result rescue.
The external benchmark's pair_set_id is used only to construct the matched-set
context; labels/sample roles are not parsed until all external predictions have
been fixed.

Run
---
python ^
  <PROJECT_ROOT>\scripts\124D_RUN_KYUSHU2017_FROZEN_EXTERNAL_VALIDATION.py ^
  --xgb-jobs 8

Output
------
<PROJECT_ROOT>\external\kyushu_2017_asakura_toho\100_external_validation\MSRR_KYUSHU2017_EXTERNAL_V1D
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


# =============================================================================
# 0. FROZEN CONSTANTS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
KYUSHU = ROOT / "external" / "kyushu_2017_asakura_toho" / "99_frozen_dataset"
OUT120B = ROOT / "experiments" / "MSRR_CROSS_BACKBONE_5FOLD_V1B"
OUT = (
    ROOT
    / "external"
    / "kyushu_2017_asakura_toho"
    / "100_external_validation"
    / "MSRR_KYUSHU2017_EXTERNAL_V1D"
)

SEED = 7
FINAL_XGB_SEED = SEED + 600
BOOTSTRAP_SEED = 20260830
BOOTSTRAP_B = 10_000
METRICS4 = ["AUROC", "AUPRC", "StrictPair", "Edge"]

EXPECTED = {
    "master_grid_rows": 13478,
    "matched_sets": 1692,
    "matched_rows": 5076,
    "positive": 1692,
    "controls": 3384,
    "dynamic192_rows": 974592,
    "dynamic70_rows": 355320,
    "steps": 70,
    "dynamic_dim": 10,
    "raw_dim": 792,
    "msrr_dim": 2446,
}

CATEGORY_COUNT = 14

UNIT_ALIASES = ["unit_id", "grid_id", "sample_id"]
PAIR_ALIASES = ["pair_set_id", "pair_id", "matched_set_id", "set_id", "triplet_id"]
ROLE_ALIASES = ["sample_role", "role"]
RANK_ALIASES = ["control_rank", "ctrl_rank"]
LABEL_ALIASES = ["y_pair", "y_true", "label", "target", "y"]
TIME_ALIASES = [
    "timestamp_utc", "timestamp", "datetime", "valid_time", "time",
    "model_step", "model_timestep", "relative_step", "sequence_step",
    "time_idx", "time_index", "timestep", "step", "rain_step", "step_70",
]


# =============================================================================
# 1. BASIC UTILITIES
# =============================================================================

def log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (OUT / "124D_RUN.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def fail(code: str, msg: str) -> None:
    log("")
    log("=" * 120)
    log(code)
    log(msg)
    log("=" * 120)
    raise RuntimeError(msg)


def to_builtin(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): to_builtin(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_builtin(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def jwrite(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_builtin(obj), ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def find_col(cols: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    exact = {str(c).lower(): str(c) for c in cols}
    normalized = {norm(c): str(c) for c in cols}
    for a in aliases:
        if a.lower() in exact:
            return exact[a.lower()]
        if norm(a) in normalized:
            return normalized[norm(a)]
    return None


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def prob_to_margin(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p) - np.log1p(-p)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported table: {path}")


# =============================================================================
# 2. IMPORT FORMAL AUTHORITIES
# =============================================================================

def locate_script(pattern: str, preferred_name: str) -> Path:
    preferred = ROOT / "scripts" / preferred_name
    if preferred.exists():
        return preferred
    cands = sorted((ROOT / "scripts").glob(pattern))
    if not cands:
        raise FileNotFoundError(f"Cannot locate {preferred_name} under {ROOT / 'scripts'}")
    return cands[0]


def import_file(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import authority script: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_authorities():
    p120 = locate_script(
        "120B_RUN_MSRR_CROSS_BACKBONE_5FOLD_NO_LIGHTGBM_ONE_SHOT*.py",
        "120B_RUN_MSRR_CROSS_BACKBONE_5FOLD_NO_LIGHTGBM_ONE_SHOT.py",
    )
    p26 = locate_script(
        "26_RUN_SEHC_NET_M_V1_FORMAL*.py",
        "26_RUN_SEHC_NET_M_V1_FORMAL.py",
    )

    m120 = import_file("external_frozen_120b", p120)
    m26 = import_file("external_static92_bridge", p26)

    # 120B imports TRAIN_SEHC_V2_TEMPLATE; load_dataset_loader gives us the
    # validated 23D runner and 20A base authority objects.
    bundle, runner, base = m120.load_dataset_loader()

    # Redirect any helper logging away from formal historical output directories.
    m120.log = log
    m120.OUT = OUT
    if hasattr(runner, "log"):
        runner.log = log

    return p120, p26, m120, m26, bundle, runner, base


# =============================================================================
# 3. PRE-RESULT PROTOCOL FREEZE
# =============================================================================

def freeze_protocol(
    p120: Path,
    p26: Path,
    runner: Any,
    base: Any,
    xgb_jobs: int,
) -> None:
    authority_paths = {
        "120B": p120,
        "26_static92_bridge": p26,
    }

    for label, attr in [
        ("23D_preprocessing", "__file__"),
    ]:
        p = Path(getattr(runner, attr))
        authority_paths[label] = p

    p20 = Path(getattr(base, "__file__"))
    authority_paths["20A_data_schema"] = p20

    jwrite(
        OUT / "124D_PROTOCOL_FROZEN.json",
        {
            "experiment": "124D_KYUSHU2017_FROZEN_EXTERNAL_VALIDATION",
            "status": "FROZEN_BEFORE_EXTERNAL_PREDICTION_AND_METRICS",
            "development_event": "Hiroshima_2018",
            "external_event": "Kyushu_2017_Asakura_Toho",
            "external_dataset_status": "PASS_KYUSHU_EXTERNAL_DATASET_CONSTRUCTION_COMPLETE_AND_FROZEN",
            "external_dataset_scope": "DATA_ONLY_NO_MODEL_NO_PREDICTION",
            "external_dataset_root": str(KYUSHU),
            "technical_predecessor": (
                "124 V1 stopped before any external prediction because fold-specific preprocessing was detected; "
                "124B stopped before any external prediction because duplicate-format frozen input exports "
                "were conservatively treated as ambiguous; 124D safely resolves only contract-equivalent duplicates."
            ),
            "primary_downstream_learner": "XGBoost",
            "representations": ["RAW", "MSRR"],
            "raw_dim": 792,
            "msrr_dim": 2446,
            "seed": SEED,
            "final_xgb_seed": FINAL_XGB_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_B": BOOTSTRAP_B,
            "deployment_preprocessing_rule": (
                "Fit the unchanged formal 23D median/IQR/log1p/category preprocessing once on ALL Hiroshima "
                "development samples after internal model selection, then apply those exact Hiroshima-fitted "
                "parameters unchanged to Kyushu raw inputs."
            ),
            "deployment_xgb_config_rule": (
                "Separately for RAW and MSRR, choose D4/D6/D8 by the highest mean validation_score across "
                "the five existing Hiroshima 120B validation folds. Kyushu is not consulted."
            ),
            "external_pair_context_rule": (
                "pair_set_id groups the frozen P+2HC set for MSRR context; within-set row ordering before "
                "prediction is label-blind and irrelevant because the set mean is permutation invariant."
            ),
            "external_label_rule": (
                "Kyushu outcome/sample-role metadata is parsed only after all RAW/MSRR external scores are fixed."
            ),
            "kyushu_used_for_feature_selection": False,
            "kyushu_used_for_preprocessor_fit": False,
            "kyushu_used_for_hyperparameter_selection": False,
            "kyushu_used_for_model_selection": False,
            "kyushu_used_for_threshold_selection": False,
            "kyushu_used_for_method_rescue": False,
            "post_external_result_rescue_allowed": False,
            "bootstrap_unit": "matched set; P+C1+C2 kept together; identical draw for RAW and MSRR",
            "gate": {
                "STRONG_CROSS_EVENT_TRANSFER": (
                    "MSRR>RAW on all four external metrics AND all four paired matched-set bootstrap 95% CI lower bounds >0"
                ),
                "CROSS_EVENT_TRANSFER_SUPPORT": (
                    "MSRR>RAW on all four external metrics, but at least one paired-bootstrap CI includes zero"
                ),
                "PARTIAL_EXTERNAL_SUPPORT": "MSRR>RAW on at least three of four external metrics",
                "otherwise": "EXTERNAL_TRANSFER_NOT_SUPPORTED",
            },
            "authority_files": {
                k: {"path": str(v), "sha256": sha256(v)}
                for k, v in authority_paths.items()
            },
            "xgb_jobs": int(xgb_jobs),
        },
    )


# =============================================================================
# 4. EXTERNAL FILE INVENTORY / DISCOVERY
# =============================================================================

def parquet_meta(path: Path) -> Tuple[Optional[int], List[str]]:
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        return int(pf.metadata.num_rows), list(pf.schema.names)
    except Exception:
        return None, []


def count_csv_rows(path: Path) -> int:
    n = 0
    with path.open("rb") as f:
        for _ in f:
            n += 1
    return max(0, n - 1)


def inventory(root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in {".parquet", ".csv", ".json"}:
            continue

        rec = {
            "path": str(p),
            "name": p.name,
            "suffix": p.suffix.lower(),
            "rows": np.nan,
            "columns": "",
            "size_mb": p.stat().st_size / 1024 / 1024,
        }

        if p.suffix.lower() == ".parquet":
            n, cols = parquet_meta(p)
            rec["rows"] = n if n is not None else np.nan
            rec["columns"] = "|".join(cols)
        elif p.suffix.lower() == ".csv":
            try:
                cols = list(pd.read_csv(p, nrows=0).columns)
                rec["columns"] = "|".join(cols)
            except Exception:
                pass

        rows.append(rec)

    return pd.DataFrame(rows)


def rowcount_if_needed(inv: pd.DataFrame, i: int) -> Optional[int]:
    if pd.notna(inv.loc[i, "rows"]):
        return int(inv.loc[i, "rows"])
    if inv.loc[i, "suffix"] == ".csv":
        n = count_csv_rows(Path(inv.loc[i, "path"]))
        inv.loc[i, "rows"] = n
        return int(n)
    return None


def columns_of_record(r: pd.Series) -> List[str]:
    s = str(r.get("columns", ""))
    return [x for x in s.split("|") if x]


def _equivalent_candidate_tables(
    paths: Sequence[Path],
    required_exact_cols: Sequence[str],
    require_unit: bool,
    require_pair: bool,
    require_role_or_label: bool,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Resolve duplicate-format exports safely.

    A frozen dataset may intentionally retain the same formal table in both
    CSV and Parquet. That is not scientific ambiguity if the contract-relevant
    contents are equivalent. We compare deterministic contract fields only.
    """
    dfs = [read_table(Path(p)).copy() for p in paths]

    if len({len(df) for df in dfs}) != 1:
        return False, {"reason": "row_count_differs"}

    compare_groups: List[Tuple[str, Sequence[str]]] = []

    for c in required_exact_cols:
        compare_groups.append((f"required:{c}", [c]))

    if require_unit:
        compare_groups.append(("unit", UNIT_ALIASES))
    if require_pair:
        compare_groups.append(("pair", PAIR_ALIASES))
    if require_role_or_label:
        compare_groups.append(("role", ROLE_ALIASES))
        compare_groups.append(("label", LABEL_ALIASES))

    compare_groups.append(("time", TIME_ALIASES))
    compare_groups.append(("rank", RANK_ALIASES))

    canonical = []

    for df in dfs:
        selected: Dict[str, pd.Series] = {}

        for semantic, aliases in compare_groups:
            col = find_col(df.columns, aliases)
            if col is not None:
                selected[semantic] = df[col]

        for c in required_exact_cols:
            if c not in df.columns:
                return False, {"reason": f"required_column_missing:{c}"}
            selected[f"required:{c}"] = df[c]

        if require_unit and "unit" not in selected:
            return False, {"reason": "unit_column_missing"}
        if require_pair and "pair" not in selected:
            return False, {"reason": "pair_column_missing"}
        if require_role_or_label and not (
            "role" in selected or "label" in selected
        ):
            return False, {"reason": "role_and_label_missing"}

        cdf = pd.DataFrame(selected)

        sort_cols = [
            c for c in ["pair", "unit", "time", "rank"]
            if c in cdf.columns
        ]

        if sort_cols:
            cdf = (
                cdf.sort_values(sort_cols, kind="mergesort")
                .reset_index(drop=True)
            )
        else:
            cdf = cdf.reset_index(drop=True)

        canonical.append(cdf)

    base_df = canonical[0]

    for other in canonical[1:]:
        if list(base_df.columns) != list(other.columns):
            return False, {
                "reason": "contract_column_set_differs",
                "base_columns": list(base_df.columns),
                "other_columns": list(other.columns),
            }

        for c in base_df.columns:
            a = base_df[c]
            b = other[c]

            an = pd.to_numeric(a, errors="coerce")
            bn = pd.to_numeric(b, errors="coerce")

            numeric_comparable = (
                an.notna().sum() + a.isna().sum() == len(a)
                and bn.notna().sum() + b.isna().sum() == len(b)
            )

            if numeric_comparable:
                # pandas nullable integer/float dtypes may carry pd.NA.
                # Convert them explicitly to ordinary NumPy float64 with NaN
                # before numerical equivalence checking.
                av = an.to_numpy(
                    dtype=np.float64,
                    na_value=np.nan,
                )
                bv = bn.to_numpy(
                    dtype=np.float64,
                    na_value=np.nan,
                )
                if not np.allclose(
                    av,
                    bv,
                    rtol=1e-7,
                    atol=1e-8,
                    equal_nan=True,
                ):
                    return False, {"reason": f"numeric_values_differ:{c}"}
            else:
                av = a.fillna("<NA>").astype(str).to_numpy()
                bv = b.fillna("<NA>").astype(str).to_numpy()
                if not np.array_equal(av, bv):
                    return False, {"reason": f"text_values_differ:{c}"}

    return True, {
        "reason": "equivalent_contract_contents",
        "rows": int(len(base_df)),
        "compared_columns": list(base_df.columns),
    }


def discover_by_schema(
    inv: pd.DataFrame,
    expected_rows: Sequence[int],
    required_exact_cols: Sequence[str] = (),
    require_unit: bool = False,
    require_pair: bool = False,
    require_role_or_label: bool = False,
    positive_name_tokens: Sequence[str] = (),
    negative_name_tokens: Sequence[str] = (),
    label: str = "FILE",
) -> Path:
    scored: List[Tuple[int, int]] = []

    for i, r in inv.iterrows():
        if r["suffix"] not in {".parquet", ".csv"}:
            continue

        cols = columns_of_record(r)
        colset = set(cols)

        if required_exact_cols and not set(required_exact_cols).issubset(colset):
            continue
        if require_unit and find_col(cols, UNIT_ALIASES) is None:
            continue
        if require_pair and find_col(cols, PAIR_ALIASES) is None:
            continue
        if require_role_or_label:
            if (
                find_col(cols, ROLE_ALIASES) is None
                and find_col(cols, LABEL_ALIASES) is None
            ):
                continue

        n = rowcount_if_needed(inv, i)
        if n is None or n not in set(int(x) for x in expected_rows):
            continue

        name = str(r["name"]).lower()
        score = 20
        score += 3 * sum(
            tok.lower() in name
            for tok in positive_name_tokens
        )
        score -= 4 * sum(
            tok.lower() in name
            for tok in negative_name_tokens
        )

        scored.append((i, score))

    if not scored:
        raise RuntimeError(
            f"{label}: no exact schema+rowcount candidate. "
            "See 124D_EXTERNAL_FILE_INVENTORY.csv"
        )

    scored.sort(key=lambda x: x[1], reverse=True)

    best_score = scored[0][1]
    tied = [i for i, s in scored if s == best_score]

    if len(tied) == 1:
        return Path(inv.loc[tied[0], "path"])

    tied_paths = [Path(inv.loc[i, "path"]) for i in tied]

    equivalent, audit = _equivalent_candidate_tables(
        tied_paths,
        required_exact_cols,
        require_unit,
        require_pair,
        require_role_or_label,
    )

    if not equivalent:
        top = [
            (str(inv.loc[i, "path"]), best_score)
            for i in tied[:8]
        ]
        raise RuntimeError(
            f"{label}: genuinely ambiguous candidates={top}; "
            f"equivalence_audit={audit}"
        )

    chosen = sorted(
        tied_paths,
        key=lambda p: (
            0 if p.suffix.lower() == ".parquet" else 1,
            str(p).lower(),
        ),
    )[0]

    log(
        f"{label}_DUPLICATE_EXPORTS_EQUIVALENT=YES "
        f"candidates={len(tied_paths)} chosen={chosen}"
    )

    return chosen


# =============================================================================
# 5. HIROSHIMA FULL-DEVELOPMENT PREPROCESSOR
# =============================================================================

def fit_full_hiroshima_preprocessor(
    bundle: Any,
    runner: Any,
    base: Any,
    m26: Any,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any]]:

    all_idx = np.arange(base.N, dtype=np.int64)

    pre = runner.fit_preprocessor(
        bundle,
        all_idx,
    )

    if int(pre.get("fit_sample_count", -1)) != base.N:
        raise RuntimeError(
            f"Full-Hiroshima preprocessor fit_sample_count={pre.get('fit_sample_count')} expected={base.N}"
        )

    xs91, category_index, rain = runner.transform_all(
        bundle,
        pre,
    )

    static92 = m26.build_static_92(
        xs91,
        category_index,
    )

    static92 = np.asarray(static92, dtype=np.float32)
    rain = np.asarray(rain, dtype=np.float32)

    if static92.shape != (base.N, 92):
        raise RuntimeError(f"Hiroshima full static92 shape={static92.shape}")
    if rain.shape != (base.N, 70, 10):
        raise RuntimeError(f"Hiroshima full rain shape={rain.shape}")
    if not np.isfinite(static92).all() or not np.isfinite(rain).all():
        raise RuntimeError("Nonfinite values after full-Hiroshima preprocessing")

    schema = {
        "raw_continuous_fields_in_order": list(bundle.ss["continuous_fields_in_order"]),
        "model_continuous_fields_in_order": list(bundle.cont),
        "categorical_field": str(bundle.cat),
        "soil_fields": list(bundle.soil),
        "rain_fields_in_order": list(base.RAIN_FIELDS),
        "excluded_raw_continuous_field": "geology_metamorphic_rock_fraction",
        "category_count": CATEGORY_COUNT,
    }

    audit = {
        "fit_scope": "ALL_HIROSHIMA_DEVELOPMENT_SAMPLES_AFTER_INTERNAL_SELECTION",
        "fit_sample_count": int(pre["fit_sample_count"]),
        "dynamic_fit_values_per_field": int(pre["dynamic_fit_values_per_field"]),
        "future_steps_used": int(pre["future_steps_used"]),
        "static_model_continuous_count": len(pre["static"]),
        "dynamic_field_count": len(pre["dynamic"]),
        "category_vocab_size": len(pre["category_map"]),
        "category_field": pre["category_field"],
        "static92_shape": list(static92.shape),
        "rain_shape": list(rain.shape),
    }

    return static92, rain, pre, {"schema": schema, "audit": audit}


# =============================================================================
# 6. LABEL-BLIND EXTERNAL MATCHED-SET INDEX
# =============================================================================

def build_external_model_index(matched_raw: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    u = find_col(matched_raw.columns, UNIT_ALIASES)
    p = find_col(matched_raw.columns, PAIR_ALIASES)

    if u is None or p is None:
        raise RuntimeError("External matched index requires unit_id and pair_set_id equivalents")

    m = pd.DataFrame(
        {
            "unit_id": matched_raw[u].astype(str),
            "pair_set_id": matched_raw[p].astype(str),
        }
    )

    if len(m) != EXPECTED["matched_rows"]:
        raise RuntimeError(
            f"External matched rows={len(m)} expected={EXPECTED['matched_rows']}"
        )

    if m["unit_id"].duplicated().any():
        raise RuntimeError("External matched index has duplicate unit_id")

    sizes = m.groupby("pair_set_id", sort=False).size()
    if len(sizes) != EXPECTED["matched_sets"] or not (sizes == 3).all():
        raise RuntimeError("External pair_set_id structure is not exactly 1692 sets x 3 rows")

    # Label-blind deterministic row order for feature construction/prediction.
    m = (
        m.sort_values(["pair_set_id", "unit_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    m["model_row"] = np.arange(len(m), dtype=np.int64)

    pair_rows = np.arange(len(m), dtype=np.int64).reshape(-1, 3)

    return m, pair_rows


# =============================================================================
# 7. APPLY HIROSHIMA-FITTED PREPROCESSOR TO KYUSHU RAW INPUTS
# =============================================================================

def external_soil_indicator(
    static_ordered: pd.DataFrame,
    soil_fields: Sequence[str],
) -> np.ndarray:
    missing = [f for f in soil_fields if f not in static_ordered.columns]
    if missing:
        raise RuntimeError(f"External static table missing soil fields: {missing}")

    x = (
        static_ordered[list(soil_fields)]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(np.float64)
    )

    c = (~np.isfinite(x)).sum(axis=1)

    partial = (c != 0) & (c != len(soil_fields))
    if np.any(partial):
        raise RuntimeError(
            f"External partial soil missingness rows={int(np.sum(partial))}; formal contract permits only 0 or all-18 missing"
        )

    return (c == len(soil_fields)).astype(np.float32)


def load_external_dynamic_raw(
    dyn_df: pd.DataFrame,
    model_index: pd.DataFrame,
    rain_fields: Sequence[str],
) -> Tuple[np.ndarray, str]:
    du = find_col(dyn_df.columns, UNIT_ALIASES)
    if du is None:
        raise RuntimeError("External dynamic70 table has no unit_id equivalent")

    missing_fields = [f for f in rain_fields if f not in dyn_df.columns]
    if missing_fields:
        raise RuntimeError(
            f"External dynamic70 table missing formal rain fields: {missing_fields}"
        )

    df = dyn_df.copy()
    df[du] = df[du].astype(str)
    df["_original_row"] = np.arange(len(df), dtype=np.int64)

    counts = df.groupby(du, sort=False).size().reindex(model_index["unit_id"])
    if counts.isna().any():
        missing_units = model_index.loc[counts.isna().to_numpy(), "unit_id"].head(20).tolist()
        raise RuntimeError(f"External dynamic70 missing matched unit_ids: {missing_units}")
    if not (counts.to_numpy() == 70).all():
        nbad = int(np.sum(counts.to_numpy() != 70))
        raise RuntimeError(f"External dynamic70 per-unit 70-step violations={nbad}")

    tcol = find_col(df.columns, TIME_ALIASES)
    if tcol is None:
        raise RuntimeError(
            "External dynamic70 table has no recognized time/step column; refuse to assume chronological order"
        )

    pieces = []

    for unit in model_index["unit_id"]:
        g = df[df[du] == unit].copy()

        # Prefer true datetime ordering when parseable, otherwise stable native sort.
        ordered = None
        try:
            parsed = pd.to_datetime(g[tcol], errors="raise", utc=True)
            if parsed.notna().all() and parsed.nunique() == 70:
                ordered = (
                    g.assign(_parsed_time=parsed)
                    .sort_values(["_parsed_time", "_original_row"], kind="mergesort")
                )
        except Exception:
            ordered = None

        if ordered is None:
            if g[tcol].nunique(dropna=False) != 70:
                raise RuntimeError(
                    f"External dynamic time/step key is not unique within unit={unit}: column={tcol}"
                )
            ordered = g.sort_values([tcol, "_original_row"], kind="mergesort")

        arr = (
            ordered[list(rain_fields)]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(np.float64)
        )

        if arr.shape != (70, 10):
            raise RuntimeError(f"External raw dynamic shape for unit={unit}: {arr.shape}")

        pieces.append(arr)

    raw = np.stack(pieces, axis=0)

    if raw.shape != (EXPECTED["matched_rows"], 70, 10):
        raise RuntimeError(f"External raw dynamic tensor shape={raw.shape}")

    return raw, tcol


def transform_external_with_hiroshima_preprocessor(
    static_df: pd.DataFrame,
    dyn_df: pd.DataFrame,
    model_index: pd.DataFrame,
    bundle: Any,
    runner: Any,
    m26: Any,
    pre: Dict[str, Any],
    base: Any,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:

    su = find_col(static_df.columns, UNIT_ALIASES)
    if su is None:
        raise RuntimeError("External static table has no unit_id equivalent")

    static = static_df.copy()
    static[su] = static[su].astype(str)

    if static[su].duplicated().any():
        raise RuntimeError("External static source has duplicate unit_id")

    required_raw_static = list(bundle.ss["continuous_fields_in_order"]) + [bundle.cat]
    missing_raw = [f for f in required_raw_static if f not in static.columns]
    if missing_raw:
        raise RuntimeError(
            "External static source is not the frozen raw 92-field contract; missing="
            + ",".join(missing_raw)
        )

    missing_units = set(model_index["unit_id"]) - set(static[su])
    if missing_units:
        raise RuntimeError(
            f"External static source missing {len(missing_units)} matched units; examples={list(sorted(missing_units))[:20]}"
        )

    ordered = (
        static.set_index(su)
        .loc[model_index["unit_id"]]
        .reset_index(drop=False)
    )

    # Exact 23D static continuous transform using Hiroshima-fitted med/IQR/log1p.
    cols = []
    for p in pre["static"]:
        f = p["field"]
        if f not in ordered.columns:
            raise RuntimeError(f"External static missing model field={f}")

        x = pd.to_numeric(ordered[f], errors="coerce").to_numpy(np.float64)
        x = runner.maybe_log1p(x, p["log1p"])
        x = np.where(np.isfinite(x), x, float(p["median"]))
        x = ((x - float(p["median"])) / float(p["iqr"])).astype(np.float32)
        cols.append(x)

    soil_ind = external_soil_indicator(
        ordered,
        bundle.soil,
    )
    cols.append(soil_ind)

    xs91 = np.column_stack(cols).astype(np.float32, copy=False)

    if xs91.shape != (EXPECTED["matched_rows"], 91):
        raise RuntimeError(f"External xs91 shape={xs91.shape}")
    if not np.isfinite(xs91).all():
        raise RuntimeError("External xs91 contains nonfinite values")

    raw_cat = ordered[bundle.cat].fillna("MISSING").astype(str)
    cmap = dict(pre["category_map"])
    unknown_index = int(pre["unknown_index"])
    category_index = raw_cat.map(lambda x: cmap.get(x, unknown_index)).to_numpy(np.int64)

    if category_index.min() < 0 or category_index.max() >= CATEGORY_COUNT:
        raise RuntimeError("External category index outside formal [0,13] range")

    unknown_count = int(np.sum(category_index == unknown_index))

    static92 = m26.build_static_92(
        xs91,
        category_index,
    )
    static92 = np.asarray(static92, dtype=np.float32)

    raw_rain, time_col = load_external_dynamic_raw(
        dyn_df,
        model_index,
        base.RAIN_FIELDS,
    )

    rain = np.empty_like(raw_rain, dtype=np.float32)

    if len(pre["dynamic"]) != 10:
        raise RuntimeError(f"Hiroshima preprocessor dynamic fields={len(pre['dynamic'])}")

    for j, p in enumerate(pre["dynamic"]):
        expected_name = str(base.RAIN_FIELDS[j])
        if str(p["field"]) != expected_name:
            raise RuntimeError(
                f"Hiroshima preprocessor dynamic order mismatch at j={j}: {p['field']} vs {expected_name}"
            )

        x = raw_rain[:, :, j]
        x = runner.maybe_log1p(x, p["log1p"])
        x = np.where(np.isfinite(x), x, float(p["median"]))
        rain[:, :, j] = ((x - float(p["median"])) / float(p["iqr"])).astype(np.float32)

    if static92.shape != (EXPECTED["matched_rows"], 92):
        raise RuntimeError(f"External static92 shape={static92.shape}")
    if rain.shape != (EXPECTED["matched_rows"], 70, 10):
        raise RuntimeError(f"External transformed rain shape={rain.shape}")
    if not np.isfinite(static92).all() or not np.isfinite(rain).all():
        raise RuntimeError("External transformed inputs contain nonfinite values")

    audit = {
        "preprocessor_fit_source": "HIROSHIMA_2018_ALL_DEVELOPMENT_SAMPLES_ONLY",
        "external_fit_rows": 0,
        "static_raw_contract_count": len(required_raw_static),
        "model_continuous_count": len(pre["static"]),
        "soil_indicator_count": 1,
        "categorical_field": str(bundle.cat),
        "category_unknown_index": unknown_index,
        "external_unknown_category_rows": unknown_count,
        "dynamic_fields": list(base.RAIN_FIELDS),
        "dynamic_time_column": time_col,
        "static92_shape": list(static92.shape),
        "rain_shape": list(rain.shape),
        "forbidden_metadata_used_as_predictive_features": False,
    }

    return static92, rain, audit


# =============================================================================
# 8. HIROSHIMA-ONLY XGBOOST DEPLOYMENT CONFIG
# =============================================================================

def select_xgb_config(rep: str) -> Tuple[str, pd.DataFrame]:
    path = OUT120B / "VALIDATION_SELECTION.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, low_memory=False)
    need = {
        "backbone", "representation", "config_id", "validation_score",
        "StrictPair", "Edge", "AUPRC", "AUROC",
    }
    if not need.issubset(df.columns):
        raise RuntimeError(
            f"120B validation table missing columns={sorted(need - set(df.columns))}"
        )

    sub = df[
        (df["backbone"].astype(str) == "XGBoost")
        & (df["representation"].astype(str) == rep)
    ].copy()

    if sub.empty:
        raise RuntimeError(f"No existing 120B XGBoost validation rows for {rep}")

    agg = (
        sub.groupby("config_id", as_index=False)
        .agg(
            mean_validation_score=("validation_score", "mean"),
            mean_StrictPair=("StrictPair", "mean"),
            mean_Edge=("Edge", "mean"),
            mean_AUPRC=("AUPRC", "mean"),
            mean_AUROC=("AUROC", "mean"),
            n_validation_rows=("validation_score", "size"),
        )
        .sort_values(
            [
                "mean_validation_score", "mean_StrictPair", "mean_Edge",
                "mean_AUPRC", "mean_AUROC", "config_id",
            ],
            ascending=[False, False, False, False, False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    return str(agg.iloc[0]["config_id"]), agg


def cfg_by_id(m120: Any, cfg_id: str) -> Dict[str, Any]:
    for cfg in m120.XGB_CANDIDATES:
        if str(cfg["config_id"]) == str(cfg_id):
            return dict(cfg)
    raise KeyError(f"Unknown frozen XGBoost config={cfg_id}")


def fit_predict_xgb(
    m120: Any,
    cfg_id: str,
    xtr: np.ndarray,
    ytr: np.ndarray,
    xext: np.ndarray,
    jobs: int,
) -> np.ndarray:
    model = m120.make_xgb(
        cfg_by_id(m120, cfg_id),
        FINAL_XGB_SEED,
        jobs,
    )
    model.fit(
        xtr,
        ytr,
        sample_weight=m120.class_weights(ytr),
    )
    return prob_to_margin(model.predict_proba(xext)[:, 1])


# =============================================================================
# 9. AFTER-PREDICTION EXTERNAL OUTCOME PARSING / METRICS
# =============================================================================

def normalize_role(v: Any) -> str:
    s = str(v).strip().lower()
    if s in {"p", "pos", "positive", "case", "landslide", "1", "true", "matched_positive"}:
        return "positive"
    if s in {"c", "ctrl", "control", "negative", "hard_control", "hardcontrol", "0", "false"}:
        return "control"
    if "pos" in s or "case" in s or "landslide" in s:
        return "positive"
    if "control" in s or "ctrl" in s or "negative" in s:
        return "control"
    return s


def build_external_evaluation_order(
    matched_raw: pd.DataFrame,
    model_index: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray]:
    u = find_col(matched_raw.columns, UNIT_ALIASES)
    p = find_col(matched_raw.columns, PAIR_ALIASES)
    r = find_col(matched_raw.columns, ROLE_ALIASES)
    c = find_col(matched_raw.columns, RANK_ALIASES)
    y = find_col(matched_raw.columns, LABEL_ALIASES)

    if u is None or p is None:
        raise RuntimeError("External matched index missing unit_id/pair_set_id during evaluation")

    e = pd.DataFrame(
        {
            "unit_id": matched_raw[u].astype(str),
            "pair_set_id": matched_raw[p].astype(str),
        }
    )

    if y is not None:
        e["y_true"] = pd.to_numeric(matched_raw[y], errors="raise").astype(np.int8)
    elif r is not None:
        nr = matched_raw[r].map(normalize_role)
        if not nr.isin(["positive", "control"]).all():
            bad = sorted(pd.unique(nr[~nr.isin(["positive", "control"])]).tolist())
            raise RuntimeError(f"Unrecognized external sample roles={bad}")
        e["y_true"] = np.where(nr == "positive", 1, 0).astype(np.int8)
    else:
        raise RuntimeError("External matched index requires y/label or sample_role for post-prediction metrics")

    if r is not None:
        e["sample_role"] = matched_raw[r].map(normalize_role).astype(str)
    else:
        e["sample_role"] = np.where(e["y_true"] == 1, "positive", "control")

    if c is not None:
        e["control_rank"] = pd.to_numeric(matched_raw[c], errors="coerce")
    else:
        e["control_rank"] = np.nan

    row_map = model_index.set_index("unit_id")["model_row"]
    if not set(e["unit_id"]).issubset(set(row_map.index)):
        raise RuntimeError("Evaluation unit_id set differs from pre-prediction model index")
    e["model_row"] = e["unit_id"].map(row_map).astype(np.int64)

    audit = e.groupby("pair_set_id").agg(n=("unit_id", "size"), npos=("y_true", "sum"))
    if len(audit) != EXPECTED["matched_sets"]:
        raise RuntimeError(f"External evaluation pair sets={len(audit)}")
    if not (audit["n"] == 3).all() or not (audit["npos"] == 1).all():
        raise RuntimeError("External evaluation is not exactly 1 positive + 2 controls per set")

    e["_role_order"] = np.where(e["y_true"] == 1, 0, 1)
    e["_rank_order"] = e["control_rank"].fillna(999999.0)
    e = (
        e.sort_values(
            ["pair_set_id", "_role_order", "_rank_order", "unit_id"],
            kind="mergesort",
        )
        .drop(columns=["_role_order", "_rank_order"])
        .reset_index(drop=True)
    )

    y3 = e["y_true"].to_numpy(np.int8).reshape(-1, 3)
    if not np.all(y3[:, 0] == 1) or not np.all(y3[:, 1:] == 0):
        raise RuntimeError("Post-prediction P/C1/C2 evaluation ordering failed")

    eval_rows = e["model_row"].to_numpy(np.int64).reshape(-1, 3)

    return e, eval_rows


def metrics_from_eval_rows(
    model_scores: np.ndarray,
    eval_rows: np.ndarray,
) -> Dict[str, float]:
    s3 = np.asarray(model_scores, dtype=np.float64)[eval_rows]
    if s3.shape != (EXPECTED["matched_sets"], 3):
        raise RuntimeError(f"Evaluation score matrix shape={s3.shape}")

    e1 = s3[:, 0] > s3[:, 1]
    e2 = s3[:, 0] > s3[:, 2]

    flat = s3.reshape(-1)
    y = np.tile(np.asarray([1, 0, 0], dtype=np.int8), len(s3))
    p = sigmoid(flat)

    return {
        "AUROC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)),
        "StrictPair": float((e1 & e2).mean()),
        "Edge": float(np.stack([e1, e2], axis=1).mean()),
    }


# =============================================================================
# 10. PAIRED MATCHED-SET BOOTSTRAP
# =============================================================================

def paired_bootstrap(
    raw_model_scores: np.ndarray,
    msrr_model_scores: np.ndarray,
    eval_rows: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    raw3 = np.asarray(raw_model_scores, dtype=np.float64)[eval_rows]
    msrr3 = np.asarray(msrr_model_scores, dtype=np.float64)[eval_rows]
    n_sets = len(eval_rows)

    raw_obs = metrics_from_eval_rows(raw_model_scores, eval_rows)
    msrr_obs = metrics_from_eval_rows(msrr_model_scores, eval_rows)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    store = {m: np.empty(BOOTSTRAP_B, dtype=np.float64) for m in METRICS4}

    # For each replicate we evaluate scores in canonical P,C1,C2 order.
    y = np.tile(np.asarray([1, 0, 0], dtype=np.int8), n_sets)

    rep_rows = []

    def metric_from_s3(s3: np.ndarray) -> Dict[str, float]:
        e1 = s3[:, 0] > s3[:, 1]
        e2 = s3[:, 0] > s3[:, 2]
        p = sigmoid(s3.reshape(-1))
        return {
            "AUROC": float(roc_auc_score(y, p)),
            "AUPRC": float(average_precision_score(y, p)),
            "StrictPair": float((e1 & e2).mean()),
            "Edge": float(np.stack([e1, e2], axis=1).mean()),
        }

    for b in range(BOOTSTRAP_B):
        draw = rng.integers(0, n_sets, size=n_sets, dtype=np.int64)
        mr = metric_from_s3(raw3[draw])
        mm = metric_from_s3(msrr3[draw])

        for metric in METRICS4:
            d = mm[metric] - mr[metric]
            store[metric][b] = d
            rep_rows.append(
                {
                    "bootstrap_id": b,
                    "metric": metric,
                    "delta_MSRR_minus_RAW": d,
                }
            )

        if b == 0 or (b + 1) % 500 == 0:
            log(f"PAIRED_BOOTSTRAP={b + 1}/{BOOTSTRAP_B}")

    rows = []

    for metric in METRICS4:
        d = store[metric]
        lo, hi = np.quantile(d, [0.025, 0.975])
        tail = min(int(np.sum(d <= 0)), int(np.sum(d >= 0)))
        p_two = min(1.0, 2.0 * ((tail + 1) / (BOOTSTRAP_B + 1)))

        rows.append(
            {
                "metric": metric,
                "RAW_EXTERNAL": raw_obs[metric],
                "MSRR_EXTERNAL": msrr_obs[metric],
                "OBSERVED_DELTA": msrr_obs[metric] - raw_obs[metric],
                "BOOTSTRAP_MEAN_DELTA": float(np.mean(d)),
                "BOOTSTRAP_MEDIAN_DELTA": float(np.median(d)),
                "CI95_LOW": float(lo),
                "CI95_HIGH": float(hi),
                "P_DELTA_GT_0": float(np.mean(d > 0)),
                "TWO_SIDED_BOOTSTRAP_P": p_two,
                "CI95_LOW_GT_0": bool(lo > 0),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(rep_rows)


# =============================================================================
# 11. FIGURES
# =============================================================================

def make_figures(metrics_df: pd.DataFrame, boot_df: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        log(f"WARNING figure generation skipped: {exc}")
        return

    raw = metrics_df[metrics_df["representation"] == "RAW"].iloc[0]
    msrr = metrics_df[metrics_df["representation"] == "MSRR"].iloc[0]

    x = np.arange(len(METRICS4), dtype=float)
    raw_vals = [float(raw[m]) for m in METRICS4]
    msrr_vals = [float(msrr[m]) for m in METRICS4]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - 0.18, raw_vals, width=0.36, label="RAW")
    ax.bar(x + 0.18, msrr_vals, width=0.36, label="MSRR")
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS4)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Kyushu 2017 external metric")
    ax.set_title("Untouched external validation: Hiroshima 2018 → Kyushu 2017")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "124D_EXTERNAL_METRICS.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "124D_EXTERNAL_METRICS.pdf", bbox_inches="tight")
    plt.close(fig)

    y = np.arange(len(boot_df), dtype=float)
    obs = boot_df["OBSERVED_DELTA"].to_numpy(float)
    lo = boot_df["CI95_LOW"].to_numpy(float)
    hi = boot_df["CI95_HIGH"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(obs, y, xerr=np.vstack([obs - lo, hi - obs]), fmt="o", capsize=3)
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(boot_df["metric"].astype(str))
    ax.set_xlabel("MSRR − RAW (95% paired matched-set bootstrap CI)")
    ax.set_title("Kyushu 2017 untouched cross-event transfer")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "124D_EXTERNAL_BOOTSTRAP_FOREST.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "124D_EXTERNAL_BOOTSTRAP_FOREST.pdf", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 12. MAIN
# =============================================================================

def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xgb-jobs", type=int, default=8)
    args = parser.parse_args(argv)

    if args.xgb_jobs < 1:
        raise ValueError("--xgb-jobs must be >=1")

    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty formal 124D output: {OUT}"
        )

    OUT.mkdir(parents=True, exist_ok=True)

    if not KYUSHU.exists():
        fail("FAIL_124D_KYUSHU_FROZEN_NOT_FOUND", str(KYUSHU))
    if not OUT120B.exists():
        fail("FAIL_124D_120B_OUTPUT_NOT_FOUND", str(OUT120B))

    p120, p26, m120, m26, bundle, runner, base = load_authorities()
    m120.require_xgboost()
    set_seed(SEED)

    freeze_protocol(
        p120,
        p26,
        runner,
        base,
        args.xgb_jobs,
    )

    log("=" * 120)
    log("START 124D KYUSHU2017 FROZEN EXTERNAL VALIDATION")
    log("PROTOCOL_FROZEN_BEFORE_EXTERNAL_PREDICTION_AND_METRICS=YES")
    log("PREPROCESSOR_FIT_SOURCE=ALL_HIROSHIMA_DEVELOPMENT_ONLY")
    log("KYUSHU_USED_FOR_PREPROCESSOR_FIT=NO")
    log("KYUSHU_USED_FOR_MODEL_SELECTION=NO")
    log("POST_EXTERNAL_RESULT_RESCUE_ALLOWED=NO")
    log("=" * 120)

    # -------------------------------------------------------------------------
    # A. Internal-only deployment decisions BEFORE external predictions.
    # -------------------------------------------------------------------------
    y_h_all = bundle.sample.y_pair.to_numpy(np.int8)
    pair_h = np.asarray(bundle.pt, dtype=np.int64)
    m120.verify_pair_order(pair_h, y_h_all, "HIROSHIMA_GLOBAL")

    cfg_raw, audit_raw = select_xgb_config("RAW")
    cfg_msrr, audit_msrr = select_xgb_config("MSRR")

    pd.concat(
        [
            audit_raw.assign(representation="RAW"),
            audit_msrr.assign(representation="MSRR"),
        ],
        ignore_index=True,
    ).to_csv(
        OUT / "124D_HIROSHIMA_CONFIG_SELECTION_AUDIT.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        [
            {
                "representation": "RAW",
                "selected_config": cfg_raw,
                "selection_source": "existing Hiroshima 120B validation only",
            },
            {
                "representation": "MSRR",
                "selected_config": cfg_msrr,
                "selection_source": "existing Hiroshima 120B validation only",
            },
        ]
    ).to_csv(
        OUT / "124D_DEPLOYMENT_CONFIGS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log(f"DEPLOYMENT_CONFIG RAW={cfg_raw} MSRR={cfg_msrr}")

    # -------------------------------------------------------------------------
    # B. Fit one final preprocessor on ALL Hiroshima development samples.
    # -------------------------------------------------------------------------
    static_h, rain_h, full_pre, hprep_info = fit_full_hiroshima_preprocessor(
        bundle,
        runner,
        base,
        m26,
    )

    jwrite(
        OUT / "124D_HIROSHIMA_FULL_PREPROCESSOR.json",
        full_pre,
    )
    jwrite(
        OUT / "124D_HIROSHIMA_PREPROCESSING_AUDIT.json",
        hprep_info,
    )

    xh_raw, ah_raw = m120.raw_features(pair_h, static_h, rain_h)
    xh_msrr, ah_msrr = m120.msrr_features(pair_h, static_h, rain_h)

    hidx = pair_h.reshape(-1)
    if not np.array_equal(ah_raw, hidx) or not np.array_equal(ah_msrr, hidx):
        fail("FAIL_124D_HIROSHIMA_FEATURE_ORDER", "Hiroshima anchor order mismatch")

    y_h = y_h_all[hidx]

    if xh_raw.shape != (base.N, 792):
        fail("FAIL_124D_HIROSHIMA_RAW_DIM", str(xh_raw.shape))
    if xh_msrr.shape != (base.N, 2446):
        fail("FAIL_124D_HIROSHIMA_MSRR_DIM", str(xh_msrr.shape))

    log(
        f"HIROSHIMA_FULL_PREPROCESSING_PASS rows={base.N} "
        f"RAW={xh_raw.shape} MSRR={xh_msrr.shape}"
    )

    # -------------------------------------------------------------------------
    # C. Discover frozen Kyushu files by exact formal schemas, not by guessing
    #    feature positions.
    # -------------------------------------------------------------------------
    inv = inventory(KYUSHU)
    inv.to_csv(
        OUT / "124D_EXTERNAL_FILE_INVENTORY.csv",
        index=False,
        encoding="utf-8-sig",
    )

    raw_static_fields = list(bundle.ss["continuous_fields_in_order"]) + [bundle.cat]

    matched_path = discover_by_schema(
        inv,
        [EXPECTED["matched_rows"]],
        require_unit=True,
        require_pair=True,
        require_role_or_label=True,
        positive_name_tokens=["matched", "pair", "triplet", "sample", "index", "formal"],
        negative_name_tokens=["dynamic", "rain", "192", "70", "prediction"],
        label="MATCHED_INDEX",
    )

    static_path = discover_by_schema(
        inv,
        [EXPECTED["matched_rows"], EXPECTED["master_grid_rows"]],
        required_exact_cols=raw_static_fields,
        require_unit=True,
        positive_name_tokens=["static", "feature", "raw", "92", "matched", "grid"],
        negative_name_tokens=["dynamic", "rain", "prediction", "result"],
        label="RAW_STATIC_92",
    )

    dynamic70_path = discover_by_schema(
        inv,
        [EXPECTED["dynamic70_rows"]],
        required_exact_cols=list(base.RAIN_FIELDS),
        require_unit=True,
        positive_name_tokens=["70", "dynamic", "rain", "view", "matched", "model"],
        negative_name_tokens=["192", "prediction", "result"],
        label="DYNAMIC70_RAW_VIEW",
    )

    log(f"MATCHED_INDEX_FILE={matched_path}")
    log(f"RAW_STATIC_FILE={static_path}")
    log(f"DYNAMIC70_FILE={dynamic70_path}")

    matched_raw = read_table(matched_path)
    static_ext_raw = read_table(static_path)
    dynamic70_raw = read_table(dynamic70_path)

    model_index, pair_k = build_external_model_index(matched_raw)

    # -------------------------------------------------------------------------
    # D. Transform Kyushu using ONLY Hiroshima-fitted preprocessing parameters.
    #    No Kyushu labels/sample roles are parsed here.
    # -------------------------------------------------------------------------
    static_k, rain_k, kprep_audit = transform_external_with_hiroshima_preprocessor(
        static_ext_raw,
        dynamic70_raw,
        model_index,
        bundle,
        runner,
        m26,
        full_pre,
        base,
    )

    xk_raw, ak_raw = m120.raw_features(pair_k, static_k, rain_k)
    xk_msrr, ak_msrr = m120.msrr_features(pair_k, static_k, rain_k)

    kidx = pair_k.reshape(-1)
    if not np.array_equal(ak_raw, kidx) or not np.array_equal(ak_msrr, kidx):
        fail("FAIL_124D_EXTERNAL_FEATURE_ORDER", "External anchor order mismatch")

    if xk_raw.shape != (EXPECTED["matched_rows"], 792):
        fail("FAIL_124D_EXTERNAL_RAW_DIM", str(xk_raw.shape))
    if xk_msrr.shape != (EXPECTED["matched_rows"], 2446):
        fail("FAIL_124D_EXTERNAL_MSRR_DIM", str(xk_msrr.shape))

    jwrite(
        OUT / "124D_EXTERNAL_PREPROCESSING_AUDIT.json",
        {
            **kprep_audit,
            "matched_index_file": str(matched_path),
            "static_file": str(static_path),
            "dynamic70_file": str(dynamic70_path),
            "labels_parsed_before_prediction": False,
            "pair_set_id_used_only_for_context_grouping": True,
        },
    )

    log(
        f"PASS_124D_EXTERNAL_MODEL_INPUT_AUDIT rows={len(model_index)} "
        f"sets={len(pair_k)} RAW={xk_raw.shape} MSRR={xk_msrr.shape}"
    )

    # -------------------------------------------------------------------------
    # E. Fit final Hiroshima models and freeze Kyushu predictions.
    #    Still no Kyushu outcomes used.
    # -------------------------------------------------------------------------
    log(f"FIT_FINAL_XGB_RAW config={cfg_raw} seed={FINAL_XGB_SEED}")
    raw_scores = fit_predict_xgb(
        m120,
        cfg_raw,
        xh_raw,
        y_h,
        xk_raw,
        args.xgb_jobs,
    )

    log(f"FIT_FINAL_XGB_MSRR config={cfg_msrr} seed={FINAL_XGB_SEED}")
    msrr_scores = fit_predict_xgb(
        m120,
        cfg_msrr,
        xh_msrr,
        y_h,
        xk_msrr,
        args.xgb_jobs,
    )

    if raw_scores.shape != (EXPECTED["matched_rows"],):
        fail("FAIL_124D_RAW_PREDICTION_SHAPE", str(raw_scores.shape))
    if msrr_scores.shape != (EXPECTED["matched_rows"],):
        fail("FAIL_124D_MSRR_PREDICTION_SHAPE", str(msrr_scores.shape))
    if not np.isfinite(raw_scores).all() or not np.isfinite(msrr_scores).all():
        fail("FAIL_124D_NONFINITE_PREDICTIONS", "RAW/MSRR predictions contain nonfinite values")

    # Save label-blind predictions immediately, before outcome parsing.
    prelabel_predictions = pd.concat(
        [
            pd.DataFrame(
                {
                    "model_row": model_index["model_row"],
                    "unit_id": model_index["unit_id"],
                    "pair_set_id": model_index["pair_set_id"],
                    "backbone": "XGBoost",
                    "representation": "RAW",
                    "risk_score": raw_scores,
                    "score_sigmoid": sigmoid(raw_scores),
                }
            ),
            pd.DataFrame(
                {
                    "model_row": model_index["model_row"],
                    "unit_id": model_index["unit_id"],
                    "pair_set_id": model_index["pair_set_id"],
                    "backbone": "XGBoost",
                    "representation": "MSRR",
                    "risk_score": msrr_scores,
                    "score_sigmoid": sigmoid(msrr_scores),
                }
            ),
        ],
        ignore_index=True,
    )

    prelabel_predictions.to_parquet(
        OUT / "124D_EXTERNAL_PREDICTIONS_FIXED_BEFORE_LABELS.parquet",
        index=False,
    )

    jwrite(
        OUT / "124D_PREDICTION_FREEZE.json",
        {
            "status": "PASS_ALL_EXTERNAL_PREDICTIONS_FIXED_BEFORE_LABEL_METRICS",
            "prediction_rows_per_representation": EXPECTED["matched_rows"],
            "raw_prediction_sha256": hashlib.sha256(raw_scores.tobytes()).hexdigest(),
            "msrr_prediction_sha256": hashlib.sha256(msrr_scores.tobytes()).hexdigest(),
            "kyushu_outcome_parsed_before_this_point": False,
        },
    )

    log("ALL_EXTERNAL_PREDICTIONS_FIXED=YES")
    log("BEGIN_KYUSHU_OUTCOME_PARSING_AND_METRICS_ONLY_NOW")

    # -------------------------------------------------------------------------
    # F. NOW parse frozen Kyushu outcome/role metadata and calculate metrics.
    # -------------------------------------------------------------------------
    eval_index, eval_rows = build_external_evaluation_order(
        matched_raw,
        model_index,
    )

    npos = int(eval_index["y_true"].sum())
    nctrl = int((eval_index["y_true"] == 0).sum())

    if npos != EXPECTED["positive"] or nctrl != EXPECTED["controls"]:
        fail(
            "FAIL_124D_EXTERNAL_LABEL_COUNTS",
            f"positive={npos} controls={nctrl}",
        )

    raw_metrics = metrics_from_eval_rows(raw_scores, eval_rows)
    msrr_metrics = metrics_from_eval_rows(msrr_scores, eval_rows)

    metrics_df = pd.DataFrame(
        [
            {"backbone": "XGBoost", "representation": "RAW", **raw_metrics},
            {"backbone": "XGBoost", "representation": "MSRR", **msrr_metrics},
        ]
    )
    metrics_df.to_csv(
        OUT / "124D_EXTERNAL_METRICS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    deltas = {m: msrr_metrics[m] - raw_metrics[m] for m in METRICS4}
    wins = int(sum(deltas[m] > 0 for m in METRICS4))

    pd.DataFrame(
        [
            {
                "backbone": "XGBoost",
                **{f"delta_{m}": deltas[m] for m in METRICS4},
                "metric_wins_out_of_4": wins,
            }
        ]
    ).to_csv(
        OUT / "124D_EXTERNAL_RAW_VS_MSRR.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Row-level predictions with outcome metadata added only after freeze.
    label_by_model_row = (
        eval_index.set_index("model_row")
        [["unit_id", "pair_set_id", "sample_role", "control_rank", "y_true"]]
        .sort_index()
        .reset_index()
    )

    if not np.array_equal(
        label_by_model_row["model_row"].to_numpy(np.int64),
        np.arange(EXPECTED["matched_rows"], dtype=np.int64),
    ):
        fail("FAIL_124D_LABEL_JOIN_ORDER", "Outcome metadata did not map one-to-one to prediction rows")

    final_predictions = pd.concat(
        [
            label_by_model_row.assign(
                backbone="XGBoost",
                representation="RAW",
                risk_score=raw_scores,
                score_sigmoid=sigmoid(raw_scores),
            ),
            label_by_model_row.assign(
                backbone="XGBoost",
                representation="MSRR",
                risk_score=msrr_scores,
                score_sigmoid=sigmoid(msrr_scores),
            ),
        ],
        ignore_index=True,
    )

    final_predictions.to_parquet(
        OUT / "124D_EXTERNAL_PREDICTIONS_WITH_LABELS.parquet",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "status": "PASS",
                "matched_index_file": str(matched_path),
                "static_file": str(static_path),
                "dynamic70_file": str(dynamic70_path),
                "matched_sets": len(eval_rows),
                "matched_rows": len(eval_index),
                "positive": npos,
                "controls": nctrl,
                "raw_shape": str(tuple(xk_raw.shape)),
                "msrr_shape": str(tuple(xk_msrr.shape)),
                "preprocessor_fit_source": "Hiroshima only",
                "external_fit_rows": 0,
                "labels_parsed_only_after_prediction_freeze": True,
                "forbidden_metadata_used_as_predictive_features": False,
            }
        ]
    ).to_csv(
        OUT / "124D_EXTERNAL_DATASET_AND_LEAKAGE_AUDIT.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # G. Paired matched-set bootstrap.
    # -------------------------------------------------------------------------
    boot_summary, boot_reps = paired_bootstrap(
        raw_scores,
        msrr_scores,
        eval_rows,
    )

    boot_summary.to_csv(
        OUT / "124D_EXTERNAL_BOOTSTRAP_SUMMARY.csv",
        index=False,
        encoding="utf-8-sig",
    )
    boot_reps.to_parquet(
        OUT / "124D_EXTERNAL_BOOTSTRAP_REPLICATES.parquet",
        index=False,
    )

    ci_positive = int(boot_summary["CI95_LOW_GT_0"].sum())

    if wins == 4 and ci_positive == 4:
        gate = "STRONG_CROSS_EVENT_TRANSFER"
    elif wins == 4:
        gate = "CROSS_EVENT_TRANSFER_SUPPORT"
    elif wins >= 3:
        gate = "PARTIAL_EXTERNAL_SUPPORT"
    else:
        gate = "EXTERNAL_TRANSFER_NOT_SUPPORTED"

    decision = {
        "status": "PASS_124D_KYUSHU2017_EXTERNAL_VALIDATION_COMPLETE",
        "gate124b_decision": gate,
        "primary_backbone": "XGBoost",
        "metric_wins_out_of_4": wins,
        "bootstrap_ci_low_gt0_out_of_4": ci_positive,
        "RAW": raw_metrics,
        "MSRR": msrr_metrics,
        "deltas": deltas,
        "post_external_result_rescue_allowed": False,
        "next_step": "FREEZE_EXTERNAL_RESULT_AND_WRITE_PAPER",
    }

    jwrite(
        OUT / "124D_GATE_DECISION.json",
        decision,
    )

    make_figures(
        metrics_df,
        boot_summary,
    )

    # -------------------------------------------------------------------------
    # H. Terminal summary.
    # -------------------------------------------------------------------------
    log("")
    log("=" * 120)
    log("PASS_124D_KYUSHU2017_EXTERNAL_VALIDATION_COMPLETE")
    log(f"GATE124D_DECISION={gate}")
    log(f"PRIMARY_XGB_METRIC_WINS={wins}/4")
    log(f"PRIMARY_XGB_BOOTSTRAP_CI_LOW_GT0={ci_positive}/4")
    log("=" * 120)

    log(
        "XGBoost RAW: "
        f"AUROC={raw_metrics['AUROC']:.6f} "
        f"AUPRC={raw_metrics['AUPRC']:.6f} "
        f"StrictPair={raw_metrics['StrictPair']:.6f} "
        f"Edge={raw_metrics['Edge']:.6f}"
    )
    log(
        "XGBoost MSRR: "
        f"AUROC={msrr_metrics['AUROC']:.6f} "
        f"AUPRC={msrr_metrics['AUPRC']:.6f} "
        f"StrictPair={msrr_metrics['StrictPair']:.6f} "
        f"Edge={msrr_metrics['Edge']:.6f}"
    )
    log(
        "DELTA: "
        f"AUROC={deltas['AUROC']:+.6f} "
        f"AUPRC={deltas['AUPRC']:+.6f} "
        f"StrictPair={deltas['StrictPair']:+.6f} "
        f"Edge={deltas['Edge']:+.6f}"
    )

    log("XGBoost external paired matched-set bootstrap:")
    for r in boot_summary.itertuples():
        log(
            f"  {r.metric}: delta={r.OBSERVED_DELTA:+.6f} "
            f"CI95=[{r.CI95_LOW:+.6f},{r.CI95_HIGH:+.6f}] "
            f"P(delta>0)={r.P_DELTA_GT_0:.6f} "
            f"p={r.TWO_SIDED_BOOTSTRAP_P:.8f}"
        )

    log(f"OUTPUT={OUT}")
    log("NEXT_STEP=FREEZE_EXTERNAL_RESULT_AND_WRITE_PAPER")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "FAIL_124D_EXTERNAL_VALIDATION",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        jwrite(OUT / "124D_FAILURE.json", failure)
        print("\n" + "=" * 120)
        print("124D EXTERNAL VALIDATION FAILED")
        print(f"{type(exc).__name__}: {exc}")
        print("=" * 120)
        traceback.print_exc()
        raise SystemExit(1)
