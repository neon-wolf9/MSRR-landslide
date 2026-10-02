#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build three experiment-128 adapters and smoke-test outer fold 1 only.

Outer-test labels are loaded only for structural/count assertions inherited
from the frozen sample table.  They are never passed to an evaluator and no
outer-test performance metric is computed or written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import imblearn
import numpy as np
import pandas as pd
import sklearn
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "MSRR_128B0_BASELINE_ADAPTER_SMOKE_TEST"
LOG = OUT / "128B0_RUN.log"
VENDOR = ROOT / "third_party" / "PU-pullbaggingDT"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.msrr128_baselines.data_contract import load_outer_fold
from scripts.msrr128_baselines.metrics import array_hash, validation_metrics
from scripts.msrr128_baselines.pu_bagging_dt import PUBaggingDTAdapter
from scripts.msrr128_baselines.pu_pullbaggingdt_adapter import PUPullBaggingDTAdapter
from scripts.msrr128_baselines.spy_pu_brf import SpyPUBRFAdapter


def log(message: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def jwrite(name: str, payload: Any) -> None:
    (OUT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                       sort_keys=True, default=str) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(VENDOR), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def mapped_prediction_hash(metadata: pd.DataFrame, score: np.ndarray) -> str:
    frame = metadata[["unit_id", "pair_set_id"]].copy()
    frame["score_hex"] = [float(x).hex() for x in np.asarray(score, np.float64)]
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_contract(method: str, metadata: pd.DataFrame, score: np.ndarray) -> dict[str, Any]:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    structure = metadata.groupby("pair_set_id")["sample_role"].agg(list)
    pair_ok = bool(structure.map(lambda x: x.count("MATCHED_POSITIVE") == 1
                                 and x.count("HARD_CONTROL") == 2).all())
    row = {
        "method": method, "n_test_rows": len(score),
        "n_test_sets": int(metadata.pair_set_id.nunique()),
        "finite_scores": int(np.isfinite(score).sum()),
        "missing_scores": int(pd.isna(score).sum()),
        "duplicate_units": int(metadata.unit_id.duplicated().sum()),
        "pair_structure_valid": pair_ok,
        "score_min": float(np.min(score)), "score_max": float(np.max(score)),
        "prediction_hash": mapped_prediction_hash(metadata, score),
    }
    row["pass"] = bool(row["n_test_rows"] == 3033 and row["n_test_sets"] == 1011
                       and row["finite_scores"] == 3033 and row["missing_scores"] == 0
                       and row["duplicate_units"] == 0 and pair_ok)
    return row


def config_identity(config: dict[str, Any]) -> str:
    # Training traces/validation floats are outcomes, not configuration identity.
    excluded = {"validation_metrics", "losses"}
    return json.dumps({k: v for k, v in config.items() if k not in excluded},
                      sort_keys=True, default=str)


def run_method(name: str, factory: Callable[[], Any], data: Any, seed: int,
               validation_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]],
               det_rows: list[dict[str, Any]]) -> tuple[Any, dict[str, Any]]:
    log(f"METHOD={name} RUN1_FIT_START")
    model1 = factory().fit(data.X_fit, data.y_fit, data.X_val, data.y_val,
                           data.val_metadata, seed)
    val1 = model1.predict_score(data.X_val)
    vm1 = validation_metrics(val1, data.y_val, data.val_metadata)
    cfg1 = model1.get_selected_config()
    if name == "Spy-PU+BRF":
        for row in model1.candidate_results:
            if row.get("valid"):
                validation_rows.append({"method": name, **{k: row[k] for k in
                    ["config_id", "AUROC", "AUPRC", "StrictPair", "Edge", "validation_score"]},
                    "selected": row["config_id"] == cfg1["config_id"]})
    else:
        validation_rows.append({"method": name, "config_id": "SMOKE_FIXED", **vm1,
                                "selected": True})
    test1 = model1.predict_score(data.X_test)
    test_rows.append(test_contract(name, data.test_metadata, test1))

    log(f"METHOD={name} RUN2_DETERMINISM_START")
    model2 = factory().fit(data.X_fit, data.y_fit, data.X_val, data.y_val,
                           data.val_metadata, seed)
    val2 = model2.predict_score(data.X_val)
    test2 = model2.predict_score(data.X_test)
    cfg2 = model2.get_selected_config()
    max_val = float(np.max(np.abs(val1 - val2)))
    max_test = float(np.max(np.abs(test1 - test2)))
    det = {
        "method": name,
        "selected_config_equal": config_identity(cfg1) == config_identity(cfg2),
        "validation_length_equal": len(val1) == len(val2),
        "test_length_equal": len(test1) == len(test2),
        "validation_hash_run1": array_hash(val1), "validation_hash_run2": array_hash(val2),
        "test_hash_run1": array_hash(test1), "test_hash_run2": array_hash(test2),
        "validation_max_abs_difference": max_val, "test_max_abs_difference": max_test,
        "tolerance": 1e-12,
    }
    det["pass"] = bool(det["selected_config_equal"] and det["validation_length_equal"]
                       and det["test_length_equal"] and max_val <= 1e-12 and max_test <= 1e-12)
    det_rows.append(det)
    log(f"METHOD={name} COMPLETE VAL_SCORE={vm1['validation_score']:.6f} TEST_CONTRACT_ONLY=PASS")
    return model1, cfg1


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--outer-fold", type=int, default=1)
    p.add_argument("--smoke", action="store_true", help="explicit smoke marker")
    p.add_argument("--seed", type=int, default=7)
    return p


def main() -> None:
    args = parser().parse_args()
    if args.outer_fold != 1:
        raise ValueError("128B0 authorizes outer fold 1 only")
    OUT.mkdir(parents=True, exist_ok=True)
    if LOG.exists():
        LOG.unlink()
    log("START 128B0; OUTER-TEST PERFORMANCE METRICS DISABLED")

    threshold_space = {
        "status": "FROZEN_BEFORE_ANY_TRAINING_OR_TEST_INFERENCE",
        "implementation_choice_selected_on": "fold 2 validation only",
        "candidate_rules": [
            {"config_id": "spy_min", "threshold": "minimum positive score among spies"},
            {"config_id": "spy_q05", "threshold": "5th percentile positive score among spies"},
            {"config_id": "spy_q10", "threshold": "10th percentile positive score among spies"},
        ],
        "selection": "mean(AUROC,AUPRC,StrictPair,Edge); tie-break StrictPair, Edge, AUPRC, AUROC, earlier candidate",
        "outer_test_used": False,
    }
    jwrite("128B0_SPY_THRESHOLD_SEARCH_SPACE.json", threshold_space)

    run_config = {
        "task": "128B0", "mode": "SMOKE_ONLY", "outer_fold": 1,
        "validation_fold": 2, "fit_folds": [3, 4, 5], "seed": args.seed,
        "cli_smoke_flag": bool(args.smoke), "outer_test_metrics_computed": False,
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "numpy": np.__version__, "sklearn": sklearn.__version__,
                        "imbalanced_learn": imblearn.__version__, "torch": torch.__version__,
                        "torch_cuda": torch.cuda.is_available()},
        "budgets": {"PU_BaggingDT_trees": 50, "Spy_stage1_RF_trees": 50,
                    "Spy_BRF_trees": 50, "PU_pull_epochs": 3,
                    "PU_pull_bagging_trees": 20},
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    jwrite("128B0_RUN_CONFIG.json", run_config)
    data = load_outer_fold(args.outer_fold)
    jwrite("128B0_DATA_CONTRACT_AUDIT.json", data.audit)
    log("DATA_CONTRACT=PASS RAW_DIM=792 FIT=3034P+6068U VAL=1011P+2022U TEST_ROWS=3033")

    commit = git("rev-parse", "HEAD")
    if commit != "a8857dad30a454e3155644b18ddd2600df860e70":
        raise RuntimeError(f"UNEXPECTED_VENDOR_COMMIT={commit}")
    if git("status", "--porcelain"):
        raise RuntimeError("VENDOR_COPY_MODIFIED")

    validation_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    det_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    fitted: dict[str, tuple[Any, dict[str, Any]]] = {}
    factories = {
        "PU-BaggingDT": lambda: PUBaggingDTAdapter(n_estimators=50, u_fraction=0.70,
                                                    max_depth=None, min_samples_leaf=1,
                                                    class_weight="balanced"),
        "Spy-PU+BRF": lambda: SpyPUBRFAdapter(spy_fraction=0.15, stage1_trees=50,
                                               brf_trees=50),
        "PU-pullbaggingDT": lambda: PUPullBaggingDTAdapter(epochs=3, pu_trees=20,
                                                           learning_rate=1e-4),
    }
    for name, factory in factories.items():
        try:
            fitted[name] = run_method(name, factory, data, args.seed, validation_rows,
                                      test_rows, det_rows)
            status_rows.append({"method": name, "adapter_built": True, "fit_pass": True,
                                "validation_predict_pass": True, "test_contract_pass": True,
                                "determinism_pass": det_rows[-1]["pass"],
                                "critical_ambiguity": "", "ready_for_full_5fold": det_rows[-1]["pass"]})
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            blockers.append(f"{name}: {reason}")
            status_rows.append({"method": name, "adapter_built": True, "fit_pass": False,
                                "validation_predict_pass": False, "test_contract_pass": False,
                                "determinism_pass": False, "critical_ambiguity": reason,
                                "ready_for_full_5fold": False})
            log(f"METHOD={name} BLOCKING_REASON={reason}")
            (OUT / f"128B0_{name.replace('+','_').replace('-','_')}_TRACEBACK.txt").write_text(
                traceback.format_exc(), encoding="utf-8")

    pd.DataFrame(validation_rows, columns=["method", "config_id", "AUROC", "AUPRC",
                 "StrictPair", "Edge", "validation_score", "selected"]).to_csv(
                     OUT / "128B0_VALIDATION_RESULTS.csv", index=False)
    pd.DataFrame(test_rows, columns=["method", "n_test_rows", "n_test_sets", "finite_scores",
                 "missing_scores", "duplicate_units", "pair_structure_valid", "score_min",
                 "score_max", "prediction_hash", "pass"]).to_csv(
                     OUT / "128B0_TEST_CONTRACT_AUDIT.csv", index=False)
    pd.DataFrame(det_rows).to_csv(OUT / "128B0_DETERMINISM_AUDIT.csv", index=False)
    pd.DataFrame(status_rows).to_csv(OUT / "128B0_ADAPTER_STATUS.csv", index=False)

    bag_cfg = fitted.get("PU-BaggingDT", (None, {}))[1]
    spy_cfg = fitted.get("Spy-PU+BRF", (None, {}))[1]
    pull_cfg = fitted.get("PU-pullbaggingDT", (None, {}))[1]
    jwrite("128B0_PU_BAGGING_IMPLEMENTATION.json", {
        "implementation_status": "PAPER_FAITHFUL_REIMPLEMENTATION",
        "authority": "Wu et al. IEEE GRSL; 128A3 frozen specification; Ouyang author reproduction as corroboration only",
        "official_code": False, "fixed_parameters": {"formal_T": 1000, "temporary_U_fraction": 0.70,
            "tree": "DecisionTreeClassifier(gini,max_depth=None,max_features=None,class_weight=balanced)"},
        "smoke_parameters": {"T": 50}, "search_space": "single predeclared smoke config",
        "adaptations": ["RAW-792", "frozen folds", "controls passed as U"],
        "selected_config": bag_cfg, "unresolved_ambiguity": "",
    })
    jwrite("128B0_SPY_PU_BRF_IMPLEMENTATION.json", {
        "implementation_status": "PAPER_INFORMED_BENCHMARK_ADAPTATION",
        "authority": "Fu et al. 2024 IEEE JSTARS", "official_code": False,
        "fixed_parameters": {"spy_fraction": 0.15, "stage1": "RandomForestClassifier",
                             "stage2": "imbalanced-learn BalancedRandomForestClassifier"},
        "smoke_parameters": {"stage1_trees": 50, "BRF_trees": 50},
        "search_space": threshold_space["candidate_rules"],
        "adaptations": ["RAW-792", "fit-only RN mining", "validation-only threshold selection"],
        "selected_config": spy_cfg,
        "unresolved_ambiguity": "RN threshold is explicitly an implementation choice selected only on validation",
    })
    jwrite("128B0_PU_PULL_IMPLEMENTATION.json", {
        "implementation_status": "OFFICIAL_CODE_INFORMED_BENCHMARK_ADAPTATION",
        "authority": "unmodified ShubingOuyangcug/PU-pullbaggingDT vendor files",
        "official_code": True, "repository": "https://github.com/ShubingOuyangcug/PU-pullbaggingDT",
        "commit": commit,
        "fixed_parameters": {"native_hidden_embedding": "50-100-50-50", "optimizer": "Adam",
            "learning_rate": 1e-4, "official_epochs": 60, "loss": "released P-P attraction / P-U denominator",
            "fusion": "equal arithmetic mean of contrastive and PU-BaggingDT scores"},
        "smoke_parameters": {"epochs": 3, "PU_trees": 20},
        "search_space": "single predeclared engineering config; no test selection",
        "adaptations": ["first Linear input only: 25 to 792; hidden/embedding widths unchanged",
            "in-memory row loader", "fit-fold-only StandardScaler and contrastive min-max",
            "single deterministic encoder instead of manual closest/farthest checkpoint curation",
            "validation interface", "scalar row score interface"],
        "selected_config": pull_cfg,
        "unresolved_ambiguity": "The single-encoder deterministic checkpoint adaptation is not an exact reproduction of the paper's manual closest/farthest ensemble.",
    })

    all_status = pd.DataFrame(status_rows)
    ready = bool(len(all_status) == 3 and all_status.ready_for_full_5fold.all()
                 and data.audit["pass"] and len(test_rows) == 3
                 and all(r["pass"] for r in test_rows))
    if not ready and not blockers:
        blockers.append("One or more determinism or test-contract checks failed.")
    decision = {
        "status": "PASS_128B0_ADAPTER_SMOKE_TEST_COMPLETE" if ready else "BLOCKED_128B0",
        "ready_for_full_5fold": ready, "outer_test_metrics_computed": False,
        "outer_fold": 1, "validation_fold": 2, "fit_folds": [3, 4, 5],
        "blocking_reasons": blockers,
        "next_step": "RUN_128B1_FULL_5FOLD_PUBLISHED_ALGORITHM_COMPARISON" if ready
                     else "RESOLVE_128B0_BLOCKERS_BEFORE_128B1",
    }
    jwrite("128B0_READY_FOR_128B1.json", decision)

    log(decision["status"])
    log("OUTER_FOLD=1 VALIDATION_FOLD=2 FIT_FOLDS=3,4,5 RAW_DIM=792")
    log("P_FIT=3034 U_FIT=6068 P_VAL=1011 U_VAL=2022 TEST_SETS=1011 TEST_ROWS=3033")
    for row in status_rows:
        log(f"{row['method']}_ADAPTER={'PASS' if row['ready_for_full_5fold'] else 'BLOCKED'}")
    log("TEST_METRICS_COMPUTED=False LEAKAGE=0")
    log(f"READY_FOR_128B1={ready}")


if __name__ == "__main__":
    main()
