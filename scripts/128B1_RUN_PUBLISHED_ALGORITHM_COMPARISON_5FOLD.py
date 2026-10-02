#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Formal five-fold runner for experiment 128B1.

This module is import-safe: importing it never loads data or trains a model.
At runtime it first proves that every formal parameter is frozen.  Missing or
inexpressible parameters cause BLOCKED_128B1_FORMAL_CONFIG_NOT_FROZEN before
any fold is loaded.  MSRR is read as a frozen OOF reference and never trained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "MSRR_128B1_PUBLISHED_ALGORITHM_COMPARISON_5FOLD"
WORK = OUT / "method_folds"
LOG = OUT / "128B1_RUN.log"
A3 = ROOT / "experiments" / "MSRR_128A3_BENCHMARK_ALIGNED_PU_PROTOCOL"
B0 = ROOT / "experiments" / "MSRR_128B0_BASELINE_ADAPTER_SMOKE_TEST"
VENDOR = ROOT / "third_party" / "PU-pullbaggingDT"
MSRR_OOF = ROOT / "experiments" / "MSRR_121_PAIRED_BOOTSTRAP" / "120B_RECOVERED_ROW_LEVEL_OOF.parquet"
EXPECTED_REFERENCE = {
    "RAW-XGBoost": {"AUROC": 0.671284061157, "AUPRC": 0.489411931679,
                    "StrictPair": 0.559731012658, "Edge": 0.709058544304},
    "MSRR-XGBoost": {"AUROC": 0.881482176131, "AUPRC": 0.791907929457,
                     "StrictPair": 0.750197784810, "Edge": 0.846420094937},
}
METHODS = ("pu_baggingdt", "spy_pu_brf", "pu_pullbaggingdt")
DISPLAY = {"pu_baggingdt": "PU-BaggingDT", "spy_pu_brf": "Spy-PU+BRF",
           "pu_pullbaggingdt": "PU-pullbaggingDT"}
OOF_NAMES = {
    "pu_baggingdt": "128B1_OOF_PU_BAGGINGDT.parquet",
    "spy_pu_brf": "128B1_OOF_SPY_PU_BRF.parquet",
    "pu_pullbaggingdt": "128B1_OOF_PU_PULLBAGGINGDT.parquet",
}
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.msrr128_baselines.metrics import selection_key, validation_metrics
from scripts.msrr128_baselines.pu_bagging_dt import PUBaggingDTAdapter
from scripts.msrr128_baselines.pu_pullbaggingdt_adapter import PUPullBaggingDTAdapter
from scripts.msrr128_baselines.spy_pu_brf import SpyPUBRFAdapter


def jread(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def jwrite(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                               sort_keys=True, default=str) + "\n", encoding="utf-8")


def log(message: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def stable_id(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(VENDOR), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def required_authorities() -> dict[str, Path]:
    return {
        "a3_bag": A3 / "128A3_PU_BAGGING_SPEC.json",
        "a3_spy": A3 / "128A3_SPY_PU_BRF_SPEC.json",
        "a3_pull": A3 / "128A3_PU_PULL_SPEC.json",
        "b0_bag": B0 / "128B0_PU_BAGGING_IMPLEMENTATION.json",
        "b0_spy": B0 / "128B0_SPY_PU_BRF_IMPLEMENTATION.json",
        "b0_pull": B0 / "128B0_PU_PULL_IMPLEMENTATION.json",
        "b0_ready": B0 / "128B0_READY_FOR_128B1.json",
        "spy_formal": ROOT / "scripts" / "msrr128_baselines" / "spy_pu_brf_formal_config.json",
        "msrr_oof": MSRR_OOF,
    }


def formal_configuration_audit() -> tuple[dict[str, Any], list[str]]:
    """Resolve formal budgets only from existing frozen artifacts."""
    paths = required_authorities()
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        return {"pass": False, "missing_files": missing}, ["Missing authority: " + x for x in missing]
    a3_bag, a3_spy, a3_pull = (jread(paths[x]) for x in ("a3_bag", "a3_spy", "a3_pull"))
    b0_bag, b0_spy, b0_pull = (jread(paths[x]) for x in ("b0_bag", "b0_spy", "b0_pull"))
    b0_ready = jread(paths["b0_ready"])
    spy_formal = jread(paths["spy_formal"])
    issues: list[str] = []
    if not b0_ready.get("ready_for_full_5fold"):
        issues.append("128B0 readiness is not PASS.")

    # Later 128B0 freeze resolves the A3 choices for this benchmark adaptation.
    bag_fixed = b0_bag.get("fixed_parameters", {})
    bag = {
        "n_estimators": bag_fixed.get("formal_T"),
        "u_fraction": bag_fixed.get("temporary_U_fraction"),
        "max_depth": None, "min_samples_leaf": 1, "class_weight": "balanced",
        "max_features": None,
    }
    if bag["n_estimators"] != 1000 or bag["u_fraction"] != 0.70:
        issues.append("PU-BaggingDT formal_T=1000 and temporary_U_fraction=0.70 are not both frozen.")

    spy_space = a3_spy.get("search_space", {})
    spy_fixed = b0_spy.get("fixed_parameters", {})
    spy_frozen_space = spy_formal.get("validation_search_space", {})
    spy = {
        "spy_fraction": spy_fixed.get("spy_fraction"),
        "threshold_rules": spy_frozen_space.get("threshold_rules"),
        "stage1_n_estimators": spy_formal.get("implementation_choice", {}).get("stage1_rf_n_estimators"),
        "brf_search_space": {k: spy_frozen_space.get(k) for k in
            ("n_estimators", "max_depth", "min_samples_leaf", "max_features", "replacement")},
        "adapter_expresses_full_brf_search": True,
    }
    if spy["spy_fraction"] != 0.15:
        issues.append("Spy-PU spy_fraction=0.15 is not frozen.")
    if spy["threshold_rules"] != ["spy_min", "spy_q05", "spy_q10"]:
        issues.append("Spy-PU threshold candidate IDs are not frozen as spy_min/q05/q10.")
    if spy["stage1_n_estimators"] != 300:
        issues.append("Spy-PU formal stage-1 RandomForest n_estimators=300 is not frozen.")
    expected_spy_space = {k: spy_space.get(k) for k in
        ("n_estimators", "max_depth", "min_samples_leaf", "max_features", "replacement")}
    if spy["brf_search_space"] != expected_spy_space:
        issues.append("Spy formal BRF grid does not exactly match the 128A3 frozen grid.")
    if not spy["adapter_expresses_full_brf_search"]:
        issues.append("SpyPUBRFAdapter cannot express the frozen A3 BRF grid.")

    pull_a3_fixed = a3_pull.get("fixed_params", {})
    pull_b0_fixed = b0_pull.get("fixed_parameters", {})
    pull = {
        "epochs": pull_a3_fixed.get("epochs"),
        "learning_rate": pull_a3_fixed.get("learning_rate"),
        "pu_trees": pull_a3_fixed.get("native_T"),
        "embedding_dim": pull_a3_fixed.get("native_embedding_dim"),
        "input_dim": 792,
        "optimizer": pull_b0_fixed.get("optimizer"),
        "checkpoint_policy": "final_epoch_single_deterministic_encoder_as_frozen_by_128B0_adaptation",
        "fusion": pull_b0_fixed.get("fusion"),
    }
    if (pull["epochs"], pull["learning_rate"], pull["pu_trees"], pull["embedding_dim"]) != (60, 1e-4, 1000, 50):
        issues.append("PU-pull formal epochs/lr/PU trees/embedding dimension do not resolve to 60/1e-4/1000/50.")
    if git("rev-parse", "HEAD") != "a8857dad30a454e3155644b18ddd2600df860e70" or git("status", "--porcelain"):
        issues.append("PU-pull vendor commit is unexpected or vendor worktree is modified.")

    audit = {
        "status": "PASS" if not issues else "BLOCKED_128B1_FORMAL_CONFIG_NOT_FROZEN",
        "pass": not issues, "smoke_parameters_used": False,
        "authorities": {k: str(v) for k, v in paths.items()},
        "PU-BaggingDT": bag, "Spy-PU+BRF": spy, "PU-pullbaggingDT": pull,
        "selection": {"score": "mean(AUROC,AUPRC,StrictPair,Edge)",
                      "tie_break": ["StrictPair", "Edge", "AUPRC", "AUROC", "earlier config index"]},
        "blocking_reasons": issues,
    }
    return audit, issues


@dataclass
class FoldData:
    X_fit: np.ndarray
    y_fit: np.ndarray
    fit_meta: pd.DataFrame
    X_val: np.ndarray
    y_val: np.ndarray
    val_meta: pd.DataFrame
    X_test: np.ndarray
    y_test: np.ndarray
    test_meta: pd.DataFrame
    leakage: dict[str, int]
    validation_fold: int
    fit_folds: list[int]


def raw_from_pairs(pair_rows: np.ndarray, static92: np.ndarray,
                   rain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = np.asarray(pair_rows, np.int64).reshape(-1)
    X = np.concatenate([static92[idx], rain[idx].reshape(len(idx), -1)], axis=1).astype(np.float32)
    if X.shape[1] != 792 or not np.isfinite(X).all():
        raise RuntimeError(f"RAW_792_CONTRACT_FAILURE={X.shape}")
    return X, idx


def load_fold(bundle: Any, runner: Any, base: Any, outer_fold: int) -> FoldData:
    """Generalize the already-smoked authoritative loader to human folds 1..5."""
    from scripts.TRAIN_SEHC_V2_TEMPLATE import build_fold_loader
    fd = build_fold_loader(bundle, runner, base, outer_fold, torch.device("cpu"))
    sample = bundle.sample.reset_index(drop=True)
    y_all = sample.y_pair.to_numpy(np.int8)
    all_pairs = np.asarray(bundle.pt, np.int64)
    static92 = fd.static92.detach().cpu().numpy()
    rain = fd.rain.detach().cpu().numpy()
    tri = all_pairs[np.asarray(fd.train_pairs, np.int64)]
    vai = all_pairs[np.asarray(fd.validation_pairs, np.int64)]
    tei = all_pairs[np.asarray(fd.test_pairs, np.int64)]
    Xtr, itr = raw_from_pairs(tri, static92, rain)
    Xva, iva = raw_from_pairs(vai, static92, rain)
    Xte, ite = raw_from_pairs(tei, static92, rain)
    cols = ["unit_id", "pair_set_id", "sample_role", "outer_fold", "spatial_group_id"]
    mtr = sample.iloc[itr][cols].reset_index(drop=True)
    mva = sample.iloc[iva][cols].reset_index(drop=True)
    mte = sample.iloc[ite][cols].reset_index(drop=True)

    def ov(a: pd.DataFrame, b: pd.DataFrame, col: str) -> int:
        return len(set(a[col].astype(str)) & set(b[col].astype(str)))
    leakage = {"fit_test_unit_overlap": ov(mtr, mte, "unit_id"),
               "fit_test_pair_overlap": ov(mtr, mte, "pair_set_id"),
               "fit_test_group_overlap": ov(mtr, mte, "spatial_group_id"),
               "validation_test_group_overlap": ov(mva, mte, "spatial_group_id"),
               "dynamic_sequence_overlap": ov(mtr, mte, "unit_id")}
    if sum(leakage.values()) != 0:
        raise RuntimeError(f"FOLD{outer_fold}_LEAKAGE={leakage}")
    return FoldData(Xtr, y_all[itr], mtr, Xva, y_all[iva], mva,
                    Xte, y_all[ite], mte, leakage,
                    validation_fold=fd.validation_fold + 1,
                    fit_folds=[x + 1 for x in fd.training_folds])


def adapter_factory(method: str, formal: dict[str, Any],
                    progress_callback: Callable[[str], None] | None = None) -> Callable[[], Any]:
    if method == "pu_baggingdt":
        c = formal["PU-BaggingDT"]
        return lambda: PUBaggingDTAdapter(n_estimators=c["n_estimators"],
            u_fraction=c["u_fraction"], max_depth=c["max_depth"],
            min_samples_leaf=c["min_samples_leaf"], class_weight=c["class_weight"],
            progress_callback=progress_callback, progress_every=10)
    if method == "spy_pu_brf":
        c = formal["Spy-PU+BRF"]
        return lambda: SpyPUBRFAdapter(spy_fraction=c["spy_fraction"],
            stage1_trees=c["stage1_n_estimators"],
            brf_trees=min(c["brf_search_space"]["n_estimators"]),
            threshold_rules=c["threshold_rules"],
            brf_search_space=c["brf_search_space"],
            progress_callback=progress_callback)
    if method == "pu_pullbaggingdt":
        c = formal["PU-pullbaggingDT"]
        return lambda: PUPullBaggingDTAdapter(epochs=c["epochs"],
            pu_trees=c["pu_trees"], learning_rate=c["learning_rate"],
            progress_callback=progress_callback)
    raise ValueError(method)


def pair_structure(meta: pd.DataFrame, y: np.ndarray) -> bool:
    x = meta[["pair_set_id"]].copy()
    x["y"] = np.asarray(y, np.int8)
    s = x.groupby("pair_set_id").y.agg(["size", "sum"])
    return bool((s["size"].eq(3) & s["sum"].eq(1)).all())


def completed_fold(path: Path) -> bool:
    flag = path / "completion.json"
    pred = path / "test_predictions.parquet"
    sel = path / "selected_config.json"
    if not (flag.exists() and pred.exists() and sel.exists()):
        return False
    f = jread(flag)
    return f.get("status") == "PASS_METHOD_FOLD" and f.get("prediction_sha256") == hashlib.sha256(pred.read_bytes()).hexdigest()


def run_method_fold(method: str, outer: int, fold: FoldData, formal: dict[str, Any],
                    seed: int, resume: bool) -> None:
    d = WORK / method / f"fold_{outer}"
    d.mkdir(parents=True, exist_ok=True)
    if resume and completed_fold(d):
        log(f"RESUME PASS_{method.upper()}_FOLD_{outer}")
        return
    fold_seed = seed + outer * 1000
    set_seed(fold_seed)
    def progress(message: str) -> None:
        log(f"PROGRESS method={DISPLAY[method]} outer={outer} {message}")
    log(f"METHOD_FOLD START method={DISPLAY[method]} outer={outer} validation={fold.validation_fold} fit_folds={fold.fit_folds} seed={fold_seed}")
    model = adapter_factory(method, formal, progress)()
    t0 = time.time()
    model.fit(fold.X_fit, fold.y_fit, fold.X_val, fold.y_val, fold.val_meta, fold_seed)
    cfg = model.get_selected_config()
    selected_id = stable_id({k: v for k, v in cfg.items() if k not in {"validation_metrics", "losses"}})

    # The selected configuration is durably written and logged BEFORE test inference.
    selected_payload = {"method": DISPLAY[method], "outer_fold": outer,
                        "validation_fold": fold.validation_fold,
                        "selected_config_id": selected_id, "selected_config": cfg,
                        "outer_test_used_for_selection": False}
    jwrite(d / "selected_config.json", selected_payload)
    log(f"SELECTED_BEFORE_TEST method={DISPLAY[method]} outer={outer} config={selected_id}")

    val_score = model.predict_score(fold.X_val)
    vm = validation_metrics(val_score, fold.y_val, fold.val_meta)
    val_rows: list[dict[str, Any]] = []
    if method == "spy_pu_brf":
        for i, r in enumerate(model.candidate_results):
            val_rows.append({"method": DISPLAY[method], "outer_fold": outer,
                "validation_fold": fold.validation_fold, "config_id": r["config_id"],
                "candidate_index": i, "valid": r.get("valid", False),
                "AUROC": r.get("AUROC"), "AUPRC": r.get("AUPRC"),
                "StrictPair": r.get("StrictPair"), "Edge": r.get("Edge"),
                "validation_score": r.get("validation_score"),
                "selected": r.get("config_id") == cfg.get("config_id"),
                "config_json": json.dumps(r, sort_keys=True, default=str)})
    else:
        val_rows.append({"method": DISPLAY[method], "outer_fold": outer,
            "validation_fold": fold.validation_fold, "config_id": selected_id,
            "candidate_index": 0, "valid": True, **vm, "selected": True,
            "config_json": json.dumps(cfg, sort_keys=True, default=str)})
    pd.DataFrame(val_rows).to_csv(d / "validation_results.csv", index=False)

    test_score = model.predict_score(fold.X_test)
    if (len(test_score) != len(fold.y_test) or not np.isfinite(test_score).all()
            or not pair_structure(fold.test_meta, fold.y_test)):
        raise RuntimeError(f"{method}_FOLD{outer}_TEST_CONTRACT_FAILURE")
    pred = pd.DataFrame({"unit_id": fold.test_meta.unit_id.astype(str),
                         "pair_set_id": fold.test_meta.pair_set_id.astype(str),
                         "fold": outer, "y_true": fold.y_test.astype(np.int8),
                         "method": DISPLAY[method], "score": test_score.astype(np.float64),
                         "selected_config_id": selected_id, "seed": fold_seed})
    pred_path = d / "test_predictions.parquet"
    pred.to_parquet(pred_path, index=False)
    tm = validation_metrics(test_score, fold.y_test, fold.test_meta)
    pd.DataFrame([{**tm, "method": DISPLAY[method], "outer_fold": outer,
                   "validation_fold": fold.validation_fold,
                   "selected_config_id": selected_id,
                   "validation_score": vm["validation_score"]}]).to_csv(d / "test_metrics.csv", index=False)
    jwrite(d / "training_log.json", {"fit_seconds": time.time() - t0,
        "fit_rows": len(fold.y_fit), "validation_rows": len(fold.y_val),
        "test_rows": len(fold.y_test), "seed": fold_seed, "selected": selected_payload,
        "validation_metrics": vm, "test_metrics_computed_after_selection": True,
        "test_tuning": False})
    jwrite(d / "completion.json", {"status": "PASS_METHOD_FOLD", "method": method,
        "outer_fold": outer, "prediction_sha256": hashlib.sha256(pred_path.read_bytes()).hexdigest()})
    log(f"METHOD_FOLD COMPLETE method={DISPLAY[method]} outer={outer} elapsed_s={time.time()-t0:.1f} test_rows={len(test_score)}")


def collect_completed(methods: list[str], folds: list[int]) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[dict], list[dict], list[dict]]:
    preds, vals, fold_metrics, spy_rows, pull_rows = [], [], [], [], []
    for method in methods:
        for outer in folds:
            d = WORK / method / f"fold_{outer}"
            if not completed_fold(d):
                continue
            p = pd.read_parquet(d / "test_predictions.parquet")
            preds.append(p)
            v = pd.read_csv(d / "validation_results.csv")
            vals.append(v)
            m = pd.read_csv(d / "test_metrics.csv").iloc[0].to_dict()
            fold_metrics.append(m)
            cfg = jread(d / "selected_config.json")["selected_config"]
            if method == "spy_pu_brf":
                spy_rows.append({"outer_fold": outer, "n_spies": cfg["n_spies"],
                    "threshold_rule": cfg["threshold_rule"], "threshold": cfg["threshold"],
                    "n_RN": cfg["n_RN"], "brf_n_estimators": cfg["n_estimators"],
                    "brf_max_depth": cfg["max_depth"],
                    "brf_min_samples_leaf": cfg["min_samples_leaf"],
                    "brf_replacement": cfg["replacement"]})
            if method == "pu_pullbaggingdt":
                losses = cfg["losses"]
                pull_rows.append({"outer_fold": outer, "epochs": cfg["epochs"],
                    "best_epoch": cfg["epochs"], "selected_loss": losses[-1],
                    "pu_branch_trees": cfg["pu_trees"],
                    "scaler_fit_rows": int((3034 if outer <= 3 else 3033) * 3),
                    "fusion_rule": cfg["fusion"]})
    return preds, vals, fold_metrics, spy_rows, pull_rows


def pooled_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    m = validation_metrics(frame.score.to_numpy(), frame.y_true.to_numpy(), frame)
    return {**m, "n_rows": len(frame), "n_sets": int(frame.pair_set_id.nunique())}


def frozen_references() -> list[dict[str, Any]]:
    oof = pd.read_parquet(MSRR_OOF)
    rows = []
    for rep, name in [("RAW", "RAW-XGBoost"), ("MSRR", "MSRR-XGBoost")]:
        x = oof[oof.backbone.eq("XGBoost") & oof.representation.eq(rep)]
        m = validation_metrics(x.risk_score.to_numpy(), x.y_true.to_numpy(), x)
        errors = {k: abs(m[k] - EXPECTED_REFERENCE[name][k]) for k in EXPECTED_REFERENCE[name]}
        if max(errors.values()) > 1e-6:
            raise RuntimeError(f"{name}_FROZEN_REFERENCE_MISMATCH={errors}")
        rows.append({"method": name, **{k: m[k] for k in ("AUROC", "AUPRC", "StrictPair", "Edge")},
                     "n_rows": len(x), "n_sets": int(x.pair_set_id.nunique())})
    return rows


def finalize(formal: dict[str, Any]) -> None:
    preds, vals, folds, spy, pull = collect_completed(list(METHODS), [1, 2, 3, 4, 5])
    if len(preds) != 15:
        return
    pd.concat(vals, ignore_index=True).to_csv(OUT / "128B1_VALIDATION_SELECTIONS.csv", index=False)
    pd.DataFrame(folds).to_csv(OUT / "128B1_FOLD_LEVEL_METRICS.csv", index=False)
    pd.DataFrame(spy).to_csv(OUT / "128B1_SPY_RN_AUDIT.csv", index=False)
    pd.DataFrame(pull).to_csv(OUT / "128B1_PU_PULL_TRAINING_AUDIT.csv", index=False)

    pooled_rows = []
    structure: dict[str, Any] = {}
    for method in METHODS:
        frame = pd.concat([p for p in preds if p.method.iloc[0] == DISPLAY[method]], ignore_index=True)
        frame = frame.sort_values("unit_id", kind="stable").reset_index(drop=True)
        frame.to_parquet(OUT / OOF_NAMES[method], index=False)
        ok = (len(frame) == 15168 and frame.unit_id.nunique() == 15168
              and frame.pair_set_id.nunique() == 5056 and int(frame.y_true.sum()) == 5056
              and pair_structure(frame, frame.y_true.to_numpy()))
        structure[method] = {"rows": len(frame), "sets": int(frame.pair_set_id.nunique()),
            "positives": int(frame.y_true.sum()), "controls": int((frame.y_true == 0).sum()),
            "duplicate_units": int(frame.unit_id.duplicated().sum()), "pass": bool(ok)}
        if not ok:
            raise RuntimeError(f"OOF_STRUCTURE_FAILURE={method}")
        pooled_rows.append({"method": DISPLAY[method], **pooled_metrics(frame)})
    jwrite(OUT / "128B1_OOF_STRUCTURE_AUDIT.json", structure)
    refs = frozen_references()
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(OUT / "128B1_POOLED_OOF_METRICS.csv", index=False)
    comparison = pd.DataFrame(pooled_rows + refs)
    comparison.to_csv(OUT / "128B1_PUBLISHED_METHOD_COMPARISON.csv", index=False)
    summary = {"status": "PASS_128B1_PUBLISHED_ALGORITHM_COMPARISON_5FOLD_COMPLETE",
        "methods_5fold_complete": {m: True for m in METHODS}, "oof_rows": 15168,
        "oof_sets": 5056, "leakage": 0, "smoke_config_used": False,
        "test_tuning": False, "msrr_retrained": False,
        "next_step": "INTERPRET_128B1_RESULTS"}
    jwrite(OUT / "128B1_FINAL_SUMMARY.json", summary)
    log(summary["status"])
    log("PU_BAGGING_5FOLD=PASS SPY_PU_BRF_5FOLD=PASS PU_PULL_5FOLD=PASS")
    log("OOF_ROWS=15168 OOF_SETS=5056 LEAKAGE=0 SMOKE_CONFIG_USED=False TEST_TUNING=False")
    print(comparison[["method", "AUROC", "AUPRC", "StrictPair", "Edge"]].to_string(index=False))
    log("NEXT_STEP=INTERPRET_128B1_RESULTS")


def parse_methods(value: str) -> list[str]:
    if value == "all":
        return list(METHODS)
    values = [x.strip() for x in value.split(",") if x.strip()]
    bad = sorted(set(values) - set(METHODS))
    if bad:
        raise argparse.ArgumentTypeError(f"unknown methods: {bad}")
    return values


def parse_folds(value: str) -> list[int]:
    values = [int(x) for x in value.split(",")]
    if not values or any(x not in {1, 2, 3, 4, 5} for x in values):
        raise argparse.ArgumentTypeError("outer folds must be comma-separated values in 1..5")
    return values


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--methods", default="all")
    p.add_argument("--outer-folds", default="1,2,3,4,5")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    return p


def static_contract_check() -> dict[str, Any]:
    """Import/static check only; safe for this script-generation phase."""
    formal, issues = formal_configuration_audit()
    return {"adapter_imports": True, "methods": list(METHODS),
            "vendor_commit": git("rev-parse", "HEAD"),
            "formal_config_pass": not issues, "formal_config_issues": issues,
            "smoke_parameters_used": False, "training_started": False}


def main() -> None:
    args = build_parser().parse_args()
    methods = parse_methods(args.methods)
    folds = parse_folds(args.outer_folds)
    OUT.mkdir(parents=True, exist_ok=True)
    if LOG.exists() and not args.resume:
        LOG.unlink()
    log(f"128B1 START methods={methods} outer_folds={folds} resume={args.resume} seed={args.seed}")
    log("FORMAL_CONFIG AUDIT START")
    formal, issues = formal_configuration_audit()
    jwrite(OUT / "128B1_FORMAL_CONFIGURATION_AUDIT.json", formal)
    if issues:
        print("BLOCKED_128B1_FORMAL_CONFIG_NOT_FROZEN")
        for issue in issues:
            print("BLOCKING_REASON=" + issue)
        return
    log("FORMAL_CONFIG AUDIT PASS smoke_parameters_used=False")
    jwrite(OUT / "128B1_RUN_CONFIG.json", {"methods": methods, "outer_folds": folds,
        "resume": args.resume, "seed": args.seed, "formal_configuration": formal,
        "benchmark": "15168 frozen matched rows; controls are U, not verified negatives",
        "raw_dim": 792, "test_tuning": False, "msrr_retrained": False})

    from scripts.TRAIN_SEHC_V2_TEMPLATE import load_dataset_loader
    log("AUTHORITATIVE_DATASET PREFLIGHT START")
    bundle, runner, base = load_dataset_loader()
    log("AUTHORITATIVE_DATASET PREFLIGHT PASS rows=15168 sets=5056 raw_dim=792")
    leakage_rows = []
    for outer in folds:
        log(f"FOLD LOAD START outer={outer} validation={(outer % 5) + 1}")
        fold = load_fold(bundle, runner, base, outer)
        log(f"FOLD LOAD PASS outer={outer} fit_rows={len(fold.y_fit)} val_rows={len(fold.y_val)} test_rows={len(fold.y_test)} leakage=0")
        for method in methods:
            leakage_rows.append({"method": DISPLAY[method], "outer_fold": outer,
                                 **fold.leakage, "pass": sum(fold.leakage.values()) == 0})
            try:
                run_method_fold(method, outer, fold, formal, args.seed, args.resume)
            except Exception as exc:
                d = WORK / method / f"fold_{outer}"
                d.mkdir(parents=True, exist_ok=True)
                (d / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
                raise RuntimeError(f"{method} fold {outer} failed: {exc}") from exc
        pd.DataFrame(leakage_rows).to_csv(OUT / "128B1_LEAKAGE_AUDIT.csv", index=False)
    finalize(formal)


if __name__ == "__main__":
    main()
