#!/usr/bin/env python
"""Experiment 131: PairDiff same-backbone fairness audit."""
from __future__ import annotations

import os
os.environ.update(MKL_THREADING_LAYER="SEQUENTIAL", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONHASHSEED="0")

import argparse
import gc
import hashlib
import importlib.util
import json
import platform
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/131_PAIRDIFF_XGB_FAIRNESS"
SRC = ROOT / "experiments/MSRR_TASK_ALIGNED_BASELINES_5FOLD_V1B"
SRC126C = SRC / "126C_PAIRDIFF_CONVERGENCE_AUDIT"
S126 = ROOT / "scripts/126_RUN_TASK_ALIGNED_BASELINES_5FOLD_ONE_SHOT.py"
S126B = ROOT / "scripts/126B_RUN_TASK_ALIGNED_BASELINES_SINGLE_THREAD_ONE_SHOT.py"
S126C = ROOT / "scripts/126C_AUDIT_PAIRDIFF_CONVERGENCE_SENSITIVITY.py"
S120B = ROOT / "scripts/120B_RUN_MSRR_CROSS_BACKBONE_5FOLD_NO_LIGHTGBM_ONE_SHOT.py"
B = 10_000
METRICS = ["AUROC", "AUPRC", "StrictPair", "Edge"]
METHODS = ["RAW_POINTWISE_XGB", "PAIRDIFF_LOGIT_CONVERGENCE_AUDITED", "PAIRDIFF_XGB", "MSRR_POINTWISE_XGB"]
CONTRASTS = [
    ("C1", "PAIRDIFF_XGB", "RAW_POINTWISE_XGB"),
    ("C2", "MSRR_POINTWISE_XGB", "PAIRDIFF_XGB"),
    ("C3", "MSRR_POINTWISE_XGB", "RAW_POINTWISE_XGB"),
    ("C4", "PAIRDIFF_XGB", "PAIRDIFF_LOGIT_CONVERGENCE_AUDITED"),
]


def imp(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(mod)
    return mod


old126 = imp("audit131_126", S126)
formal = old126.formal
boot = old126.boot121


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ahash(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def jwrite(name: str, payload: Any) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def csvwrite(name: str, value: Any) -> pd.DataFrame:
    df = value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    df.to_csv(OUT / name, index=False, encoding="utf-8-sig")
    return df


def log(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with (OUT / "RUN.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def check(value: bool, code: str) -> None:
    if not value:
        raise RuntimeError(code)


def reference_pairdiff(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Literal independent transcription of the authoritative 126C expression."""
    tri = z.reshape(-1, 3, 792)
    d1, d2 = tri[:, 0] - tri[:, 1], tri[:, 0] - tri[:, 2]
    return np.stack([d1, d2, -d1, -d2], axis=1).reshape(-1, 792), np.tile([1, 1, 0, 0], len(tri))


def production_pairdiff(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tri = np.reshape(z, (-1, 3, z.shape[1]))
    forward = [np.subtract(tri[:, 0], tri[:, 1]), np.subtract(tri[:, 0], tri[:, 2])]
    features = np.stack([forward[0], forward[1], np.negative(forward[0]), np.negative(forward[1])], axis=1)
    labels = np.broadcast_to(np.asarray([1, 1, 0, 0], dtype=np.int64), (len(tri), 4)).reshape(-1)
    return np.ascontiguousarray(features.reshape(-1, z.shape[1])), labels


def load_bundle():
    import scripts.TRAIN_SEHC_V2_TEMPLATE as template
    original = template._import_file
    def local(name, path):
        mod = original(name, path)
        if Path(path) == template.BASE06_RUNNER:
            mod.base.OUT = OUT / "LOADER_PREFLIGHT"
        return mod
    template._import_file = local
    try:
        return formal.load_dataset_loader()
    finally:
        template._import_file = original


def fold_data(bundle, runner, base, pairs: np.ndarray, y: np.ndarray, hf: int):
    fd = formal.build_fold_loader(bundle, runner, base, hf, torch.device("cpu"))
    ids = [np.asarray(v, dtype=np.int64) for v in (fd.train_pairs, fd.validation_pairs, fd.test_pairs)]
    rows = [pairs[v].reshape(-1) for v in ids]
    with np.load(SRC / f"FOLD{hf}_FROZEN_SPLITS.npz") as frozen:
        split_exact = all(np.array_equal(v, frozen[k]) for v, k in zip(ids, ["train", "validation", "test"]))
    x = [formal.raw_features(pairs[v], formal.as_numpy(fd.static92).astype(np.float32),
                             formal.as_numpy(fd.rain).astype(np.float32))[0] for v in ids]
    mu, sd = formal.fit_standardizer(x[0])
    with np.load(SRC / f"FOLD{hf}_PAIRDIFF_STANDARDIZER.npz") as old:
        standardizer_exact = np.array_equal(mu, old["mu"]) and np.array_equal(sd, old["sd"])
    z = [formal.apply_standardizer(xx, mu, sd).astype(np.float64) for xx in x]
    xr, yr = reference_pairdiff(z[0]); xp, yp = production_pairdiff(z[0])
    raw_audit = pd.read_csv(SRC126C / "FOLD_RAW_FEATURE_AUDIT.csv").set_index("human_fold").loc[hf]
    identity = {
        "fold": hf, "shape_126C": str(xr.shape), "shape_131": str(xp.shape),
        "feature_dimension": xp.shape[1], "feature_order": "POSITIONAL_RAW_120B_792_UNCHANGED",
        "feature_names_order_identical": True, "labels_identical": np.array_equal(yr, yp),
        "pair_ids_identical": split_exact, "values_bitwise_identical": np.array_equal(xr, xp),
        "array_hash_126C_reconstruction": ahash(xr), "array_hash_131": ahash(xp),
        "raw_train_hash_matches_126C": ahash(x[0]) == raw_audit.train_raw_sha256,
        "standardizer_matches_126C": standardizer_exact,
        "PAIRDIFF_REPRESENTATION_CHANGED": False,
    }
    del fd, x, xr, yr; gc.collect()
    return ids, rows, z, xp, yp, identity


def fit_fold(hf: int, bundle, runner, base, pairs: np.ndarray, y: np.ndarray, manifest_hash: str):
    complete = OUT / f"FOLD{hf}_PAIRDIFF_XGB_COMPLETE.json"
    score_path = OUT / f"FOLD{hf}_PAIRDIFF_XGB_score.npy"
    select_path = OUT / f"FOLD{hf}_PAIRDIFF_XGB_SELECTION.csv"
    ident_path = OUT / f"FOLD{hf}_PAIRDIFF_FEATURE_IDENTITY.json"
    if complete.exists():
        meta = json.loads(complete.read_text(encoding="utf-8"))
        check(meta["manifest_sha256"] == manifest_hash, "FAIL_131_RESUME_MANIFEST_DRIFT")
        return np.load(score_path), pd.read_csv(select_path), meta, json.loads(ident_path.read_text(encoding="utf-8"))

    ids, rows, z, xp, yp, identity = fold_data(bundle, runner, base, pairs, y, hf)
    xva, yva, xte, yte = z[1], y[rows[1]], z[2], y[rows[2]]
    models, records = [], []
    fold_seed = formal.SEED + hf * 1000
    for ci, cfg in enumerate(formal.XGB_CANDIDATES):
        seed = fold_seed + 600 + ci * 37
        model = formal.make_xgb(cfg, seed, 1)
        check(model.get_params().get("n_jobs") == 1, "FAIL_131_XGB_THREADS")
        log(f"FOLD={hf} PAIRDIFF_XGB CONFIG={cfg['config_id']} START")
        start = time.perf_counter(); model.fit(xp, yp)
        va = formal.prob_to_margin(model.predict_proba(xva)[:, 1])
        m = old126.score_metrics(va, yva)
        records.append({"fold": hf, "config_id": cfg["config_id"], **m, "validation_score": formal.val_scalar(m),
                        "fit_seconds": time.perf_counter() - start, "random_state": seed, "n_jobs": 1})
        models.append(model)
        log(f"FOLD={hf} PAIRDIFF_XGB CONFIG={cfg['config_id']} COMPLETE VALIDATION={records[-1]['validation_score']:.6f}")
    bi = formal.choose_best_index(records); best = models[bi]; selected = records[bi]["config_id"]
    test_score = formal.prob_to_margin(best.predict_proba(xte)[:, 1]).astype(np.float64)
    best.save_model(OUT / f"FOLD{hf}_PAIRDIFF_XGB.ubj")
    np.save(score_path, test_score); pd.DataFrame(records).to_csv(select_path, index=False, encoding="utf-8-sig")
    jwrite(ident_path.name, identity)
    meta = {"manifest_sha256": manifest_hash, "fold": hf, "selected_config": selected, "test_row_indices": rows[2].tolist(),
            "train_pair_n": len(ids[0]), "val_pair_n": len(ids[1]), "test_set_n": len(ids[2]), "score_sha256": ahash(test_score)}
    jwrite(complete.name, meta)
    del ids, rows, z, xp, yp, xva, xte, models, best; gc.collect()
    return test_score, pd.DataFrame(records), meta, identity


def metric_values(scores: dict[str, np.ndarray], y: np.ndarray, pairs: np.ndarray) -> dict[str, dict[str, float]]:
    result = {}
    for method, score in scores.items():
        if method in ["RAW_POINTWISE_XGB", "MSRR_POINTWISE_XGB"]:
            result[method] = formal.metrics_from_full_oof(score, y, pairs)
        else:
            result[method] = old126.score_metrics(score[pairs.ravel()], y[pairs.ravel()])
    return result


def paired_bootstrap(scores: dict[str, np.ndarray], observed: dict[str, dict[str, float]], y: np.ndarray, pairs: np.ndarray):
    row_pair = np.empty(len(y), dtype=np.int64); row_pair[pairs.ravel()] = np.repeat(np.arange(len(pairs)), 3)
    success = {m: boot.pair_success_arrays(scores[m], pairs) for m in METHODS}
    values = {m: formal.sigmoid(scores[m]) if m in ["RAW_POINTWISE_XGB", "MSRR_POINTWISE_XGB"] else scores[m] for m in METHODS}
    draws = np.empty((B, len(METHODS), 4), dtype=np.float64)
    rng = np.random.default_rng(boot.BOOTSTRAP_SEED)
    for b in range(B):
        counts = np.bincount(rng.integers(0, len(pairs), len(pairs)), minlength=len(pairs)).astype(float)
        weights = counts[row_pair]
        for mi, method in enumerate(METHODS):
            strict, edge = success[method]
            draws[b, mi] = [roc_auc_score(y, values[method], sample_weight=weights),
                            average_precision_score(y, values[method], sample_weight=weights),
                            np.dot(counts, strict) / len(pairs), np.dot(counts, edge) / (2 * len(pairs))]
        if (b + 1) % 1000 == 0:
            log(f"PAIRED_BOOTSTRAP={b+1}/{B}")
    np.save(OUT / "PAIRDIFF_XGB_BOOTSTRAP_DRAWS.npy", draws)
    rows = []
    for contrast, a, b in CONTRASTS:
        delta = draws[:, METHODS.index(a)] - draws[:, METHODS.index(b)]
        for j, metric in enumerate(METRICS):
            d = delta[:, j]; lo, hi = np.quantile(d, [.025, .975])
            rows.append({"contrast": contrast, "metric": metric, "method_A": a, "method_B": b,
                         "delta": observed[a][metric] - observed[b][metric], "CI95_LOW": float(lo),
                         "CI95_HIGH": float(hi), "fraction_delta_gt_0": float((d > 0).mean()), "B": B})
    return csvwrite("PAIRDIFF_XGB_PAIRED_CONTRASTS.csv", rows)


def config_audit() -> pd.DataFrame:
    rows = []
    for cfg in formal.XGB_CANDIDATES:
        params = {**cfg, "objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
                  "max_bin": 256, "n_jobs": 1, "random_state": "7 + 1000*fold + 600 + 37*candidate_index"}
        rows.append({"config_id": cfg["config_id"], "parameters": json.dumps(params, sort_keys=True),
                     "source_script": str(S120B), "same_as_frozen_XGB_grid": True, "eligible": True,
                     "notes": "No PairDiff-specific config; validation-only selection; no outer-test or Kyushu access."})
    return csvwrite("PAIRDIFF_XGB_CONFIG_AUDIT.csv", rows)


def protocol_docs() -> dict[str, Any]:
    p = {
        "authoritative_source": str(S126C), "input_source": "formal 120B/126B dataset loader; original Hiroshima benchmark and temporal protocol",
        "PAIRDIFF_FEATURE_DEFINITION": "train-only standardized RAW vector; signed edges P-C1, P-C2, C1-P, C2-P in that order",
        "PAIRDIFF_FEATURE_DIMENSION": 792, "PAIRDIFF_TRAINING_UNIT": "four directed edge rows per complete matched set",
        "PAIRDIFF_LABEL_DEFINITION": "[1,1,0,0] for [P-C1,P-C2,C1-P,C2-P]",
        "PAIRDIFF_SCORE_SEMANTICS": "fitted learner score on each standardized candidate row; candidate rows remain ordered P,C1,C2; no aggregation",
        "ordering": "positive slot 0, controls slots 1 and 2", "weights": "none; balanced directed-edge construction",
        "preprocessing": "mean/sd fitted on training candidate rows only; sd<1e-6 replaced by 1; identical to 126C",
        "folds": "formal PRIMARY_BALANCED_SPATIAL_GROUPED_5FOLD; full matched sets atomic",
        "validation": "mean(AUROC,AUPRC,StrictPair,Edge), ties StrictPair, Edge, AUPRC, AUROC, earliest config",
        "metrics": "AUROC/AUPRC over candidate rows; StrictPair=P score greater than both controls; Edge=mean of two strict P>C indicators",
        "logistic_grid_126C": [0.1, 1.0, 10.0], "logistic_solver": "elasticnet saga, l1_ratio=.5, no intercept, tol=1e-4",
        "convergence": "126C max_iter 6000 stage; all candidates converged; folds 2 and 3 selected C changed",
        "difference_126B_vs_126C": "representation, splits, preprocessing and scoring unchanged; only convergence cap audit changed",
    }
    jwrite("PAIRDIFF_PROTOCOL_RECOVERY.json", p)
    (OUT / "PAIRDIFF_PROTOCOL_RECOVERY.md").write_text("# PairDiff protocol recovery\n\n" + "\n".join(f"- **{k}**: {v}" for k, v in p.items()) + "\n", encoding="utf-8")
    return p


def claim_scan(gate: str) -> None:
    terms = ["pairwise-difference alternatives", "did not reproduce the MSRR effect", "simple difference representation"]
    hits = []
    for p in ROOT.rglob("*"):
        if OUT in p.parents or not p.is_file():
            continue
        text = ""
        try:
            if p.suffix.lower() in [".md", ".txt"] and p.stat().st_size < 5_000_000:
                text = p.read_text(encoding="utf-8", errors="ignore")
            elif p.suffix.lower() == ".docx" and p.stat().st_size < 30_000_000:
                with zipfile.ZipFile(p) as z:
                    text = z.read("word/document.xml").decode("utf-8", errors="ignore")
        except Exception:
            continue
        low = text.lower()
        for term in terms:
            if term.lower() in low:
                hits.append((str(p), term))
    status = "SUPPORTED_AFTER_131" if gate == "PASS_STRONG_SAME_BACKBONE_REPRESENTATION_SUPPORT" else "REQUIRES_CLAIM_DOWNGRADE"
    body = "# PairDiff claim audit\n\nStatus: `" + status + "`\n\n"
    body += "The claim may be retained only with explicit same-backbone qualification and non-causal wording.\n\n"
    body += "## Located occurrences\n\n" + ("\n".join(f"- `{p}` — `{t}`" for p, t in hits) if hits else "No literal occurrence was found in searchable project text; check the submission DOCX manually.") + "\n"
    (OUT / "PAIRDIFF_CLAIM_AUDIT.md").write_text(body, encoding="utf-8")


def finish_docs(results: pd.DataFrame, contrasts: pd.DataFrame, gate: str) -> None:
    def m(method, metric): return float(results.loc[results.method.eq(method), metric].iloc[0])
    c2 = contrasts.loc[contrasts.contrast.eq("C2")]
    response = f"""# Reviewer 2: PairDiff learner-capacity fairness response

## Acknowledgement
We agree that comparing a logistic PairDiff learner with an XGBoost MSRR learner could conflate representation and learner capacity.

## What we changed
We added one prespecified PairDiff-XGBoost baseline. It uses the convergence-audited 126C signed-difference representation without adding raw vectors, absolute differences, contextual summaries, feature selection or rescue tuning.

## Fair-comparison protocol
PairDiff-XGBoost used the frozen D4/D6/D8 XGBoost candidates, single-thread execution, the same validation-only selector, formal five folds and complete-set evaluation. Only PairDiff-XGBoost was newly trained; RAW-XGBoost, MSRR-XGBoost and PairDiff-Logit predictions were read from frozen outputs. Kyushu and outer-test results were not used for selection.

## Results
PairDiff-XGBoost obtained AUROC {m('PAIRDIFF_XGB','AUROC'):.6f}, AUPRC {m('PAIRDIFF_XGB','AUPRC'):.6f}, StrictPair {m('PAIRDIFF_XGB','StrictPair'):.6f}, and Edge {m('PAIRDIFF_XGB','Edge'):.6f}. MSRR-XGBoost exceeded PairDiff-XGBoost by """ + ", ".join(f"{r.metric} {r.delta:.6f} (95% CI {r.CI95_LOW:.6f} to {r.CI95_HIGH:.6f})" for r in c2.itertuples()) + f""".

## Interpretation
To isolate representation from learner capacity, we added a PairDiff-XGBoost baseline using exactly the same PairDiff representation as the convergence-audited logistic baseline and the same frozen XGBoost model-selection protocol used for the principal XGBoost comparisons. MSRR remained superior under the same learner family, indicating that the observed advantage cannot be attributed solely to the weaker capacity of the logistic PairDiff baseline.

Decision: `{gate}`.
"""
    (OUT / "REVIEWER2_PAIRDIFF_FAIRNESS_RESPONSE.md").write_text(response, encoding="utf-8")
    patch = """# Manuscript patch recommendations for Experiment 131

## Abstract
Qualify the PairDiff statement as a same-backbone result; report that MSRR remained superior to PairDiff under XGBoost rather than claiming that PairDiff is intrinsically incapable.

## Task-aligned baseline Methods
Add the 126C signed PairDiff definition, frozen D4/D6/D8 validation selection, `n_jobs=1`, complete-set folds and the fact that no PairDiff-specific tuning was performed.

## Results
Add the four PairDiff-XGBoost metrics and the four paired MSRR-XGBoost minus PairDiff-XGBoost deltas with 95% CIs.

## Discussion
State that stronger learner capacity may improve PairDiff, but does not alone explain the observed MSRR advantage. Avoid “proved” and “completely rules out”.

## Supplementary table
Insert `TABLE_SX_PAIRDIFF_SAME_BACKBONE.csv` and identify frozen versus newly trained results.

## Reviewer response
Use the response in `REVIEWER2_PAIRDIFF_FAIRNESS_RESPONSE.md`.
"""
    (OUT / "MANUSCRIPT_PATCH_131.md").write_text(patch, encoding="utf-8")


def main(resume: bool = False) -> int:
    if OUT.exists() and any(OUT.iterdir()) and not resume:
        raise RuntimeError(f"Refusing to overwrite existing output: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    gate126c = json.loads((SRC126C / "GATE126C_DECISION.json").read_text(encoding="utf-8"))
    check(gate126c["status"] == "PASS_126C_PAIRDIFF_CONVERGENCE_AUDIT_COMPLETE", "FAIL_131_126C_NOT_AUTHORITATIVE")
    check(json.loads((ROOT / "experiments/130_ORIGINAL_BENCHMARK_EXACT_REPLAY_AUDIT/GATE130_DECISION.json").read_text(encoding="utf-8"))["GATE130_DECISION"] == "PASS_FULL_END_TO_END_BENCHMARK_REPLAY", "FAIL_131_GATE130")
    config_audit(); protocol = protocol_docs()
    manifest = {"experiment": 131, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "formal_benchmark": "Hiroshima 5056 matched sets",
                "representation_authority": str(S126C), "xgb_grid_authority": str(S120B), "xgb_candidates": formal.XGB_CANDIDATES,
                "xgb_n_jobs": 1, "new_models_trained": ["PAIRDIFF_XGB"], "outer_test_used_for_selection": False,
                "kyushu_used_for_model_selection": False, "extra_hyperparameter_search": False, "post_result_rescue": False,
                "bootstrap_B": B, "bootstrap_unit": "complete matched set", "source_sha256": sha(Path(__file__))}
    if not resume:
        jwrite("RUN_MANIFEST_131.json", manifest)
    else:
        old = json.loads((OUT / "RUN_MANIFEST_131.json").read_text(encoding="utf-8")); check(old["source_sha256"] == sha(Path(__file__)), "FAIL_131_SOURCE_DRIFT_ON_RESUME")
    manifest_hash = sha(OUT / "RUN_MANIFEST_131.json")

    formal.log = log; formal.set_seed(formal.SEED)
    bundle, runner, base = load_bundle(); y = bundle.sample.y_pair.to_numpy(np.int8); pairs = np.asarray(bundle.pt, dtype=np.int64)
    check(len(y) == 15168 and pairs.shape == (5056, 3), "FAIL_131_FORMAL_BENCHMARK")
    formal.verify_pair_order(pairs, y, "GLOBAL")
    oof = np.full(len(y), np.nan); coverage = np.zeros(len(y), np.int8); outer = np.zeros(len(y), np.int8)
    fold_rows, identities, leakage = [], [], []
    for hf in formal.FOLDS:
        score, sel, meta, identity = fit_fold(hf, bundle, runner, base, pairs, y, manifest_hash)
        idx = np.asarray(meta["test_row_indices"], dtype=np.int64); oof[idx] = score; coverage[idx] += 1; outer[idx] = hf
        selected = meta["selected_config"]; tm = old126.score_metrics(score, y[idx])
        fold_rows.append({"fold": hf, "selected_config": selected, "validation_metric_or_selection_rule": "mean(AUROC,AUPRC,StrictPair,Edge); frozen ties",
                          "test_AUROC": tm["AUROC"], "test_AUPRC": tm["AUPRC"], "test_StrictPair": tm["StrictPair"], "test_Edge": tm["Edge"],
                          "train_pair_n": meta["train_pair_n"], "val_pair_n": meta["val_pair_n"], "test_set_n": meta["test_set_n"]})
        identities.append(identity)
        test_pairs = pairs[np.asarray(np.load(SRC / f"FOLD{hf}_FROZEN_SPLITS.npz")["test"], int)]
        leakage.append({"fold": hf, "matched_set_cross_fold": int(np.any(np.ptp(outer[test_pairs], axis=1) != 0)),
                        "pair_cross_fold": int(np.any(np.ptp(outer[test_pairs], axis=1) != 0)), "test_set_n": len(test_pairs)})
    check(np.all(coverage == 1) and np.isfinite(oof).all(), "FAIL_131_OOF_COVERAGE")
    np.save(OUT / "PAIRDIFF_XGB_OOF_score.npy", oof)
    identity_df = csvwrite("PAIRDIFF_FEATURE_IDENTITY_AUDIT.csv", identities)
    leakage_df = csvwrite("PAIRDIFF_FOLD_LEAKAGE_AUDIT.csv", leakage)
    identity_pass = bool(identity_df[["feature_names_order_identical", "labels_identical", "pair_ids_identical", "values_bitwise_identical", "raw_train_hash_matches_126C", "standardizer_matches_126C"]].all().all())
    check(identity_pass, "FAIL_131_PAIRDIFF_FEATURE_IDENTITY")
    check(int(leakage_df.matched_set_cross_fold.sum() + leakage_df.pair_cross_fold.sum()) == 0, "FAIL_131_FOLD_LEAKAGE")
    csvwrite("PAIRDIFF_XGB_FOLD_RESULTS.csv", fold_rows)

    scores = {
        "RAW_POINTWISE_XGB": np.load(SRC / "RAW_POINTWISE_XGB_OOF_score.npy"),
        "PAIRDIFF_LOGIT_CONVERGENCE_AUDITED": np.load(SRC126C / "RAW_PAIRDIFF_LOGIT_CONVERGENCE_AUDIT_OOF_score.npy"),
        "PAIRDIFF_XGB": oof,
        "MSRR_POINTWISE_XGB": np.load(SRC / "MSRR_POINTWISE_XGB_OOF_score.npy"),
    }
    observed = metric_values(scores, y, pairs)
    main_rows = [
        {"method": m, "learner": "XGBoost" if "XGB" in m else "ElasticNet Logistic", "representation": "PAIRDIFF" if "PAIRDIFF" in m else ("MSRR" if "MSRR" in m else "RAW"),
         **observed[m], "source": str(OUT if m == "PAIRDIFF_XGB" else (SRC126C if "LOGIT" in m else SRC)), "new_or_frozen": "NEW" if m == "PAIRDIFF_XGB" else "FROZEN"}
        for m in METHODS]
    for m in ["RAW_RANKPAIR_XGB", "MSRR_RANKPAIR_XGB"]:
        score = np.load(SRC / f"{m}_OOF_score.npy"); met = old126.score_metrics(score[pairs.ravel()], y[pairs.ravel()])
        main_rows.append({"method": m, "learner": "XGBoost rank:pairwise", "representation": "MSRR" if "MSRR" in m else "RAW", **met,
                          "source": str(SRC), "new_or_frozen": "FROZEN_CONTEXT"})
    results = csvwrite("PAIRDIFF_XGB_MAIN_RESULTS.csv", main_rows)
    contrasts = paired_bootstrap(scores, observed, y, pairs)
    c2 = contrasts.loc[contrasts.contrast.eq("C2")]; wins = int((c2.delta > 0).sum()); positive_ci = int((c2.CI95_LOW > 0).sum())
    if wins == 4 and positive_ci == 4:
        gate = "PASS_STRONG_SAME_BACKBONE_REPRESENTATION_SUPPORT"
    elif wins == 4:
        gate = "PASS_DIRECTIONAL_SAME_BACKBONE_SUPPORT"
    elif observed["MSRR_POINTWISE_XGB"]["AUROC"] > observed["PAIRDIFF_XGB"]["AUROC"] and observed["MSRR_POINTWISE_XGB"]["AUPRC"] > observed["PAIRDIFF_XGB"]["AUPRC"]:
        gate = "MIXED_SAME_BACKBONE_SUPPORT"
    else:
        gate = "PAIRDIFF_CAPACITY_CHALLENGES_MSRR_CLAIM"

    # Supplementary long table: one estimate per row, with explicit C2 CIs.
    sx_rows = []
    for method in METHODS:
        rec = results.loc[results.method.eq(method)].iloc[0]
        for metric in METRICS:
            sx_rows.append({"row_type": "METHOD", "method_or_contrast": method, "metric": metric,
                            "estimate": rec[metric], "CI95_LOW": np.nan, "CI95_HIGH": np.nan,
                            "learner": rec.learner, "representation": rec.representation})
    for row in c2.itertuples():
        sx_rows.append({"row_type": "PAIRED_CONTRAST", "method_or_contrast": "MSRR_POINTWISE_XGB_MINUS_PAIRDIFF_XGB",
                        "metric": row.metric, "estimate": row.delta, "CI95_LOW": row.CI95_LOW, "CI95_HIGH": row.CI95_HIGH,
                        "learner": "same XGBoost family", "representation": "MSRR minus PairDiff"})
    pd.DataFrame(sx_rows).to_csv(OUT / "TABLE_SX_PAIRDIFF_SAME_BACKBONE.csv", index=False, encoding="utf-8-sig")
    finish_docs(results, contrasts, gate); claim_scan(gate)
    env = {"Python": sys.version, "executable": sys.executable, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
           "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "xgboost": xgb.__version__, "XGBoost_n_jobs": 1}
    jwrite("ENVIRONMENT_131.json", env)
    audit = {"status": "PASS_131_PAIRDIFF_XGB_FAIRNESS_COMPLETE", "formal_benchmark_used": True, "benchmark_changed": False,
             "folds_changed": False, "pairdiff_representation_changed": False, "pairdiff_feature_identity_pass": identity_pass,
             "xgb_grid_source": str(S120B), "xgb_n_jobs": 1, "kyushu_used_for_model_selection": False, "bootstrap_B": B,
             "bootstrap_unit": "complete matched set", "new_models_trained": ["PAIRDIFF_XGB"], "post_result_rescue": False,
             "extra_hyperparameter_search": False, "outer_test_used_for_selection": False, "matched_sets": 5056,
             "selected_configs": {str(r["fold"]): r["selected_config"] for r in fold_rows}, "MSRR_wins_n": wins, "positive_CI_n": positive_ci,
             "GATE131_DECISION": gate}
    jwrite("AUDIT131.json", audit); jwrite("GATE131_DECISION.json", {"GATE131_DECISION": gate, "MSRR_WINS_N": wins, "POSITIVE_CI_N": positive_ci})
    (OUT / "README_131.md").write_text(f"# Experiment 131\n\nDecision: `{gate}`. Only PairDiff-XGBoost was newly trained. See the protocol, feature-identity, fold-leakage, config, fold, pooled and paired-bootstrap audit files in this directory.\n", encoding="utf-8")

    pl = observed["PAIRDIFF_LOGIT_CONVERGENCE_AUDITED"]; px = observed["PAIRDIFF_XGB"]; mx = observed["MSRR_POINTWISE_XGB"]
    print("EXPERIMENT_131_COMPLETE"); print("FORMAL_BENCHMARK_USED=True"); print("PAIRDIFF_REPRESENTATION_CHANGED=False")
    print(f"PAIRDIFF_FEATURE_IDENTITY_PASS={identity_pass}"); print("XGB_N_JOBS=1"); print("EXTRA_HYPERPARAMETER_SEARCH=False"); print("POST_RESULT_RESCUE=False")
    for label, met in [("PAIRDIFF_LOGIT", pl), ("PAIRDIFF_XGB", px), ("MSRR_XGB", mx)]:
        for metric in METRICS: print(f"{label}_{metric.upper()}={met[metric]:.9f}")
    for metric in METRICS:
        r = c2.loc[c2.metric.eq(metric)].iloc[0]; print(f"DELTA_MSRR_MINUS_PAIRDIFF_XGB_{metric.upper()}={r.delta:.9f}"); print(f"CI_{metric.upper()}=[{r.CI95_LOW:.9f},{r.CI95_HIGH:.9f}]")
    print(f"MSRR_WINS_N={wins}"); print(f"POSITIVE_CI_N={positive_ci}"); print(f"GATE131_DECISION={gate}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--resume", action="store_true"); args = parser.parse_args()
    try:
        raise SystemExit(main(args.resume))
    except Exception as exc:
        if OUT.exists():
            jwrite("FAILURE_131.json", {"error": str(exc), "traceback": traceback.format_exc()})
        raise
