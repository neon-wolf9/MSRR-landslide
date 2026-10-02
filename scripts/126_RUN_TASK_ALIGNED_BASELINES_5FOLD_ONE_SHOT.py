#!/usr/bin/env python
"""Frozen Major Revision experiment 126. No result-dependent method changes."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import inspect
import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/MSRR_TASK_ALIGNED_BASELINES_5FOLD_V1"
sys.path.insert(0, str(ROOT))


def import_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


formal = import_script("formal120b", "120B_RUN_MSRR_CROSS_BACKBONE_5FOLD_NO_LIGHTGBM_ONE_SHOT.py")
boot121 = import_script("bootstrap121", "121_RECOVER_OOF_AND_PAIRED_BOOTSTRAP.py")
REFERENCE = formal.OUT
METHODS = ["RAW_POINTWISE_XGB", "RAW_PAIRDIFF_LOGIT", "RAW_RANKPAIR_XGB",
           "MSRR_POINTWISE_XGB", "MSRR_RANKPAIR_XGB"]
METRICS = formal.METRICS4
CONTRASTS = [
    ("C1", METHODS[2], METHODS[0], "Objective effect under RAW"),
    ("C2", METHODS[3], METHODS[0], "Representation effect under pointwise"),
    ("C3", METHODS[4], METHODS[2], "Representation effect under pairwise"),
    ("C4", METHODS[4], METHODS[3], "Objective effect under MSRR"),
    ("C5", METHODS[3], METHODS[2], "MSRR-pointwise vs RAW-rankpair"),
    ("C6", METHODS[3], METHODS[1], "MSRR-pointwise vs PairDiff"),
]
GATES = {
    "STRONG_REPRESENTATION_BEYOND_OBJECTIVE": "C3 wins >=3 AND C5 wins >=3 AND C6 wins >=3",
    "REPRESENTATION_OBJECTIVE_COMPLEMENTARY": "C3 wins >=3 AND C5 wins <3",
    "OBJECTIVE_DOMINANT_OR_MIXED": "otherwise",
}
# Fixed before fitting; same-environment deterministic replication is expected.
METRIC_ATOL = 1e-6
SCORE_ATOL = 1e-5
B = 10000
AUDIT = {}


def check(condition, code):
    if not condition:
        raise RuntimeError(code)


def log(message):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with (OUT / "RUN.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def csv(name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")
    return frame


def score_metrics(scores, labels):
    """Use raw scalar scores to avoid sigmoid saturation creating ranking ties."""
    scores = np.asarray(scores, dtype=np.float64)
    check(np.isfinite(scores).all(), "FAIL_126_NONFINITE_SCORE")
    formal.verify_pair_order(np.arange(len(labels)).reshape(-1, 3), labels, "METRICS")
    s = scores.reshape(-1, 3)
    edges = s[:, :1] > s[:, 1:]
    return dict(AUROC=float(roc_auc_score(labels, scores)),
                AUPRC=float(average_precision_score(labels, scores)),
                StrictPair=float(edges.all(axis=1).mean()), Edge=float(edges.mean()))


def pairdiff(xtr, ytr, xva, yva, xte, seed, fold, jobs):
    mu, sd = formal.fit_standardizer(xtr)
    np.savez(OUT / f"FOLD{fold}_PAIRDIFF_STANDARDIZER.npz", mu=mu, sd=sd)
    ztr, zva, zte = [formal.apply_standardizer(x, mu, sd).astype(np.float64)
                       for x in (xtr, xva, xte)]
    formal.verify_pair_order(np.arange(len(ytr)).reshape(-1, 3), ytr, "PAIRDIFF_TRAIN")
    tri = ztr.reshape(-1, 3, ztr.shape[1])
    d1, d2 = tri[:, 0] - tri[:, 1], tri[:, 0] - tri[:, 2]
    xp = np.stack([d1, d2, -d1, -d2], axis=1).reshape(-1, ztr.shape[1])
    yp = np.tile([1, 1, 0, 0], len(tri))
    check(len(xp) == 4 * len(tri), "FAIL_126_PAIRDIFF_TRAIN_COUNT")
    models, rows = [], []
    for ci, c in enumerate(formal.LOGISTIC_C_GRID):
        log(f"FOLD{fold} RAW_PAIRDIFF_LOGIT C={c}")
        model = LogisticRegression(C=c, penalty="elasticnet", l1_ratio=0.5,
            solver="saga", max_iter=3000, fit_intercept=False, tol=1e-4,
            random_state=seed + ci * 31, n_jobs=jobs)
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(xp, yp)
        m = score_metrics(model.decision_function(zva), yva)
        rows.append(dict(config_id=f"C={c:g}", **m, validation_score=formal.val_scalar(m),
            fit_seconds=time.perf_counter()-t0, n_iter=int(model.n_iter_.max()),
            convergence_warning=any(issubclass(w.category, ConvergenceWarning) for w in caught)))
        models.append(model)
    bi = formal.choose_best_index(rows)
    model = models[bi]
    errors = []
    # No labels or roles are supplied to inference or this algebraic identity check.
    for z in (ztr, zva, zte):
        s = model.decision_function(z).reshape(-1, 3)
        t = z.reshape(-1, 3, z.shape[1])
        for c in (1, 2):
            delta = model.decision_function(t[:, 0]-t[:, c])
            errors.append(float(np.max(np.abs(s[:, 0]-s[:, c]-delta))))
            check(np.allclose(s[:, 0]-s[:, c], delta, atol=1e-8, rtol=1e-7),
                  "FAIL_126_PAIRDIFF_SCORE_INCONSISTENT")
    AUDIT.setdefault("pairdiff_identity", []).append(dict(human_fold=fold, max_abs_error=max(errors)))
    np.save(OUT / f"FOLD{fold}_PAIRDIFF_COEF.npy", model.coef_)
    return model.decision_function(zte), pd.DataFrame(rows), rows[bi]["config_id"]


def rankpair(rep, fold, xtr, ytr, xva, yva, xte, seed, jobs, mode):
    models, rows = [], []
    nsets = len(ytr) // 3
    groups = {"qid": np.repeat(np.arange(nsets), 3)} if mode == "qid" else {
        "group": np.full(nsets, 3, dtype=np.int32)}
    for ci, cfg in enumerate(formal.XGB_CANDIDATES):
        log(f"FOLD{fold} {rep}_RANKPAIR_XGB {cfg['config_id']}")
        kwargs = formal.make_xgb(cfg, seed + ci * 37, jobs).get_params()
        kwargs.update(objective="rank:pairwise", eval_metric="ndcg@3")
        model = xgb.XGBRanker(**kwargs)
        t0 = time.perf_counter()
        model.fit(xtr, ytr, **groups)
        m = score_metrics(model.predict(xva), yva)
        rows.append(dict(config_id=cfg["config_id"], **m, validation_score=formal.val_scalar(m),
                         fit_seconds=time.perf_counter()-t0))
        models.append(model)
    bi = formal.choose_best_index(rows)
    models[bi].save_model(OUT / f"FOLD{fold}_{rep}_RANKPAIR_XGB.ubj")
    return models[bi].predict(xte).astype(np.float64), pd.DataFrame(rows), rows[bi]["config_id"]


def bootstrap(oof, y, pairs, observed):
    """121's same-draw complete-set counts, shared across all six contrasts."""
    row_pair = np.empty(len(y), dtype=np.int64)
    row_pair[pairs.ravel()] = np.repeat(np.arange(len(pairs)), 3)
    success = {m: boot121.pair_success_arrays(oof[m], pairs) for m in METHODS}
    # 120B metrics use sigmoid; retain that exact pointwise convention only.
    values = {m: formal.sigmoid(oof[m]) if "POINTWISE" in m else oof[m] for m in METHODS}
    draws = np.empty((B, len(METHODS), 4), dtype=np.float64)
    rng = np.random.default_rng(boot121.BOOTSTRAP_SEED)
    for b in range(B):
        counts = np.bincount(rng.integers(0, len(pairs), size=len(pairs)),
                             minlength=len(pairs)).astype(np.float64)
        weights = counts[row_pair]
        for mi, m in enumerate(METHODS):
            strict, edge = success[m]
            draws[b, mi] = [roc_auc_score(y, values[m], sample_weight=weights),
                average_precision_score(y, values[m], sample_weight=weights),
                np.dot(counts, strict)/len(pairs), np.dot(counts, edge)/(2*len(pairs))]
        if (b+1) % 100 == 0:
            log(f"BOOTSTRAP {b+1}/{B}")
    np.save(OUT / "BOOTSTRAP_METRIC_DRAWS.npy", draws)
    rows = []
    for cid, a, b, description in CONTRASTS:
        delta = draws[:, METHODS.index(a)] - draws[:, METHODS.index(b)]
        for k, metric in enumerate(METRICS):
            d = delta[:, k]
            lo, hi = np.quantile(d, [0.025, 0.975])
            rows.append(dict(contrast=cid, description=description, method_a=a, method_b=b,
                metric=metric, observed_delta=observed[a][metric]-observed[b][metric],
                bootstrap_mean=float(d.mean()), CI95_LOW=float(lo), CI95_HIGH=float(hi),
                fraction_delta_gt_0=float((d > 0).mean()), B=B, seed=boot121.BOOTSTRAP_SEED))
    return csv("BOOTSTRAP_KEY_COMPARISONS.csv", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xgb-jobs", type=int, default=8)
    args = parser.parse_args()
    check(not OUT.exists() or not any(OUT.iterdir()), "FAIL_126_OUTPUT_ALREADY_EXISTS")
    OUT.mkdir(parents=True, exist_ok=True)
    formal.log = log  # Redirect imported training helper logs; never touch 120B outputs.
    formal.set_seed(formal.SEED)
    mode = "qid" if "qid" in inspect.signature(xgb.XGBRanker.fit).parameters else "group"
    protected = [Path(formal.__file__), Path(boot121.__file__), ROOT / "scripts/TRAIN_SEHC_V2_TEMPLATE.py"]
    for name in ("MSRR_CROSS_BACKBONE_5FOLD_V1B", "MSRR_121_PAIRED_BOOTSTRAP"):
        protected.extend(p for p in (ROOT / "experiments" / name).rglob("*") if p.is_file())
    hashes = {str(p): boot121.sha256_file(p) for p in protected}
    manifest = dict(experiment=126, source_sha256=boot121.sha256_file(Path(__file__)),
        source_hashes=hashes, methods=METHODS, metrics=METRICS, seed=formal.SEED,
        fold_seed="7 + human_fold*1000; XGB +600; candidate +37*index; PairDiff +100; candidate +31*index",
        protocol="PRIMARY_BALANCED_SPATIAL_GROUPED_5FOLD", dataset="Hiroshima 2018",
        loader="120B.load_dataset_loader / build_fold_loader; bundle.pt; fd.train_pairs/validation_pairs/test_pairs",
        raw_dim=792, msrr_dim=2446, feature_functions="120B.raw_features/msrr_features/dynsum7 imported unchanged",
        xgboost_version=xgb.__version__, ranking_API_mode=mode, xgb_jobs=args.xgb_jobs,
        xgb_candidates=formal.XGB_CANDIDATES, pointwise_weights={"positive":1.0,"control":0.5},
        pointwise_common_params={"objective":"binary:logistic", "eval_metric":"logloss", "tree_method":"hist", "max_bin":256},
        ranker_changes={"objective":"rank:pairwise", "eval_metric":"ndcg@3", "group_size":3, "group_weight":"uniform"},
        pairdiff=dict(C=formal.LOGISTIC_C_GRID, penalty="elasticnet", l1_ratio=0.5, solver="saga",
            max_iter=3000, fit_intercept=False, tol=1e-4, standardizer="training candidate rows only, sd<1e-6->1",
            training_edges="P-C1,P-C2,C1-P,C2-P", candidate_score="w^T z"),
        selection="validation mean(AUROC,AUPRC,StrictPair,Edge); ties StrictPair,Edge,AUPRC,AUROC,earliest config",
        scoring="pointwise: unchanged 120B logit discrimination score and metrics; others: raw scalar ranking score",
        bootstrap_B=B, bootstrap_seed=boot121.BOOTSTRAP_SEED, bootstrap_unit="COMPLETE MATCHED SET",
        contrasts=CONTRASTS, interpretation_gate=GATES,
        replication_metric_atol=METRIC_ATOL, replication_score_atol=SCORE_ATOL,
        replication_config_match_required=True, outer_test_used_for_selection=False,
        post_result_rescue_allowed=False)
    formal.jwrite(OUT / "RUN_MANIFEST.json", manifest)
    manifest_hash = boot121.sha256_file(OUT / "RUN_MANIFEST.json")
    log("Frozen manifest saved before any model fit.")
    bundle, runner, base = formal.load_dataset_loader()
    y = bundle.sample.y_pair.to_numpy(np.int8)
    pairs = np.asarray(bundle.pt, dtype=np.int64)
    check(len(y) == int(base.N) == 15168 and pairs.shape == (5056, 3), "FAIL_126_BENCHMARK_SIZE")
    check(np.array_equal(np.sort(pairs.ravel()), np.arange(len(y))), "FAIL_126_GLOBAL_ROW_COVERAGE")
    formal.verify_pair_order(pairs, y, "GLOBAL")
    AUDIT.update(benchmark_rows=len(y), matched_sets=len(pairs), exactly_3_rows=True,
        slot0_label_1=True, slot1_label_0=True, slot2_label_0=True,
        positive=int(y.sum()), controls=int((y == 0).sum()), RAW_dim=792, MSRR_dim=2446,
        standardizer_fitted_train_only=True, pairdiff_test_labels_never_used_in_feature_construction=True,
        ranker_qid_never_used_as_predictor=True, sample_role_never_used_as_predictor=True,
        control_rank_never_used_as_predictor=True, pair_set_id_never_used_as_predictor=True,
        outer_test_never_used_for_hyperparameter_selection=True, post_result_rescue_allowed=False)
    oof = {m: np.full(len(y), np.nan) for m in METHODS}
    coverage = {m: np.zeros(len(y), dtype=np.int8) for m in METHODS}
    outer = np.zeros(len(y), dtype=np.int8)
    split_records, folds, selections, repaudit = [], [], [], []
    reference = pd.read_csv(REFERENCE / "OOF_RESULTS.csv")
    fold_reference = pd.read_csv(REFERENCE / "FOLD_METRICS.csv")
    frozen_scores = {rep: np.load(REFERENCE / f"XGBoost_{rep}_OOF_margin.npy") for rep in ("RAW", "MSRR")}
    # First phase gates replication before any new baseline is interpreted or fitted.
    for phase in ("replication", "new_baselines"):
        for hf in formal.FOLDS:
            log(f"PHASE={phase} FOLD={hf}")
            formal.set_seed(formal.SEED + hf*1000)
            fd = formal.build_fold_loader(bundle, runner, base, hf, torch.device("cpu"))
            ids = [np.asarray(v, dtype=np.int64) for v in (fd.train_pairs, fd.validation_pairs, fd.test_pairs)]
            rows = [pairs[v].reshape(-1) for v in ids]
            for i in range(3):
                check(len(np.unique(ids[i])) == len(ids[i]), "FAIL_126_DUPLICATE_SPLIT_PAIR")
                formal.verify_pair_order(pairs[ids[i]], y, f"FOLD{hf}_SPLIT{i}")
                for j in range(i):
                    check(not np.intersect1d(ids[i], ids[j]).size, "FAIL_126_PAIR_OVERLAP")
                    check(not np.intersect1d(rows[i], rows[j]).size, "FAIL_126_ROW_OVERLAP")
            check(np.array_equal(np.sort(np.concatenate(ids)), np.arange(len(pairs))), "FAIL_126_SPLIT_COVERAGE")
            check(len(ids[2]) == [1011,1011,1011,1011,1012][hf-1], "FAIL_126_FOLD_SIZE")
            splitfile = OUT / f"FOLD{hf}_FROZEN_SPLITS.npz"
            if phase == "replication":
                check(np.all(outer[rows[2]] == 0), "FAIL_126_OUTER_FOLD_OVERLAP")
                outer[rows[2]] = hf
                np.savez(splitfile, train=ids[0], validation=ids[1], test=ids[2])
                split_records.append(dict(human_fold=hf, train=len(ids[0]), validation=len(ids[1]), test=len(ids[2]),
                    row_overlap=False, matched_set_overlap=False, sha256=boot121.sha256_file(splitfile)))
            else:
                with np.load(splitfile) as saved:
                    check(all(np.array_equal(v, saved[k]) for v,k in zip(ids, ("train","validation","test"))),
                          "FAIL_126_SPLIT_CHANGED")
            static = formal.as_numpy(fd.static92).astype(np.float32, copy=False)
            rain = formal.as_numpy(fd.rain).astype(np.float32, copy=False)
            for rep in ("RAW", "MSRR"):
                feature_fn = formal.raw_features if rep == "RAW" else formal.msrr_features
                features = [feature_fn(pairs[v], static, rain) for v in ids]
                check(all(np.array_equal(a, r) for (_,a),r in zip(features,rows)), "FAIL_126_FEATURE_ORDER")
                xtr,xva,xte = [x for x,a in features]
                common = (xtr, y[rows[0]], xva, y[rows[1]], xte)
                seed = formal.SEED + hf*1000 + 600
                todo = [f"{rep}_POINTWISE_XGB"] if phase == "replication" else [f"{rep}_RANKPAIR_XGB"]
                if phase == "new_baselines" and rep == "RAW":
                    todo.insert(0, "RAW_PAIRDIFF_LOGIT")
                for method in todo:
                    if "POINTWISE" in method:
                        score, sel, cfg = formal.fit_xgb_select_predict(rep,hf,*common,seed,args.xgb_jobs)
                    elif "PAIRDIFF" in method:
                        score, sel, cfg = pairdiff(*common,formal.SEED+hf*1000+100,hf,args.xgb_jobs)
                    else:
                        score, sel, cfg = rankpair(rep,hf,*common,seed,args.xgb_jobs,mode)
                    check(np.isfinite(score).all() and score.shape == rows[2].shape, "FAIL_126_SCORE_SHAPE")
                    check(np.all(coverage[method][rows[2]] == 0), "FAIL_126_DUPLICATE_OOF")
                    oof[method][rows[2]] = score
                    coverage[method][rows[2]] += 1
                    evaluator = formal.metrics_from_ordered_rows if "POINTWISE" in method else score_metrics
                    m = evaluator(score,y[rows[2]])
                    folds.append(dict(method=method,human_fold=hf,selected_config=cfg,**m))
                    sel["method"], sel["human_fold"] = method,hf
                    sel["selected"] = sel.config_id == cfg
                    selections.append(sel)
                    csv("FOLD_METRICS.csv", folds)
                    csv("VALIDATION_SELECTION.csv", pd.concat(selections,ignore_index=True))
                    np.save(OUT / f"FOLD{hf}_{method}_score.npy",score)
                    if "POINTWISE" in method:
                        ref = fold_reference[(fold_reference.backbone == "XGBoost") &
                            (fold_reference.representation == rep) & (fold_reference.human_fold == hf)].iloc[0]
                        diff = float(np.max(np.abs(score-frozen_scores[rep][rows[2]])))
                        passed = diff <= SCORE_ATOL and str(ref.selected_config) == cfg and all(abs(m[k]-ref[k]) <= METRIC_ATOL for k in METRICS)
                        repaudit.append(dict(method=method,human_fold=hf,max_score_abs_diff=diff,
                            selected_config=cfg,reference_config=str(ref.selected_config),passed=passed))
                        csv("REPLICATION_AUDIT.csv",repaudit)
                        check(passed,"FAIL_126_FROZEN_XGB_REPLICATION")
                    log(f"FOLD{hf} {method} SELECTED={cfg} COMPLETE")
                del features,xtr,xva,xte,common
                gc.collect()
            del fd,static,rain
            gc.collect()
        if phase == "replication":
            for rep in ("RAW","MSRR"):
                m = formal.metrics_from_full_oof(oof[f"{rep}_POINTWISE_XGB"],y,pairs)
                ref = reference[(reference.backbone == "XGBoost") & (reference.representation == rep)].iloc[0]
                check(all(abs(m[k]-ref[k]) <= METRIC_ATOL for k in METRICS),"FAIL_126_FROZEN_XGB_REPLICATION")
            log("PASS_126_FROZEN_XGB_REPLICATION; proceeding to fixed new baselines")
    check(all(np.all(c == 1) for c in coverage.values()),"FAIL_126_OOF_COVERAGE")
    check(all(np.isfinite(s).all() for s in oof.values()),"FAIL_126_NONFINITE_OOF")
    check(np.all(outer > 0),"FAIL_126_INCOMPLETE_FOLDS")
    observed = {}
    row_pair = np.empty(len(y),dtype=np.int64)
    row_pair[pairs.ravel()] = np.repeat(np.arange(len(pairs)),3)
    predictions = pd.DataFrame(dict(row_index=np.arange(len(y)),matched_set_index=row_pair,human_fold=outer,y_pair=y))
    for method in METHODS:
        s = oof[method]
        observed[method] = (formal.metrics_from_full_oof(s,y,pairs) if "POINTWISE" in method else
                            score_metrics(s[pairs.ravel()],y[pairs.ravel()]))
        np.save(OUT / f"{method}_OOF_score.npy",s)
        np.save(OUT / f"{method}_OOF_M.npy",s[pairs[:,0]]-s[pairs[:,1:]].max(axis=1))
        predictions[method+"_score"] = s
    csv("OOF_PREDICTIONS.csv",predictions)
    result = csv("OOF_RESULTS.csv",[dict(method=m,**observed[m]) for m in METHODS])
    matrix = result.copy()
    matrix["Objective"] = ["pointwise","pairwise-difference","rank:pairwise","pointwise","rank:pairwise"]
    matrix["Representation"] = ["RAW","RAW difference","RAW","MSRR","MSRR"]
    csv("REPRESENTATION_OBJECTIVE_MATRIX.csv",matrix)
    contrasts = csv("KEY_COMPARISONS.csv",[dict(contrast=cid,description=desc,method_a=a,method_b=b,
        **{f"delta_{k}":observed[a][k]-observed[b][k] for k in METRICS}) for cid,a,b,desc in CONTRASTS])
    bs = bootstrap(oof,y,pairs,observed)
    wins = {cid:sum(observed[a][k] > observed[b][k] for k in METRICS) for cid,a,b,_ in CONTRASTS}
    if wins["C3"] >= 3 and wins["C5"] >= 3 and wins["C6"] >= 3:
        gate = "STRONG_REPRESENTATION_BEYOND_OBJECTIVE"
    elif wins["C3"] >= 3 and wins["C5"] < 3:
        gate = "REPRESENTATION_OBJECTIVE_COMPLEMENTARY"
    else:
        gate = "OBJECTIVE_DOMINANT_OR_MIXED"
    check(all(boot121.sha256_file(Path(p)) == h for p,h in hashes.items()),"FAIL_126_PROTECTED_SOURCE_CHANGED")
    check(boot121.sha256_file(OUT / "RUN_MANIFEST.json") == manifest_hash,"FAIL_126_MANIFEST_CHANGED")
    AUDIT.update(status="PASS_126_TASK_ALIGNED_BASELINES_COMPLETE", five_folds_complete=True,
        splits=split_records, train_validation_test_no_row_overlap=True,
        train_validation_test_no_matched_set_overlap=True, all_OOF_scores_finite=True,
        each_candidate_exactly_one_OOF_score=True, pointwise_XGB_reproduces_120B=True,
        replication=repaudit, protected_source_hashes_unchanged=True, manifest_unchanged=True)
    formal.jwrite(OUT / "AUDIT126.json",AUDIT)
    decision = dict(status=AUDIT["status"],GATE126_DECISION=gate,
        representation_pairwise_wins=wins["C3"],msrr_pointwise_vs_raw_rank_wins=wins["C5"],pairdiff_beaten=wins["C6"])
    formal.jwrite(OUT / "GATE126_DECISION.json",decision)
    report = "# Experiment 126: task-aligned baselines\n\n" + AUDIT["status"] + "\n\n"
    report += matrix.to_markdown(index=False,floatfmt=".6f") + "\n\n"
    report += contrasts.drop(columns=["method_a","method_b"]).to_markdown(index=False,floatfmt=".6f") + "\n\n"
    report += bs.drop(columns=["method_a","method_b"]).to_markdown(index=False,floatfmt=".6f") + "\n\n"
    report += f"Pre-registered gate: **{gate}**. C3/C5/C6 wins: {wins['C3']}/{wins['C5']}/{wins['C6']} out of 4.\n\n"
    report += "Scores are candidate discrimination/ranking scores; M is the matched-set discrimination margin. "
    report += "Bootstrap resamples 5,056 complete matched sets with replacement, using the same draws for every method. "
    report += "Percentile 95% intervals describe paired OOF contrasts conditional on fitted models; folds, metrics, and draws are not independent scientific replications. "
    report += "No threshold tuning, post-result rescue, or automatic manuscript conclusion.\n"
    (OUT / "GATE126_REPORT.md").write_text(report,encoding="utf-8")
    log(AUDIT["status"])
    log(result.to_string(index=False))
    log(contrasts.to_string(index=False))
    log(bs.to_string(index=False))
    log(json.dumps(decision))
    log(f"OUTPUT={OUT}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        if OUT.exists() and str(exc) != "FAIL_126_OUTPUT_ALREADY_EXISTS":
            AUDIT.update(status=str(exc) if str(exc).startswith("FAIL_126_") else "FAIL_126_TECHNICAL_ERROR",
                         error=str(exc),traceback=traceback.format_exc())
            formal.jwrite(OUT / "AUDIT126.json",AUDIT)
            formal.jwrite(OUT / "FAILURE126.json",AUDIT)
        raise
