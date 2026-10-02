#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
119_RUN_MSRR_REPRESENTATION_ABLATION_5FOLD_ONE_SHOT.py

Full 5-fold ablation for the frozen MSRR paper mainline.
This is NOT a new-model search.

Seven frozen variants
---------------------
A0_RAW
  raw static + raw 70x10 dynamic.

A1_RAW_PLUS_SIGNED
  raw + signed matched-set residuals.

A2_RAW_PLUS_ABSDEV
  raw + absolute matched-set residuals.

A3_FULL_REL_NO_SUMMARY
  raw + signed + absolute residuals for static and dynamic, no extra summaries.

A4_STATIC_REL_ONLY
  static raw+signed+absolute, dynamic raw only.

A5_RAIN_REL_ONLY
  static raw, dynamic raw+signed+absolute + 7x10 residual summaries.

A6_FULL_MSRR
  static raw+signed+absolute + dynamic raw+signed+absolute + summaries.

A3 and A6 are intentionally distinct:
A3 tests the pure relative transformation; A6 adds the fixed temporal residual
summary block used by the current full MSRR implementation.

Protocol
--------
- seed 7, complete leakage-controlled 5-fold.
- exact same XGBoost candidate grid for ALL seven variants.
- configuration selected on validation only.
- outer test is never used for selection.
- all seven rows are kept, favorable or not.

Run
---
python ^
 <PROJECT_ROOT>\scripts\119_RUN_MSRR_REPRESENTATION_ABLATION_5FOLD_ONE_SHOT.py ^
 --xgb-jobs 8

Output
------
<PROJECT_ROOT>\experiments\MSRR_REPRESENTATION_ABLATION_5FOLD_V1
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "MSRR_REPRESENTATION_ABLATION_5FOLD_V1"

SEED = 7
FOLDS = [1, 2, 3, 4, 5]
METRICS = ["AUROC", "AUPRC", "StrictPair", "Edge"]

VARIANTS = [
    "A0_RAW",
    "A1_RAW_PLUS_SIGNED",
    "A2_RAW_PLUS_ABSDEV",
    "A3_FULL_REL_NO_SUMMARY",
    "A4_STATIC_REL_ONLY",
    "A5_RAIN_REL_ONLY",
    "A6_FULL_MSRR",
]

EXPECTED_DIMS = {
    "A0_RAW": 792,
    "A1_RAW_PLUS_SIGNED": 1584,
    "A2_RAW_PLUS_ABSDEV": 1584,
    "A3_FULL_REL_NO_SUMMARY": 2376,
    "A4_STATIC_REL_ONLY": 976,
    "A5_RAIN_REL_ONLY": 2262,
    "A6_FULL_MSRR": 2446,
}

XGB_GRID = [
    dict(config_id="D4", n_estimators=450, max_depth=4, learning_rate=0.04,
         min_child_weight=2.0, subsample=0.90, colsample_bytree=0.85,
         reg_lambda=5.0, reg_alpha=0.05, gamma=0.0),
    dict(config_id="D6", n_estimators=650, max_depth=6, learning_rate=0.03,
         min_child_weight=2.0, subsample=0.90, colsample_bytree=0.80,
         reg_lambda=5.0, reg_alpha=0.05, gamma=0.0),
    dict(config_id="D8", n_estimators=850, max_depth=8, learning_rate=0.02,
         min_child_weight=2.0, subsample=0.90, colsample_bytree=0.75,
         reg_lambda=7.0, reg_alpha=0.10, gamma=0.0),
]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.TRAIN_SEHC_V2_TEMPLATE import build_fold_loader, load_dataset_loader


def log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (OUT / "RUN.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def jwrite(path, obj):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def sha256_self():
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except Exception:
        return "UNAVAILABLE"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def as_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def get_xgb():
    try:
        import xgboost as xgb
        return xgb
    except Exception as e:
        raise RuntimeError("xgboost unavailable in current environment") from e


def verify_pairs(pair_rows, y_all, label):
    pair_rows = np.asarray(pair_rows, dtype=np.int64)
    if pair_rows.ndim != 2 or pair_rows.shape[1] != 3:
        raise RuntimeError(f"{label}_PAIR_SHAPE_BAD={pair_rows.shape}")
    y3 = y_all[pair_rows]
    if not np.all(y3[:, 0] == 1):
        raise RuntimeError(f"{label}_POS_NOT_SLOT0")
    if not np.all(y3[:, 1:] == 0):
        raise RuntimeError(f"{label}_CTRL_LABEL_BAD")


def dynsum7(x):
    # x [N,70,10] -> [N,70]
    return np.concatenate([
        x.mean(1), x.std(1), x.max(1), x.min(1), x.sum(1),
        x[:, -1, :], np.abs(x).max(1)
    ], axis=1).astype(np.float32, copy=False)


def build_components(pair_rows, static92, rain):
    tri = np.asarray(pair_rows, dtype=np.int64)
    anchors = tri.reshape(-1)
    s = static92[tri].astype(np.float32, copy=False)   # [sets,3,92]
    r = rain[tri].astype(np.float32, copy=False)       # [sets,3,70,10]
    sm = s.mean(1, keepdims=True)
    rm = r.mean(1, keepdims=True)
    sr = s - sm
    rr = r - rm
    n = len(anchors)

    comp = {
        "s_raw": s.reshape(n, 92),
        "s_signed": sr.reshape(n, 92),
        "s_abs": np.abs(sr).reshape(n, 92),
        "r_raw": r.reshape(n, 700),
        "r_signed": rr.reshape(n, 700),
        "r_abs": np.abs(rr).reshape(n, 700),
        "r_sum": dynsum7(rr.reshape(n, 70, 10)),
    }
    for k, v in comp.items():
        if not np.isfinite(v).all():
            raise RuntimeError(f"NONFINITE_COMPONENT={k}")
    return comp, anchors


def make_variant(name, c):
    sr, ss, sa = c["s_raw"], c["s_signed"], c["s_abs"]
    rr, rs, ra, rz = c["r_raw"], c["r_signed"], c["r_abs"], c["r_sum"]

    blocks = {
        "A0_RAW": [sr, rr],
        "A1_RAW_PLUS_SIGNED": [sr, rr, ss, rs],
        "A2_RAW_PLUS_ABSDEV": [sr, rr, sa, ra],
        "A3_FULL_REL_NO_SUMMARY": [sr, rr, ss, sa, rs, ra],
        "A4_STATIC_REL_ONLY": [sr, ss, sa, rr],
        "A5_RAIN_REL_ONLY": [sr, rr, rs, ra, rz],
        "A6_FULL_MSRR": [sr, ss, sa, rr, rs, ra, rz],
    }[name]

    x = np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
    if x.shape[1] != EXPECTED_DIMS[name]:
        raise RuntimeError(
            f"{name}_DIM_BAD={x.shape[1]} expected={EXPECTED_DIMS[name]}"
        )
    if not np.isfinite(x).all():
        raise RuntimeError(f"{name}_NONFINITE")
    return x


def metrics_ordered(margin, y):
    margin = np.asarray(margin, dtype=np.float64)
    y = np.asarray(y, dtype=np.int8)
    if len(margin) % 3 != 0 or margin.shape != y.shape:
        raise RuntimeError("ORDERED_METRIC_SHAPE_BAD")

    y3 = y.reshape(-1, 3)
    if not np.all(y3[:, 0] == 1) or not np.all(y3[:, 1:] == 0):
        raise RuntimeError("ORDERED_METRIC_PAIR_ORDER_BAD")

    s3 = margin.reshape(-1, 3)
    e1 = s3[:, 0] > s3[:, 1]
    e2 = s3[:, 0] > s3[:, 2]
    p = sigmoid(margin)

    return {
        "AUROC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)),
        "StrictPair": float((e1 & e2).mean()),
        "Edge": float(np.column_stack([e1, e2]).mean()),
    }


def metrics_oof(margin, y_all, pair_rows):
    p = sigmoid(margin)
    s3 = margin[pair_rows]
    e1 = s3[:, 0] > s3[:, 1]
    e2 = s3[:, 0] > s3[:, 2]
    return {
        "AUROC": float(roc_auc_score(y_all, p)),
        "AUPRC": float(average_precision_score(y_all, p)),
        "StrictPair": float((e1 & e2).mean()),
        "Edge": float(np.column_stack([e1, e2]).mean()),
    }


def make_model(cfg, seed, jobs):
    xgb = get_xgb()
    kw = {k: v for k, v in cfg.items() if k != "config_id"}
    kw.update(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        max_bin=256,
        random_state=seed,
        n_jobs=jobs,
        verbosity=0,
    )
    return xgb.XGBClassifier(**kw)


def prob_to_margin(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def fit_select_predict(name, fold, xtr, ytr, xva, yva, xte, seed, jobs):
    sw = np.where(ytr == 1, 1.0, 0.5).astype(np.float32)
    rows, models = [], []

    for ci, cfg in enumerate(XGB_GRID):
        model = make_model(cfg, seed + 37 * ci, jobs)
        t0 = time.perf_counter()
        model.fit(xtr, ytr, sample_weight=sw)

        vm = prob_to_margin(model.predict_proba(xva)[:, 1])
        m = metrics_ordered(vm, yva)
        score4 = float(np.mean([m[k] for k in METRICS]))

        rows.append({
            "human_fold": fold,
            "variant": name,
            "config_id": cfg["config_id"],
            **m,
            "validation_mean4": score4,
            "fit_seconds": time.perf_counter() - t0,
        })
        models.append(model)

        log(
            f"FOLD{fold} {name} {cfg['config_id']} "
            f"VAL AUC={m['AUROC']:.6f} AP={m['AUPRC']:.6f} "
            f"SP={m['StrictPair']:.6f} Edge={m['Edge']:.6f}"
        )

    ranked = sorted(
        range(len(rows)),
        key=lambda i: (
            rows[i]["validation_mean4"],
            rows[i]["StrictPair"],
            rows[i]["Edge"],
            rows[i]["AUPRC"],
            rows[i]["AUROC"],
            -i,
        ),
        reverse=True,
    )
    bi = ranked[0]
    log(f"FOLD{fold} {name} SELECTED={rows[bi]['config_id']}")
    tm = prob_to_margin(models[bi].predict_proba(xte)[:, 1])
    return tm, pd.DataFrame(rows), rows[bi]["config_id"]


def delta(a, b):
    return {m: float(a[m] - b[m]) for m in METRICS}


def wins(a, b):
    return sum(a[m] > b[m] for m in METRICS)


def fold_wins(df, a, b, metric):
    aa = df[df.variant == a].set_index("human_fold")[metric]
    bb = df[df.variant == b].set_index("human_fold")[metric]
    folds = sorted(set(aa.index) & set(bb.index))
    return sum(float(aa.loc[f]) > float(bb.loc[f]) for f in folds)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--xgb-jobs", type=int, default=4)
    args = parser.parse_args(argv)

    get_xgb()
    set_seed(SEED)

    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)

    jwrite(OUT / "RUN_MANIFEST.json", {
        "script": Path(__file__).name,
        "sha256": sha256_self(),
        "experiment": "MSRR representation ablation, not new-model search",
        "seed": SEED,
        "folds": FOLDS,
        "variants": VARIANTS,
        "expected_dims": EXPECTED_DIMS,
        "xgb_grid": XGB_GRID,
        "outer_test_used_for_selection": False,
        "post_test_rescue_allowed": False,
    })

    bundle, runner, base = load_dataset_loader()
    y_all = bundle.sample.y_pair.to_numpy(np.int8)
    all_pairs = np.asarray(bundle.pt, dtype=np.int64)
    verify_pairs(all_pairs, y_all, "GLOBAL")

    oof = {v: np.full(int(base.N), np.nan, dtype=np.float64) for v in VARIANTS}
    fold_rows, select_tabs, dim_rows = [], [], []

    for hf in FOLDS:
        log("=" * 100)
        log(f"START FOLD{hf}")
        fd = build_fold_loader(
            bundle, runner, base, hf, torch.device("cpu")
        )

        static92 = as_np(fd.static92).astype(np.float32, copy=False)
        rain = as_np(fd.rain).astype(np.float32, copy=False)

        if static92.shape[1] != 92:
            raise RuntimeError(f"STATIC_SHAPE_BAD={static92.shape}")
        if rain.shape[1:] != (70, 10):
            raise RuntimeError(f"RAIN_SHAPE_BAD={rain.shape}")

        tr_pairs = all_pairs[np.asarray(fd.train_pairs, dtype=np.int64)]
        va_pairs = all_pairs[np.asarray(fd.validation_pairs, dtype=np.int64)]
        te_pairs = all_pairs[np.asarray(fd.test_pairs, dtype=np.int64)]

        for label, tri in [
            (f"FOLD{hf}_TRAIN", tr_pairs),
            (f"FOLD{hf}_VAL", va_pairs),
            (f"FOLD{hf}_TEST", te_pairs),
        ]:
            verify_pairs(tri, y_all, label)

        tr_idx, va_idx, te_idx = (
            tr_pairs.reshape(-1), va_pairs.reshape(-1), te_pairs.reshape(-1)
        )

        if np.intersect1d(tr_idx, va_idx).size:
            raise RuntimeError(f"FOLD{hf}_TRAIN_VAL_OVERLAP")
        if np.intersect1d(tr_idx, te_idx).size:
            raise RuntimeError(f"FOLD{hf}_TRAIN_TEST_OVERLAP")
        if np.intersect1d(va_idx, te_idx).size:
            raise RuntimeError(f"FOLD{hf}_VAL_TEST_OVERLAP")

        trc, tra = build_components(tr_pairs, static92, rain)
        vac, vaa = build_components(va_pairs, static92, rain)
        tec, tea = build_components(te_pairs, static92, rain)

        if not np.array_equal(tra, tr_idx):
            raise RuntimeError("TRAIN_ORDER_MISMATCH")
        if not np.array_equal(vaa, va_idx):
            raise RuntimeError("VAL_ORDER_MISMATCH")
        if not np.array_equal(tea, te_idx):
            raise RuntimeError("TEST_ORDER_MISMATCH")

        ytr, yva, yte = y_all[tr_idx], y_all[va_idx], y_all[te_idx]

        for vi, v in enumerate(VARIANTS):
            log(f"FOLD{hf} START {v}")
            xtr, xva, xte = make_variant(v, trc), make_variant(v, vac), make_variant(v, tec)

            dim_rows.append({
                "human_fold": hf,
                "variant": v,
                "n_train": len(xtr),
                "n_validation": len(xva),
                "n_test": len(xte),
                "feature_dim": xtr.shape[1],
            })

            tm, stab, best = fit_select_predict(
                v, hf, xtr, ytr, xva, yva, xte,
                seed=SEED + hf * 1000 + vi * 100,
                jobs=args.xgb_jobs,
            )
            oof[v][te_idx] = tm
            fm = metrics_ordered(tm, yte)

            fold_rows.append({
                "human_fold": hf,
                "variant": v,
                "selected_config": best,
                "feature_dim": xtr.shape[1],
                **fm,
            })
            select_tabs.append(stab)

            log(
                f"FOLD{hf} TEST {v} "
                f"AUC={fm['AUROC']:.6f} AP={fm['AUPRC']:.6f} "
                f"SP={fm['StrictPair']:.6f} Edge={fm['Edge']:.6f}"
            )

            del xtr, xva, xte, tm
            gc.collect()

        del fd, static92, rain, trc, vac, tec, tra, vaa, tea
        gc.collect()
        log(f"COMPLETE FOLD{hf}")

    final = {}
    for v in VARIANTS:
        if not np.isfinite(oof[v]).all():
            raise RuntimeError(
                f"OOF_INCOMPLETE {v} missing={(~np.isfinite(oof[v])).sum()}"
            )
        np.save(OUT / f"{v}_OOF_margin.npy", oof[v])
        final[v] = metrics_oof(oof[v], y_all, all_pairs)

    result_df = pd.DataFrame([
        {"variant": v, "feature_dim": EXPECTED_DIMS[v], **final[v]}
        for v in VARIANTS
    ])
    result_df.to_csv(OUT / "OOF_RESULTS.csv", index=False, encoding="utf-8-sig")

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(OUT / "FOLD_METRICS.csv", index=False, encoding="utf-8-sig")

    pd.concat(select_tabs, ignore_index=True).to_csv(
        OUT / "VALIDATION_SELECTION.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(dim_rows).to_csv(
        OUT / "FEATURE_DIMENSIONS.csv", index=False, encoding="utf-8-sig"
    )

    raw = final["A0_RAW"]
    delta_rows = []
    for v in VARIANTS:
        d = delta(final[v], raw)
        delta_rows.append({
            "variant": v,
            **{f"delta_{m}_vs_RAW": d[m] for m in METRICS},
            "metrics_beating_RAW": wins(final[v], raw),
        })
    pd.DataFrame(delta_rows).to_csv(
        OUT / "DELTA_VS_RAW.csv", index=False, encoding="utf-8-sig"
    )

    comparisons = [
        ("FULL_MSRR_vs_RAW", "A6_FULL_MSRR", "A0_RAW"),
        ("SIGNED_ONLY_vs_RAW", "A1_RAW_PLUS_SIGNED", "A0_RAW"),
        ("ABSDEV_ONLY_vs_RAW", "A2_RAW_PLUS_ABSDEV", "A0_RAW"),
        ("ADD_SIGNED_GIVEN_ABSDEV", "A3_FULL_REL_NO_SUMMARY", "A2_RAW_PLUS_ABSDEV"),
        ("ADD_ABSDEV_GIVEN_SIGNED", "A3_FULL_REL_NO_SUMMARY", "A1_RAW_PLUS_SIGNED"),
        ("STATIC_REL_vs_RAW", "A4_STATIC_REL_ONLY", "A0_RAW"),
        ("RAIN_REL_vs_RAW", "A5_RAIN_REL_ONLY", "A0_RAW"),
        ("FULL_vs_STATIC_ONLY", "A6_FULL_MSRR", "A4_STATIC_REL_ONLY"),
        ("FULL_vs_RAIN_ONLY", "A6_FULL_MSRR", "A5_RAIN_REL_ONLY"),
        ("SUMMARY_INCREMENT", "A6_FULL_MSRR", "A3_FULL_REL_NO_SUMMARY"),
    ]

    comp_rows = []
    for name, a, b in comparisons:
        d = delta(final[a], final[b])
        row = {
            "comparison": name,
            "new_variant": a,
            "base_variant": b,
            "metric_win_count": wins(final[a], final[b]),
        }
        for m in METRICS:
            row[f"delta_{m}"] = d[m]
            row[f"fold_wins_{m}_out_of_5"] = fold_wins(fold_df, a, b, m)
        comp_rows.append(row)

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(
        OUT / "KEY_COMPONENT_DELTAS.csv", index=False, encoding="utf-8-sig"
    )

    core = all(final["A6_FULL_MSRR"][m] > raw[m] for m in METRICS)
    signed = (
        wins(final["A1_RAW_PLUS_SIGNED"], raw) >= 3
        or wins(final["A3_FULL_REL_NO_SUMMARY"], final["A2_RAW_PLUS_ABSDEV"]) >= 3
    )
    absdev = (
        wins(final["A2_RAW_PLUS_ABSDEV"], raw) >= 3
        or wins(final["A3_FULL_REL_NO_SUMMARY"], final["A1_RAW_PLUS_SIGNED"]) >= 3
    )
    static_ok = wins(final["A4_STATIC_REL_ONLY"], raw) >= 3
    rain_ok = wins(final["A5_RAIN_REL_ONLY"], raw) >= 3
    dual = (
        wins(final["A6_FULL_MSRR"], final["A4_STATIC_REL_ONLY"]) >= 3
        and wins(final["A6_FULL_MSRR"], final["A5_RAIN_REL_ONLY"]) >= 3
    )
    summary_ok = wins(
        final["A6_FULL_MSRR"], final["A3_FULL_REL_NO_SUMMARY"]
    ) >= 3

    strong = core and (signed or absdev) and (static_ok or rain_ok)

    if strong and dual:
        gate = "STRONG_MSRR_ABLATION_SUPPORT"
    elif strong:
        gate = "CORE_MSRR_ABLATION_SUPPORT"
    elif core:
        gate = "CORE_GAIN_ONLY_COMPONENT_STORY_WEAK"
    else:
        gate = "FAIL_CORE_MSRR_ABLATION"

    decision = {
        "status": "PASS_119_ABLATION_COMPLETED",
        "gate119_decision": gate,
        "results": final,
        "gates": {
            "CORE_MSRR_SUPPORTED": core,
            "SIGNED_SIGNAL_SUPPORTED": signed,
            "ABSDEV_SIGNAL_SUPPORTED": absdev,
            "STATIC_REL_SUPPORTED": static_ok,
            "RAIN_REL_SUPPORTED": rain_ok,
            "DUAL_SOURCE_GAIN_SUPPORTED": dual,
            "TEMPORAL_SUMMARY_GAIN_SUPPORTED": summary_ok,
            "STRONG_PAPER_LEVEL_ABLATION_SUPPORT": strong,
        },
        "outer_test_used_for_selection": False,
        "post_result_rescue_allowed": False,
        "next_if_core_supported": "Run cross-backbone Raw-vs-MSRR; no new model branch.",
        "next_if_core_failed": "Stop and audit exact MSRR replication before any new experiment.",
    }
    jwrite(OUT / "GATE119_DECISION.json", decision)

    report = f"""# MSRR Representation Ablation — Full 5-Fold

## Decision
**{gate}**

## OOF results
{result_df.to_string(index=False)}

## Gates
CORE_MSRR_SUPPORTED={core}
SIGNED_SIGNAL_SUPPORTED={signed}
ABSDEV_SIGNAL_SUPPORTED={absdev}
STATIC_REL_SUPPORTED={static_ok}
RAIN_REL_SUPPORTED={rain_ok}
DUAL_SOURCE_GAIN_SUPPORTED={dual}
TEMPORAL_SUMMARY_GAIN_SUPPORTED={summary_ok}
STRONG_PAPER_LEVEL_ABLATION_SUPPORT={strong}

## Discipline
All seven variants are retained. Individual component claims follow their own
gates; a successful full MSRR does not automatically prove every component.
No post-test rescue of the representation definition is allowed.
"""
    (OUT / "GATE119_REPORT.md").write_text(report, encoding="utf-8")

    print("\n" + "=" * 112)
    print("PASS_119_MSRR_REPRESENTATION_ABLATION_COMPLETE")
    print(f"GATE119_DECISION={gate}")
    for v in VARIANTS:
        m = final[v]
        print(
            f"{v}: AUROC={m['AUROC']:.6f} AUPRC={m['AUPRC']:.6f} "
            f"StrictPair={m['StrictPair']:.6f} Edge={m['Edge']:.6f}"
        )
    print("CORE_MSRR_SUPPORTED=" + ("YES" if core else "NO"))
    print("SIGNED_SIGNAL_SUPPORTED=" + ("YES" if signed else "NO"))
    print("ABSDEV_SIGNAL_SUPPORTED=" + ("YES" if absdev else "NO"))
    print("STATIC_REL_SUPPORTED=" + ("YES" if static_ok else "NO"))
    print("RAIN_REL_SUPPORTED=" + ("YES" if rain_ok else "NO"))
    print("DUAL_SOURCE_GAIN_SUPPORTED=" + ("YES" if dual else "NO"))
    print("TEMPORAL_SUMMARY_GAIN_SUPPORTED=" + ("YES" if summary_ok else "NO"))
    print("NEXT_STEP=" + (
        "CROSS_BACKBONE_RAW_VS_MSRR" if core else "STOP_AND_AUDIT_REPLICATION"
    ))
    print(f"OUTPUT={OUT}")
    print("=" * 112)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        jwrite(OUT / "FAILURE.json", {
            "status": "FAIL_119_MSRR_REPRESENTATION_ABLATION",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "sha256": sha256_self(),
        })
        print(traceback.format_exc(), flush=True)
        raise
