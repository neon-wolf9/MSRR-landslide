# -*- coding: utf-8 -*-
"""128C: paired matched-set bootstrap against published PU baselines.

This script never trains a model.  It reads only frozen row-level OOF scores,
audits one-to-one sample alignment, and resamples complete matched sets.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
B1_DIR = ROOT / "experiments" / "MSRR_128B1_PUBLISHED_ALGORITHM_COMPARISON_5FOLD"
MSRR_FILE = ROOT / "experiments" / "MSRR_121_PAIRED_BOOTSTRAP" / "120B_RECOVERED_ROW_LEVEL_OOF.parquet"
OUT = ROOT / "experiments" / "MSRR_128C_PUBLISHED_METHOD_PAIRED_BOOTSTRAP"
LOG_FILE = OUT / "128C_RUN.log"

B = 10_000
SEED = 20260829  # frozen Experiment 121 authority
N_SETS = 5_056
N_ROWS = 15_168
BATCH_SIZE = 64
METRICS = ["AUROC", "AUPRC", "StrictPair", "Edge"]
COMPETITORS = ["PU-BaggingDT", "Spy-PU+BRF", "PU-pullbaggingDT"]
MSRR_NAME = "MSRR-XGBoost"

INPUTS = {
    "PU-BaggingDT": B1_DIR / "128B1_OOF_PU_BAGGINGDT.parquet",
    "Spy-PU+BRF": B1_DIR / "128B1_OOF_SPY_PU_BRF.parquet",
    "PU-pullbaggingDT": B1_DIR / "128B1_OOF_PU_PULLBAGGINGDT.parquet",
    MSRR_NAME: MSRR_FILE,
}

EXPECTED = {
    "PU-BaggingDT": {"AUROC": 0.6286912626838899, "AUPRC": 0.43638737897571406,
                     "StrictPair": 0.49505537974683544, "Edge": 0.6524920886075949},
    "Spy-PU+BRF": {"AUROC": 0.6305394318600035, "AUPRC": 0.46501723365347014,
                   "StrictPair": 0.5439082278481012, "Edge": 0.6893789556962026},
    "PU-pullbaggingDT": {"AUROC": 0.6214169693421026, "AUPRC": 0.43718073228683013,
                         "StrictPair": 0.5089003164556962, "Edge": 0.6626780063291139},
    MSRR_NAME: {"AUROC": 0.881482176130503, "AUPRC": 0.7919079294572564,
                "StrictPair": 0.7501977848101266, "Edge": 0.8464200949367089},
}


def log(message: str = "") -> None:
    print(message, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def blocked(code: str, detail: str) -> None:
    log(f"BLOCKED_128C_{code}")
    log(detail)
    raise RuntimeError(detail)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def load_inputs() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for method, path in INPUTS.items():
        if not path.exists():
            blocked("INPUT_MISSING", str(path))
        log(f"  loading {method}: {path.name}")
        frame = pd.read_parquet(path)
        if method == MSRR_NAME:
            frame = frame.loc[
                (frame["backbone"] == "XGBoost") & (frame["representation"] == "MSRR"),
                ["unit_id", "pair_set_id", "outer_fold", "y_true", "risk_score"],
            ].rename(columns={"outer_fold": "fold", "risk_score": "score"})
        else:
            frame = frame[["unit_id", "pair_set_id", "fold", "y_true", "score"]]
        frame = frame.copy()
        frame["unit_id"] = frame["unit_id"].astype(str)
        frame["pair_set_id"] = frame["pair_set_id"].astype(str)
        frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
        frame["y_true"] = pd.to_numeric(frame["y_true"], errors="raise").astype(int)
        frame["score"] = pd.to_numeric(frame["score"], errors="raise").astype(float)
        frames[method] = frame
        log(f"    rows={len(frame):,}, sets={frame['pair_set_id'].nunique():,}")
    return frames


def audit_and_align(frames: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict]:
    method_audits = []
    aligned: dict[str, pd.DataFrame] = {}
    key = ["unit_id", "pair_set_id"]

    for method, frame in frames.items():
        group = frame.groupby("pair_set_id", sort=False)["y_true"].agg(["size", "sum"])
        checks = {
            "rows_pass": len(frame) == N_ROWS,
            "sets_pass": frame["pair_set_id"].nunique() == N_SETS,
            "positives_pass": int((frame["y_true"] == 1).sum()) == N_SETS,
            "controls_pass": int((frame["y_true"] == 0).sum()) == 2 * N_SETS,
            "unique_key_pass": not frame.duplicated(key).any(),
            "finite_score_pass": bool(np.isfinite(frame["score"]).all()),
            "set_structure_pass": bool(((group["size"] == 3) & (group["sum"] == 1)).all()),
        }
        row = {
            "method": method,
            "rows": len(frame),
            "sets": int(frame["pair_set_id"].nunique()),
            "positives": int((frame["y_true"] == 1).sum()),
            "controls": int((frame["y_true"] == 0).sum()),
            **checks,
            "pass": bool(all(checks.values())),
        }
        method_audits.append(row)
        if not row["pass"]:
            blocked("METHOD_STRUCTURE", json.dumps(row, ensure_ascii=False))

    reference = frames[MSRR_NAME].sort_values(key).reset_index(drop=True)
    reference_index = pd.MultiIndex.from_frame(reference[key])
    cross = {}
    for method, frame in frames.items():
        indexed = frame.set_index(key, verify_integrity=True)
        current_index = indexed.index
        unit_alignment = set(frame["unit_id"]) == set(reference["unit_id"])
        pair_alignment = set(frame["pair_set_id"]) == set(reference["pair_set_id"])
        key_alignment = len(current_index) == len(reference_index) and set(current_index) == set(reference_index)
        if key_alignment:
            current = indexed.loc[reference_index].reset_index()
            label_alignment = bool(np.array_equal(current["y_true"], reference["y_true"]))
            fold_alignment = bool(np.array_equal(current["fold"], reference["fold"]))
        else:
            current = frame
            label_alignment = False
            fold_alignment = False
        cross[method] = {
            "unit_alignment": unit_alignment,
            "pair_alignment": pair_alignment,
            "unit_pair_key_alignment": key_alignment,
            "label_alignment": label_alignment,
            "fold_alignment": fold_alignment,
            "pass": bool(unit_alignment and pair_alignment and key_alignment and label_alignment and fold_alignment),
        }
        if not cross[method]["pass"]:
            blocked("ALIGNMENT_MISMATCH", f"{method}: {cross[method]}")
        aligned[method] = current

    audit = {
        "alignment_key": ["unit_id", "pair_set_id"],
        "row_order_join_used": False,
        "methods": method_audits,
        "cross_method_alignment_against_msrr": cross,
        "all_input_alignment_pass": True,
        "pass": True,
    }
    return aligned, audit


def pair_layout(aligned: dict[str, pd.DataFrame]) -> tuple[np.ndarray, dict[str, dict]]:
    # Alignment is key-based above.  This second sort creates a deterministic 3-row
    # layout per set: controls (unit-id order), followed by the positive.
    order = ["pair_set_id", "y_true", "unit_id"]
    layouts: dict[str, dict] = {}
    canonical_pairs = None
    for method, frame in aligned.items():
        f = frame.sort_values(order).reset_index(drop=True)
        pairs = f["pair_set_id"].to_numpy().reshape(N_SETS, 3)
        if not np.all(pairs == pairs[:, :1]):
            blocked("NONCONTIGUOUS_SET_LAYOUT", method)
        pair_ids = pairs[:, 0]
        if canonical_pairs is None:
            canonical_pairs = pair_ids
        elif not np.array_equal(pair_ids, canonical_pairs):
            blocked("PAIR_ORDER_MISMATCH", method)
        y3 = f["y_true"].to_numpy(dtype=int).reshape(N_SETS, 3)
        if not np.all(y3 == np.array([0, 0, 1])):
            blocked("PAIR_LABEL_LAYOUT", method)
        score3 = f["score"].to_numpy(dtype=float).reshape(N_SETS, 3)
        pos, controls = score3[:, 2], score3[:, :2]
        layouts[method] = {
            "score3": score3,
            "y": y3.reshape(-1),
            "score": score3.reshape(-1),
            "strict": (pos > controls.max(axis=1)).astype(float),
            "edge": (pos[:, None] > controls).mean(axis=1),
        }
    assert canonical_pairs is not None
    return canonical_pairs, layouts


def observed_metrics(layout: dict) -> dict[str, float]:
    return {
        "AUROC": float(roc_auc_score(layout["y"], layout["score"])),
        "AUPRC": float(average_precision_score(layout["y"], layout["score"])),
        "StrictPair": float(layout["strict"].mean()),
        "Edge": float(layout["edge"].mean()),
    }


def ranking_plan(layout: dict) -> dict:
    order = np.argsort(layout["score"], kind="mergesort")
    score_sorted = layout["score"][order]
    starts = np.r_[0, np.flatnonzero(np.diff(score_sorted) != 0) + 1]
    return {
        "order": order,
        "starts": starts,
        "y_sorted": layout["y"][order].astype(float),
        "row_pair_sorted": (np.arange(N_ROWS) // 3)[order],
    }


def weighted_global_metrics(counts: np.ndarray, plan: dict) -> tuple[np.ndarray, np.ndarray]:
    weights = counts[:, plan["row_pair_sorted"]].astype(float, copy=False)
    pos_group = np.add.reduceat(weights * plan["y_sorted"], plan["starts"], axis=1)
    neg_group = np.add.reduceat(weights * (1.0 - plan["y_sorted"]), plan["starts"], axis=1)

    neg_before = np.cumsum(neg_group, axis=1) - neg_group
    auc = np.sum(pos_group * (neg_before + 0.5 * neg_group), axis=1) / (N_SETS * 2.0 * N_SETS)

    pos_desc = pos_group[:, ::-1]
    neg_desc = neg_group[:, ::-1]
    cum_pos = np.cumsum(pos_desc, axis=1)
    cum_total = cum_pos + np.cumsum(neg_desc, axis=1)
    ap = np.sum((pos_desc / N_SETS) * np.divide(cum_pos, cum_total, out=np.zeros_like(cum_pos), where=cum_total > 0), axis=1)
    return auc, ap


def explicit_duplicate_audit(sampled: np.ndarray, layouts: dict[str, dict], boot: dict[str, dict[str, np.ndarray]]) -> dict:
    # The draw position is the required bootstrap_instance_id.  Materializing
    # score3[sampled] produces one independent 3-row block for every occurrence,
    # including repeated pair_set_id values.
    instance_ids = np.arange(N_SETS, dtype=int)
    result = {
        "replicate_checked": 1,
        "bootstrap_instance_id_min": int(instance_ids.min()),
        "bootstrap_instance_id_max": int(instance_ids.max()),
        "sampled_occurrences": int(len(instance_ids)),
        "unique_source_sets": int(np.unique(sampled).size),
        "duplicate_occurrences_preserved": bool(np.unique(sampled).size < len(sampled)),
        "implementation": "draw-column position is bootstrap_instance_id; multiplicity weights are algebraically identical to explicit repeated 3-row blocks",
        "methods": {},
    }
    y = np.tile(np.array([0, 0, 1]), N_SETS)
    for method, layout in layouts.items():
        score3 = layout["score3"][sampled]
        pos, ctrl = score3[:, 2], score3[:, :2]
        explicit = {
            "AUROC": float(roc_auc_score(y, score3.reshape(-1))),
            "AUPRC": float(average_precision_score(y, score3.reshape(-1))),
            "StrictPair": float(np.mean(pos > ctrl.max(axis=1))),
            "Edge": float(np.mean(pos[:, None] > ctrl)),
        }
        optimized = {metric: float(boot[method][metric][0]) for metric in METRICS}
        max_error = max(abs(explicit[m] - optimized[m]) for m in METRICS)
        result["methods"][method] = {"max_abs_error": max_error, "pass": max_error < 1e-12}
        if max_error >= 1e-12:
            blocked("DUPLICATE_IMPLEMENTATION_AUDIT", f"{method}: error={max_error}")
    result["pass"] = True
    return result


def run_bootstrap(layouts: dict[str, dict]) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    rng = np.random.default_rng(SEED)
    plans = {method: ranking_plan(layout) for method, layout in layouts.items()}
    boot = {method: {metric: np.empty(B, dtype=float) for metric in METRICS} for method in layouts}
    first_draw = None
    started = time.perf_counter()
    next_report = 500

    for start in range(0, B, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, B)
        draws = rng.integers(0, N_SETS, size=(stop - start, N_SETS))
        if first_draw is None:
            first_draw = draws[0].copy()
        counts = np.stack([np.bincount(row, minlength=N_SETS) for row in draws])
        for method, layout in layouts.items():
            auc, ap = weighted_global_metrics(counts, plans[method])
            boot[method]["AUROC"][start:stop] = auc
            boot[method]["AUPRC"][start:stop] = ap
            boot[method]["StrictPair"][start:stop] = counts @ layout["strict"] / N_SETS
            boot[method]["Edge"][start:stop] = counts @ layout["edge"] / N_SETS

        if stop >= next_report or stop == B:
            elapsed = time.perf_counter() - started
            rate = stop / elapsed
            eta = (B - stop) / rate if rate else float("nan")
            log(f"  bootstrap {stop:5d}/{B} ({100.0 * stop / B:5.1f}%) | elapsed={elapsed:6.1f}s | ETA={eta:6.1f}s")
            while next_report <= stop:
                next_report += 500

    assert first_draw is not None
    duplicate_audit = explicit_duplicate_audit(first_draw, layouts, boot)
    return boot, duplicate_audit


def make_plot(results: pd.DataFrame) -> None:
    plot = results.copy()
    plot["competitor"] = pd.Categorical(plot["competitor"], COMPETITORS, ordered=True)
    plot["metric"] = pd.Categorical(plot["metric"], METRICS, ordered=True)
    plot = plot.sort_values(["competitor", "metric"]).reset_index(drop=True)
    y = np.arange(len(plot))
    x = plot["observed_delta"].to_numpy()
    low = plot["ci_low"].to_numpy()
    high = plot["ci_high"].to_numpy()
    labels = [f"{r.competitor} — {r.metric}" for r in plot.itertuples()]
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.errorbar(x, y, xerr=np.vstack([x - low, high - x]), fmt="o", color="#215a9a", ecolor="#527ba6", capsize=4)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y, labels=labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("MSRR − competitor (paired delta)", fontsize=11)
    ax.set_title("Published-method comparison: paired matched-set bootstrap 95% CI", fontsize=13)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "128C_PUBLISHED_METHOD_BOOTSTRAP_FOREST.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "128C_PUBLISHED_METHOD_BOOTSTRAP_FOREST.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("", encoding="utf-8")
    total_started = time.perf_counter()
    log("=" * 96)
    log("128C_PUBLISHED_METHOD_PAIRED_BOOTSTRAP")
    log("NO_TRAINING=True | FROZEN_OOF_ONLY=True")
    log(f"B={B} | SEED={SEED} | BOOTSTRAP_UNIT=MATCHED_SET")
    log("=" * 96)

    log("[1/6] Loading four frozen OOF inputs...")
    frames = load_inputs()
    hashes = {method: {"path": str(INPUTS[method]), "sha256": sha256(INPUTS[method])} for method in INPUTS}

    log("[2/6] Auditing structure and key-based one-to-one alignment...")
    aligned, alignment_audit = audit_and_align(frames)
    write_json(OUT / "128C_INPUT_ALIGNMENT_AUDIT.json", alignment_audit)
    log("  INPUT_ALIGNMENT=PASS (unit, set, label, fold; keyed by unit_id + pair_set_id)")

    log("[3/6] Reproducing authoritative pooled metrics...")
    pair_ids, layouts = pair_layout(aligned)
    observed = {method: observed_metrics(layout) for method, layout in layouts.items()}
    reproduction_rows = []
    for method in [*COMPETITORS, MSRR_NAME]:
        for metric in METRICS:
            error = abs(observed[method][metric] - EXPECTED[method][metric])
            reproduction_rows.append({
                "method": method, "metric": metric, "expected": EXPECTED[method][metric],
                "observed": observed[method][metric], "abs_error": error, "pass": error <= 1e-6,
            })
            log(f"  {method:20s} {metric:10s} observed={observed[method][metric]:.12f} error={error:.3g}")
    reproduction = pd.DataFrame(reproduction_rows)
    reproduction.to_csv(OUT / "128C_POOLED_METRIC_REPRODUCTION.csv", index=False, encoding="utf-8-sig")
    if not reproduction["pass"].all():
        blocked("POOLED_REPRODUCTION", reproduction.loc[~reproduction["pass"]].to_string(index=False))
    log("  POOLED_METRIC_REPRODUCTION=PASS")

    config = {
        "experiment": "128C_PUBLISHED_METHOD_PAIRED_BOOTSTRAP",
        "B": B,
        "seed": SEED,
        "seed_authority": "Experiment 121 actual frozen authority",
        "bootstrap_unit": "matched_set / pair_set_id",
        "n_matched_sets_per_replicate": N_SETS,
        "bootstrap_instance_id": "zero-based draw position within each replicate",
        "duplicate_handling": "multiplicity-preserving pair counts, verified against explicit repeated 3-row blocks",
        "same_sampled_sets_all_methods": True,
        "CI_method": "percentile [2.5, 97.5]",
        "p_value_method": "Experiment 121 two-sided add-one: 2*min((1+count(delta<=0))/(B+1),(1+count(delta>=0))/(B+1)), capped at 1",
        "comparison_direction": "MSRR-XGBoost minus competitor",
        "model_training_performed": False,
        "inputs": hashes,
    }
    write_json(OUT / "128C_RUN_CONFIG.json", config)

    log("[4/6] Running 10,000 paired matched-set bootstrap replicates...")
    boot, duplicate_audit = run_bootstrap(layouts)
    alignment_audit["duplicate_occurrence_implementation_audit"] = duplicate_audit
    write_json(OUT / "128C_INPUT_ALIGNMENT_AUDIT.json", alignment_audit)
    log("  DUPLICATE_OCCURRENCE_AUDIT=PASS")

    log("[5/6] Computing 12 paired comparisons and saving all replicates...")
    result_rows = []
    replicate_frames = []
    msrr_boot = boot[MSRR_NAME]
    for competitor in COMPETITORS:
        deltas = {metric: msrr_boot[metric] - boot[competitor][metric] for metric in METRICS}
        replicate_frames.append(pd.DataFrame({
            "bootstrap_id": np.arange(1, B + 1), "competitor": competitor,
            **{f"{metric}_delta": deltas[metric] for metric in METRICS},
        }))
        for metric in METRICS:
            values = deltas[metric]
            ci_low, ci_high = np.quantile(values, [0.025, 0.975])
            p_lower = (1.0 + np.sum(values <= 0.0)) / (B + 1.0)
            p_upper = (1.0 + np.sum(values >= 0.0)) / (B + 1.0)
            result_rows.append({
                "competitor": competitor,
                "metric": metric,
                "MSRR_metric": observed[MSRR_NAME][metric],
                "competitor_metric": observed[competitor][metric],
                "observed_delta": observed[MSRR_NAME][metric] - observed[competitor][metric],
                "bootstrap_mean_delta": float(values.mean()),
                "bootstrap_median_delta": float(np.median(values)),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "p_value": float(min(1.0, 2.0 * min(p_lower, p_upper))),
                "prob_delta_gt_0": float(np.mean(values > 0.0)),
                "B": B,
                "seed": SEED,
            })
    results = pd.DataFrame(result_rows)
    replicates = pd.concat(replicate_frames, ignore_index=True)
    results.to_csv(OUT / "128C_PAIRED_BOOTSTRAP_RESULTS.csv", index=False, encoding="utf-8-sig")
    replicates.to_parquet(OUT / "128C_BOOTSTRAP_REPLICATE_DELTAS.parquet", index=False)
    if len(results) != 12 or len(replicates) != 30_000:
        blocked("OUTPUT_ROW_COUNT", f"results={len(results)}, replicates={len(replicates)}")
    log(f"  results_rows={len(results)} | replicate_rows={len(replicates):,}")

    observed_positive = int((results["observed_delta"] > 0).sum())
    ci_positive = int((results["ci_low"] > 0).sum())
    decision = (
        "STRONG_PUBLISHED_METHOD_STATISTICAL_SUPPORT"
        if observed_positive == 12 and ci_positive == 12
        else "PARTIAL_PUBLISHED_METHOD_SUPPORT"
    )
    gate = {
        "comparisons": 12,
        "observed_delta_gt_0": observed_positive,
        "ci_low_gt_0": ci_positive,
        "all_input_alignment_pass": True,
        "decision": decision,
    }
    write_json(OUT / "128C_GATE_SUMMARY.json", gate)

    log("[6/6] Generating forest plot and final gate summary...")
    make_plot(results)
    elapsed = time.perf_counter() - total_started
    log("")
    log("=" * 96)
    log("PASS_128C_PUBLISHED_METHOD_PAIRED_BOOTSTRAP_COMPLETE")
    log(f"B={B}")
    log("BOOTSTRAP_UNIT=MATCHED_SET")
    log("INPUT_ALIGNMENT=PASS")
    log("=" * 96)
    for competitor in COMPETITORS:
        log(f"\n{competitor}:")
        subset = results.loc[results["competitor"] == competitor].set_index("metric")
        for metric in METRICS:
            row = subset.loc[metric]
            log(f"{metric} Delta={row['observed_delta']:+.6f} CI=[{row['ci_low']:+.6f}, {row['ci_high']:+.6f}] p={row['p_value']:.8f}")
    log("")
    log(f"OBSERVED_POSITIVE={observed_positive}/12")
    log(f"CI_LOW_POSITIVE={ci_positive}/12")
    log(f"GATE128C_DECISION={decision}")
    log(f"TOTAL_RUNTIME_SECONDS={elapsed:.1f}")
    log(f"OUTPUT={OUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"128C EXPERIMENT FAILED: {type(exc).__name__}: {exc}", flush=True)
        sys.exit(1)
