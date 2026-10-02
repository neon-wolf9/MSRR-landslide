# -*- coding: utf-8 -*-
"""
23_RUN_NEURAL_BASELINES_REPAIRED_V2.py
Formal repaired-V2 rerun for BASE-05 / BASE-06 only.

Recommended order:
  1) --preflight-only
  2) --models BASE-05
  3) review BASE-05, then --models BASE-06

It reuses the already validated 20A repaired-V2 loader/metric code and the
formal BASE-00 model classes/candidate registry. No FULL model, ablation,
strict17, external A26, new matching, or new split is touched.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SCRIPT20A = ROOT / "scripts" / "20A_RUN_TABULAR_BASELINES_REPAIRED_V2.py"
OUT = ROOT / "experiments" / "FORMAL_BASELINES_REPAIRED_V2_V1"
BASE00 = ROOT / "modeling" / "BASE" / "BASE-00_contract"

if str(BASE00) not in sys.path:
    sys.path.insert(0, str(BASE00))

from baseline_tuning import candidate_configs
from baseline_plain_fusion import PlainStaticDynamicFusion
from baseline_ranknet import PairwiseRankNet

SEEDS = [7, 11, 21]
CATEGORY_COUNT = 14
EXPECTED_PARAMS = 85978
DROPOUT = 0.10
GRAD_CLIP = 5.0


def load_20a():
    if not SCRIPT20A.exists():
        raise RuntimeError(f"Missing repaired-V2 source: {SCRIPT20A}")
    spec = importlib.util.spec_from_file_location("repaired_v2_tabular_base", SCRIPT20A)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


base = load_20a()


def log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (OUT / "23_NEURAL_RUN.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def jwrite(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def jread(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def cfg_get(cfg, names, cast=float, required=True, default=None):
    for n in names:
        if n in cfg:
            return cast(cfg[n])
    if required:
        raise RuntimeError(f"Candidate missing {names}: {cfg}")
    return default


def norm_cfg(mid, cfg):
    x = {
        "learning_rate": cfg_get(cfg, ["learning_rate", "lr"], float),
        "weight_decay": cfg_get(cfg, ["weight_decay", "wd"], float),
        "pairs_per_batch": cfg_get(cfg, ["pairs_per_batch", "batch_pairs", "batch_size"], int),
        "max_epochs": cfg_get(cfg, ["max_epochs", "epochs", "maximum_epochs"], int),
        "patience": cfg_get(cfg, ["patience", "early_stopping_patience"], int),
        "min_epochs": cfg_get(cfg, ["min_epochs", "minimum_epochs"], int, False, 1),
    }
    if mid == "BASE-06":
        # BASE-06 frozen candidate registry does NOT contain tau/temperature.
        # The formal baseline is standard two-edge logistic RankNet:
        #   softplus(-(s_pos - s_ctrl))
        # which is equivalent to a fixed temperature of 1.0.
        # This value is fixed by the loss definition and is NOT a tuned candidate.
        x["ranknet_temperature_fixed"] = 1.0
    return x


def model_dir(mid):
    return OUT / mid.replace("-", "")


def make_model(mid, seed, device):
    set_seed(seed)
    model = PlainStaticDynamicFusion() if mid == "BASE-05" else PairwiseRankNet()
    model = model.to(device)
    n = sum(p.numel() for p in model.parameters())
    if n != EXPECTED_PARAMS:
        raise RuntimeError(f"{mid} parameter count={n}, expected={EXPECTED_PARAMS}")
    return model


def preflight(device):
    b = base.preflight()
    cfgs = candidate_configs()
    for mid in ("BASE-05", "BASE-06"):
        if mid not in cfgs or len(cfgs[mid]) == 0:
            raise RuntimeError(f"No frozen candidates for {mid}")
        for c in cfgs[mid]:
            norm_cfg(mid, c)

    lc5 = PlainStaticDynamicFusion.loss_contract()
    lc6 = PairwiseRankNet.loss_contract()
    if lc5.get("ranking_loss") is not False or lc5.get("trigger_gate") is not False:
        raise RuntimeError(f"BASE05 contract mismatch: {lc5}")
    if lc6.get("auxiliary_BCE") is not False:
        raise RuntimeError(f"BASE06 must be pure RankNet: {lc6}")
    if lc6.get("edge_reliability_weight") is not False or lc6.get("trigger_gate") is not False:
        raise RuntimeError(f"BASE06 forbidden component active: {lc6}")

    set_seed(7)
    m5 = PlainStaticDynamicFusion()
    set_seed(7)
    m6 = PairwiseRankNet()
    if sum(p.numel() for p in m5.parameters()) != EXPECTED_PARAMS:
        raise RuntimeError("BASE05 capacity mismatch")
    if sum(p.numel() for p in m6.parameters()) != EXPECTED_PARAMS:
        raise RuntimeError("BASE06 capacity mismatch")

    p5 = list(m5.parameters())
    p6 = list(m6.parameters())
    if len(p5) != len(p6):
        raise RuntimeError("BASE05/06 parameter-list mismatch")
    same_seed_exact = True
    for a, c in zip(p5, p6):
        if a.shape != c.shape:
            raise RuntimeError("BASE05/06 parameter-shape mismatch")
        same_seed_exact = same_seed_exact and torch.equal(a.detach(), c.detach())

    z1 = torch.zeros(3, 91, device=device)
    z2 = torch.zeros(3, dtype=torch.long, device=device)
    z3 = torch.zeros(3, 70, 10, device=device)
    with torch.no_grad():
        o5 = make_model("BASE-05", 7, device)(z1, z2, z3)
        o6 = make_model("BASE-06", 7, device).forward_pairs(z1, z2, z3)
    if o5["risk_score"].shape != (3,) or o6["pair_scores"].shape != (1, 3):
        raise RuntimeError("forward shape contract failed")

    dump = {
        mid: [
            {"candidate_index": i, "raw": c, "normalized": norm_cfg(mid, c)}
            for i, c in enumerate(cfgs[mid], 1)
        ] for mid in ("BASE-05", "BASE-06")
    }
    jwrite(OUT / "23_FROZEN_NEURAL_CANDIDATES.json", dump)
    jwrite(OUT / "23_NEURAL_PREFLIGHT.json", {
        "status": "PASS_NEURAL_PREFLIGHT",
        "dataset": str(base.DATA),
        "samples": base.N,
        "pairs": base.NP,
        "controls": base.NC,
        "dynamic_shape": list(b.dynamic.shape),
        "future_steps_read": 0,
        "BASE05_candidates": len(cfgs["BASE-05"]),
        "BASE06_candidates": len(cfgs["BASE-06"]),
        "BASE05_parameters": EXPECTED_PARAMS,
        "BASE06_parameters": EXPECTED_PARAMS,
        "BASE05_loss_contract": lc5,
        "BASE06_loss_contract": lc6,
        "same_seed_initialization_exact": bool(same_seed_exact),
        "device": str(device),
        "torch_version": torch.__version__,
        "source_hashes": {
            "20A": sha(SCRIPT20A),
            "baseline_tuning": sha(BASE00 / "baseline_tuning.py"),
            "baseline_plain_fusion": sha(BASE00 / "baseline_plain_fusion.py"),
            "baseline_ranknet": sha(BASE00 / "baseline_ranknet.py"),
        },
    })
    return b, cfgs

# ============================================================================
# Fold-only neural preprocessing
# ============================================================================
def robust_fit(x):
    z = np.asarray(x, dtype=np.float64)
    z = z[np.isfinite(z)]
    if len(z) == 0:
        return 0.0, 1.0, True
    med = float(np.median(z))
    q1, q3 = np.quantile(z, [0.25, 0.75])
    iqr = float(q3 - q1)
    zero = (not np.isfinite(iqr)) or iqr <= 0
    return med, (1.0 if zero else iqr), bool(zero)


def maybe_log1p(x, yes):
    z = np.asarray(x, dtype=np.float64)
    if yes:
        finite = np.isfinite(z)
        if (z[finite] < -1).any():
            raise RuntimeError("LOG1P_VALUE_LT_MINUS1")
        z = np.log1p(z)
    return z


def fit_preprocessor(b, train_idx):
    sp = []
    for f in b.cont:
        x = pd.to_numeric(b.static[f], errors="coerce").to_numpy(np.float64)[train_idx]
        use = b.splan.get(f) == "LOG1P_AT_MODEL_STAGE"
        x = maybe_log1p(x, use)
        med, iqr, zero = robust_fit(x)
        sp.append({"field": f, "median": med, "iqr": iqr, "log1p": use, "zero_iqr": zero})

    dp = []
    for j, f in enumerate(base.RAIN_FIELDS):
        x = np.asarray(b.dynamic[train_idx, :, j], dtype=np.float64).reshape(-1)
        use = b.dlog[f]
        x = maybe_log1p(x, use)
        med, iqr, zero = robust_fit(x)
        dp.append({"field": f, "median": med, "iqr": iqr, "log1p": use, "zero_iqr": zero})

    # Frozen neural contract explicitly reserves MISSING and UNKNOWN tokens.
    cat = b.static[b.cat].fillna("MISSING").astype(str).iloc[train_idx]
    real = sorted(v for v in pd.unique(cat).tolist() if v not in {"MISSING", "UNKNOWN"})
    vocab = ["MISSING", "UNKNOWN"] + real
    if len(vocab) > CATEGORY_COUNT:
        raise RuntimeError(f"category vocab incl. MISSING/UNKNOWN={len(vocab)} exceeds {CATEGORY_COUNT}")
    mp = {v: i for i, v in enumerate(vocab)}
    return {
        "static": sp,
        "dynamic": dp,
        "category_field": b.cat,
        "category_map": mp,
        "unknown_index": mp["UNKNOWN"],
        "category_count": CATEGORY_COUNT,
        "fit_sample_count": int(len(train_idx)),
        "dynamic_fit_values_per_field": int(len(train_idx) * 70),
        "future_steps_used": 0,
    }


def transform_all(b, pre):
    cols = []
    for p in pre["static"]:
        x = pd.to_numeric(b.static[p["field"]], errors="coerce").to_numpy(np.float64)
        x = maybe_log1p(x, p["log1p"])
        x = np.where(np.isfinite(x), x, p["median"])
        cols.append(((x - p["median"]) / p["iqr"]).astype(np.float32))
    cols.append(base.soil_indicator(b, np.arange(base.N)).astype(np.float32))
    xs = np.column_stack(cols).astype(np.float32, copy=False)
    if xs.shape != (base.N, 91) or not np.isfinite(xs).all():
        raise RuntimeError(f"static transform failed: {xs.shape}")

    xd = np.empty((base.N, 70, 10), dtype=np.float32)
    for j, p in enumerate(pre["dynamic"]):
        x = np.asarray(b.dynamic[:, :, j], dtype=np.float64)
        x = maybe_log1p(x, p["log1p"])
        x = np.where(np.isfinite(x), x, p["median"])
        xd[:, :, j] = ((x - p["median"]) / p["iqr"]).astype(np.float32)
    if not np.isfinite(xd).all():
        raise RuntimeError("dynamic transform nonfinite")

    raw = b.static[pre["category_field"]].fillna("MISSING").astype(str)
    mp = pre["category_map"]
    unk = int(pre["unknown_index"])
    xc = raw.map(lambda x: mp.get(x, unk)).to_numpy(np.int64)
    if xc.min() < 0 or xc.max() >= CATEGORY_COUNT:
        raise RuntimeError("category index out of range")
    return xs, xc, xd


# ============================================================================
# Pair forward / loss / validation
# ============================================================================
def forward_pairs(model, mid, flat_idx, xs, xc, xd):
    a, c, d = xs[flat_idx], xc[flat_idx], xd[flat_idx]
    if mid == "BASE-05":
        score = model(a, c, d)["risk_score"].reshape(-1, 3)
    else:
        score = model.forward_pairs(a, c, d)["pair_scores"]
    if score.ndim != 2 or score.shape[1] != 3:
        raise RuntimeError(f"bad pair score shape {score.shape}")
    return score


def loss_fn(mid, scores, cfg):
    if mid == "BASE-05":
        target = torch.zeros_like(scores)
        target[:, 0] = 1.0
        weights = torch.tensor([1.0, 0.5, 0.5], device=scores.device).view(1, 3)
        raw = F.binary_cross_entropy_with_logits(scores, target, reduction="none")
        return (raw * weights).mean()
    margin = scores[:, :1] - scores[:, 1:]
    # Standard logistic RankNet; no tau hyperparameter in the frozen BASE-06 grid.
    return F.softplus(-margin).mean()


@torch.no_grad()
def predict_pair_rows(model, mid, pair_rows, b, xs, xc, xd, device, batch_pairs=512):
    model.eval()
    score_all = np.full(base.N, np.nan, dtype=np.float64)
    seen = np.zeros(base.N, dtype=np.int8)
    for st in range(0, len(pair_rows), batch_pairs):
        rr = pair_rows[st:st + batch_pairs]
        trip = b.pt[rr]
        flat_np = trip.reshape(-1)
        flat = torch.as_tensor(flat_np, dtype=torch.long, device=device)
        arr = forward_pairs(model, mid, flat, xs, xc, xd).detach().cpu().numpy().reshape(-1)
        if not np.isfinite(arr).all():
            raise RuntimeError("nonfinite prediction")
        score_all[flat_np] = arr
        seen[flat_np] += 1
    used = b.pt[pair_rows].reshape(-1)
    if not np.all(seen[used] == 1):
        raise RuntimeError("prediction coverage mismatch")
    return score_all


def validation_metrics(model, mid, rows, b, xs, xc, xd, device):
    score = predict_pair_rows(model, mid, rows, b, xs, xc, xd, device)
    p, e = base.pair_tables(b, rows, score, mid, 0, -1)
    idx = b.pt[rows].reshape(-1)
    sc = score[idx]
    prob = 1.0 / (1.0 + np.exp(-np.clip(sc, -60, 60)))
    pred = (prob >= 0.5).astype(np.int8)
    y = b.sample.y_pair.to_numpy(np.int8)[idx]
    return base.metrics(p, e, y, sc, prob, pred)


def val_key(m, tie):
    return (
        float(m["pair_set_concordance"]),
        float(m["edge_wise_ranking_accuracy"]),
        float(m["min_gap_median"]),
        float(m["min_gap_mean"]),
        float(m["AUPRC"]),
        float(tie),
    )


# ============================================================================
# Candidate training with epoch-level resume
# ============================================================================
def candidate_dir(mid, outer, seed, ci):
    return model_dir(mid) / "candidate_runs" / f"FOLD{outer+1}_SEED{seed}_C{ci:02d}"


def save_ckpt(path, model, optimizer, scheduler, extra, *, required=True):
    """
    Windows-safe checkpoint writer.

    Scientific rule:
    - best.pt is REQUIRED: if it cannot be committed after retries, stop.
    - latest.pt is RESUME-ONLY: if Windows/antivirus temporarily locks the
      destination, keep a versioned fallback and continue training.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        **extra,
    }

    # Unique temp name prevents a stale *.tmp from a previous interrupted save
    # from colliding with the current process.
    tmp = path.with_name(
        f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    torch.save(payload, tmp)

    last_error = None
    for attempt in range(12):
        try:
            os.replace(tmp, path)
            return str(path)
        except PermissionError as exc:
            last_error = exc
            # Windows Defender/indexer/another reader can transiently lock the
            # destination. Back off instead of aborting a valid training run.
            time.sleep(min(0.15 * (attempt + 1), 1.5))
        except OSError as exc:
            last_error = exc
            time.sleep(min(0.15 * (attempt + 1), 1.5))

    if not required:
        epoch = int(extra.get("epoch", -1))
        fallback = path.with_name(
            f"{path.stem}_fallback_epoch{epoch:04d}_{os.getpid()}.pt"
        )
        try:
            os.replace(tmp, fallback)
            log(
                f"WARNING checkpoint destination locked: {path.name}; "
                f"resume fallback saved as {fallback.name}"
            )
            return str(fallback)
        except Exception as fallback_exc:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            log(
                f"WARNING latest checkpoint save skipped after retries: "
                f"{type(last_error).__name__}: {last_error}; "
                f"fallback also failed: {fallback_exc}"
            )
            return None

    # best.pt is part of the formal selected-model evidence and cannot be skipped.
    try:
        if tmp.exists():
            tmp.unlink()
    except Exception:
        pass
    raise PermissionError(
        f"Required checkpoint could not be committed after retries: {path}; "
        f"last_error={last_error}"
    )


def train_candidate(mid, outer, seed, ci, raw_cfg, train_cfg, train_rows, val_rows, b, xs, xc, xd, device, restart=False):
    run = candidate_dir(mid, outer, seed, ci)
    run.mkdir(parents=True, exist_ok=True)
    config = {
        "mid": mid, "outer_fold": outer, "seed": seed, "candidate_index": ci,
        "raw_candidate": raw_cfg, "normalized": train_cfg,
        "parameter_count": EXPECTED_PARAMS, "dropout": DROPOUT,
        "train_pairs": int(len(train_rows)), "validation_pairs": int(len(val_rows)),
        "outer_test_used_for_selection": False, "refit": False,
    }
    ctext = json.dumps(config, sort_keys=True, default=str)
    chash = hashlib.sha256(ctext.encode()).hexdigest()
    jwrite(run / "CONFIG.json", {**config, "config_sha256": chash})

    complete = run / "COMPLETE.json"
    best = run / "best.pt"
    latest = run / "latest.pt"
    hist = run / "training_history.csv"
    if complete.exists():
        r = jread(complete)
        if r.get("config_sha256") != chash:
            raise RuntimeError(f"completed config mismatch: {run}")
        log(f"reuse {mid} F{outer+1} seed={seed} C{ci:02d}")
        return r

    if restart:
        for p in (best, latest, hist):
            if p.exists():
                p.unlink()

    model = make_model(mid, seed, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5)

    start_epoch, best_key, best_epoch, bad, steps = 1, None, 0, 0, 0
    history = []
    # Resume from the newest valid resume checkpoint. Normally this is
    # latest.pt; a versioned fallback is considered only when Windows blocked
    # an atomic overwrite in an earlier process.
    resume_candidates = []
    if latest.exists():
        resume_candidates.append(latest)
    resume_candidates.extend(sorted(run.glob("latest_fallback_epoch*.pt")))

    resume_ck = None
    resume_path = None
    resume_epoch = -1
    for rp in resume_candidates:
        try:
            cand = torch.load(rp, map_location="cpu")
            if cand.get("config_sha256") != chash:
                continue
            ep = int(cand.get("epoch", -1))
            if ep > resume_epoch:
                resume_ck = cand
                resume_path = rp
                resume_epoch = ep
        except Exception as exc:
            log(f"WARNING unreadable resume checkpoint ignored: {rp.name}: {exc}")

    if resume_ck is not None:
        ck = resume_ck
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        scheduler.load_state_dict(ck["scheduler_state"])
        start_epoch = int(ck["epoch"]) + 1
        best_key = tuple(ck["best_key"]) if ck.get("best_key") is not None else None
        best_epoch = int(ck.get("best_epoch", 0))
        bad = int(ck.get("bad", 0))
        steps = int(ck.get("steps", 0))
        if hist.exists():
            hh = pd.read_csv(hist)
            # A failed checkpoint write can leave history one epoch ahead of
            # the resume state. Trim it to the committed checkpoint.
            hh = hh[pd.to_numeric(hh["epoch"], errors="coerce") <= int(ck["epoch"])]
            history = hh.to_dict("records")
            hh.to_csv(hist, index=False, encoding="utf-8-sig")
        log(
            f"RESUME {mid} F{outer+1} seed={seed} C{ci:02d} "
            f"from {resume_path.name} at epoch={start_epoch}"
        )

    stop_reason = "MAX_EPOCH_REACHED"
    started = time.perf_counter()
    for epoch in range(start_epoch, train_cfg["max_epochs"] + 1):
        # Epoch-specific RNG keeps dropout and pair order deterministic across resume.
        epoch_seed = seed * 1000003 + outer * 10007 + ci * 101 + epoch
        torch.manual_seed(epoch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(epoch_seed)
        model.train()
        gen = torch.Generator().manual_seed(epoch_seed)
        order = torch.randperm(len(train_rows), generator=gen).numpy()
        total, seen_pairs, grad_max = 0.0, 0, 0.0
        bs = train_cfg["pairs_per_batch"]

        for st in range(0, len(order), bs):
            rr = train_rows[order[st:st + bs]]
            flat = torch.as_tensor(b.pt[rr].reshape(-1), dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            scores = forward_pairs(model, mid, flat, xs, xc, xd)
            loss = loss_fn(mid, scores, train_cfg)
            if not torch.isfinite(loss):
                raise RuntimeError("NONFINITE_LOSS")
            loss.backward()
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise RuntimeError("NONFINITE_GRADIENT")
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
            if not np.isfinite(gn):
                raise RuntimeError("NONFINITE_GRAD_NORM")
            grad_max = max(grad_max, gn)
            optimizer.step()
            steps += 1
            nb = len(rr)
            total += float(loss.detach().cpu()) * nb
            seen_pairs += nb

        train_loss = total / seen_pairs
        vm = validation_metrics(model, mid, val_rows, b, xs, xc, xd, device)
        key = val_key(vm, -epoch)
        is_best = best_key is None or key > best_key
        if is_best:
            best_key, best_epoch, bad = key, epoch, 0
            save_ckpt(best, model, optimizer, scheduler, {
                "epoch": epoch, "best_epoch": best_epoch, "best_key": list(best_key),
                "bad": bad, "steps": steps, "config_sha256": chash,
                "validation_metrics": vm,
            })
        else:
            bad += 1

        scheduler.step(vm["pair_set_concordance"])
        row = {
            "epoch": epoch, "train_loss": train_loss,
            "validation_pair_set_concordance": vm["pair_set_concordance"],
            "validation_edge_wise_ranking_accuracy": vm["edge_wise_ranking_accuracy"],
            "validation_min_gap_median": vm["min_gap_median"],
            "validation_min_gap_mean": vm["min_gap_mean"],
            "validation_AUPRC": vm["AUPRC"], "validation_AUROC": vm["AUROC"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "max_gradient_norm_before_clip": grad_max,
            "is_best": is_best, "best_epoch_so_far": best_epoch, "early_stop_counter": bad,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(hist, index=False, encoding="utf-8-sig")
        save_ckpt(latest, model, optimizer, scheduler, {
            "epoch": epoch, "best_epoch": best_epoch,
            "best_key": list(best_key) if best_key is not None else None,
            "bad": bad, "steps": steps, "config_sha256": chash,
        }, required=False)
        log(
            f"{mid} F{outer+1} s{seed} C{ci:02d} e{epoch:03d} "
            f"loss={train_loss:.6f} valPair={vm['StrictPair']:.4f} "
            f"valEdge={vm['Edge']:.4f} AP={vm['AUPRC']:.4f} best={best_epoch} bad={bad}/{train_cfg['patience']}"
        )
        if epoch >= train_cfg["min_epochs"] and bad >= train_cfg["patience"]:
            stop_reason = "EARLY_STOPPING"
            break

    if not best.exists():
        raise RuntimeError(f"best checkpoint missing: {run}")
    bck = torch.load(best, map_location="cpu")
    vm = bck["validation_metrics"]
    result = {
        **config, "config_sha256": chash,
        "best_epoch": int(bck["best_epoch"]), "stop_epoch": int(history[-1]["epoch"]),
        "stop_reason": stop_reason, "optimizer_steps": int(steps),
        "training_seconds_this_process": time.perf_counter() - started,
        "best_checkpoint": str(best), "best_checkpoint_sha256": sha(best),
        **{"validation_" + k: v for k, v in vm.items()},
        "status": "PASS_CANDIDATE_COMPLETED",
    }
    jwrite(complete, result)
    return result

# ============================================================================
# Candidate selection and outer-test evaluation
# ============================================================================
def select_candidate(results):
    def key(r):
        return (
            float(r["validation_pair_set_concordance"]),
            float(r["validation_edge_wise_ranking_accuracy"]),
            float(r["validation_min_gap_median"]),
            float(r["validation_min_gap_mean"]),
            float(r["validation_AUPRC"]),
            -int(r["candidate_index"]),
        )
    return max(results, key=key)


def evaluate_selected(mid, outer, seed, selected, test_pair_rows, b, xs, xc, xd, device):
    model = make_model(mid, seed, device)
    ck = torch.load(Path(selected["best_checkpoint"]), map_location="cpu")
    model.load_state_dict(ck["model_state"])
    model.to(device)

    # Exactly one outer-test scoring pass after validation-only selection.
    score_all = predict_pair_rows(model, mid, test_pair_rows, b, xs, xc, xd, device)
    pp, ee = base.pair_tables(b, test_pair_rows, score_all, mid, seed, outer)
    test_idx = base.srows(b, [outer])
    score = np.asarray(score_all[test_idx], dtype=np.float64)
    prob = 1.0 / (1.0 + np.exp(-np.clip(score, -60, 60)))
    pred = (prob >= 0.5).astype(np.int8)
    ss = base.sample_frame(
        b, test_idx, mid, seed, outer, score, prob, pred,
        {
            "candidate_index": int(selected["candidate_index"]),
            "selected_configuration": json.dumps(selected["raw_candidate"], sort_keys=True),
            "best_epoch": int(selected["best_epoch"]),
        },
    )
    mm = base.metrics(pp, ee, ss.y_true, ss.risk_score, ss.probability, ss.predicted_label)
    row = {
        "experiment_id": mid,
        "outer_fold": outer,
        "seed": seed,
        "candidate_index": int(selected["candidate_index"]),
        "selected_configuration": json.dumps(selected["raw_candidate"], sort_keys=True),
        "best_epoch": int(selected["best_epoch"]),
        "stop_epoch": int(selected["stop_epoch"]),
        "outer_test_forward_count": 1,
        "refit_performed": "NO",
        "outer_test_used_for_selection": "NO",
        **mm,
    }
    return ss, pp, ee, row


# ============================================================================
# Full formal neural baseline runner
# ============================================================================
def run_model(b, cfgs, mid, args, device):
    d = model_dir(mid)
    if (d / "COMPLETED.flag").exists() and not args.force:
        log(f"{mid} already complete; reusing frozen repaired-V2 output")
        return jread(d / "MODEL_SUMMARY.json")

    raw_configs = cfgs[mid]
    sfs, pfs, efs = [], [], []
    foldrows, candrows, selrows, prerows = [], [], [], []

    for outer in range(5):
        val = (outer + 1) % 5
        train_folds = sorted(set(range(5)) - {outer, val})
        train_idx = base.srows(b, train_folds)
        val_idx = base.srows(b, [val])
        test_idx = base.srows(b, [outer])
        train_pairs = base.prows(b, train_folds)
        val_pairs = base.prows(b, [val])
        test_pairs = base.prows(b, [outer])

        log(
            f"{mid} FOLD{outer+1}: train={train_folds}, validation=FOLD{val+1}, "
            f"outer-test=FOLD{outer+1}; fitting fold-only preprocessor"
        )
        pre = fit_preprocessor(b, train_idx)
        pre_path = d / "preprocessors" / f"FOLD{outer+1}.json"
        jwrite(pre_path, pre)
        xs_np, xc_np, xd_np = transform_all(b, pre)
        unk = int(pre["unknown_index"])
        prerows.append({
            "experiment_id": mid,
            "outer_fold": outer,
            "fit_scope": "TRAINING_FOLDS_ONLY",
            "training_folds": "|".join(map(str, train_folds)),
            "validation_fold": val,
            "test_fold": outer,
            "fit_samples": int(len(train_idx)),
            "validation_fit_rows": 0,
            "test_fit_rows": 0,
            "static_raw_continuous": 90,
            "soil_missing_indicator": 1,
            "static_model_continuous": 91,
            "categorical_raw_fields": 1,
            "category_vocab_train_unique": int(len(pre["category_map"])),
            "category_unknown_index": unk,
            "validation_unknown_rows": int((xc_np[val_idx] == unk).sum()),
            "test_unknown_rows": int((xc_np[test_idx] == unk).sum()),
            "dynamic_steps": 70,
            "dynamic_fields": 10,
            "dynamic_fit_values_per_field": int(len(train_idx) * 70),
            "future_steps_used": 0,
            "static_nan_inf": int(np.size(xs_np) - np.isfinite(xs_np).sum()),
            "dynamic_nan_inf": int(np.size(xd_np) - np.isfinite(xd_np).sum()),
            "preprocessor_path": str(pre_path),
            "preprocessor_sha256": sha(pre_path),
        })

        xs = torch.as_tensor(xs_np, dtype=torch.float32, device=device)
        xc = torch.as_tensor(xc_np, dtype=torch.long, device=device)
        xd = torch.as_tensor(xd_np, dtype=torch.float32, device=device)

        for seed in SEEDS:
            log(f"{mid} FOLD{outer+1} seed={seed}: {len(raw_configs)} frozen candidates")
            results = []
            for ci, raw_cfg in enumerate(raw_configs, 1):
                train_cfg = norm_cfg(mid, raw_cfg)
                r = train_candidate(
                    mid, outer, seed, ci, raw_cfg, train_cfg,
                    train_pairs, val_pairs, b, xs, xc, xd, device,
                    restart=args.restart_incomplete,
                )
                results.append(r)
                candrows.append({
                    "experiment_id": mid,
                    "outer_fold": outer,
                    "seed": seed,
                    "candidate_index": ci,
                    "configuration": json.dumps(raw_cfg, sort_keys=True),
                    "best_epoch": r["best_epoch"],
                    "stop_epoch": r["stop_epoch"],
                    "stop_reason": r["stop_reason"],
                    "validation_pair_set_concordance": r["validation_pair_set_concordance"],
                    "validation_edge_wise_ranking_accuracy": r["validation_edge_wise_ranking_accuracy"],
                    "validation_min_gap_median": r["validation_min_gap_median"],
                    "validation_min_gap_mean": r["validation_min_gap_mean"],
                    "validation_AUPRC": r["validation_AUPRC"],
                    "validation_AUROC": r["validation_AUROC"],
                    "outer_test_used_for_selection": "NO",
                })

            selected = select_candidate(results)
            selrows.append({
                "experiment_id": mid,
                "outer_fold": outer,
                "seed": seed,
                "training_folds": "|".join(map(str, train_folds)),
                "validation_fold": val,
                "test_fold": outer,
                "candidate_index": int(selected["candidate_index"]),
                "selected_configuration": json.dumps(selected["raw_candidate"], sort_keys=True),
                "best_epoch": int(selected["best_epoch"]),
                "validation_pair_set_concordance": selected["validation_pair_set_concordance"],
                "validation_edge_wise_ranking_accuracy": selected["validation_edge_wise_ranking_accuracy"],
                "validation_min_gap_median": selected["validation_min_gap_median"],
                "validation_min_gap_mean": selected["validation_min_gap_mean"],
                "validation_AUPRC": selected["validation_AUPRC"],
                "refit_performed": "NO",
                "outer_test_used_for_selection": "NO",
            })
            log(
                f"SELECT {mid} FOLD{outer+1} seed={seed}: C{selected['candidate_index']:02d} "
                f"best_epoch={selected['best_epoch']}; outer-test now scored once"
            )

            ss, pp, ee, fr = evaluate_selected(
                mid, outer, seed, selected, test_pairs, b, xs, xc, xd, device
            )
            sfs.append(ss)
            pfs.append(pp)
            efs.append(ee)
            foldrows.append(fr)
            log(
                f"TEST {mid} FOLD{outer+1} seed={seed} "
                f"StrictPair={fr['StrictPair']:.4f} Edge={fr['Edge']:.4f} "
                f"AUPRC={fr['AUPRC']:.4f} AUROC={fr['AUROC']:.4f}"
            )

        del xs, xc, xd, xs_np, xc_np, xd_np
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    extra = {
        "baseline_type": "PLAIN_STATIC_DYNAMIC_FUSION" if mid == "BASE-05" else "PAIRWISE_RANKNET",
        "trainable_parameters": EXPECTED_PARAMS,
        "static_model_continuous": 91,
        "categorical_embedding_count": CATEGORY_COUNT,
        "dynamic_shape": [70, 10],
        "dropout": DROPOUT,
        "gradient_clip": GRAD_CLIP,
        "candidate_count_per_fold_seed": len(raw_configs),
        "loss_contract": PlainStaticDynamicFusion.loss_contract() if mid == "BASE-05" else PairwiseRankNet.loss_contract(),
        "neural_preprocessing": "train-fold robust median/IQR; dynamic fit uses train samples x allowed 70 steps",
        "reference_time_utc": str(base.REF),
        "outer_test_forward_count_per_fold_seed": 1,
    }
    return base.finalize(
        mid, sfs, pfs, efs, foldrows, candrows, selrows, prerows,
        args.bootstrap, extra,
    )


# ============================================================================
# Six-baseline comparison + BASE-06 vs BASE-05 pair transition audit
# ============================================================================
def update_comparison():
    rows = []
    for mid in ["BASE-01", "BASE-02", "BASE-03", "BASE-04", "BASE-05", "BASE-06"]:
        p = model_dir(mid) / "MODEL_SUMMARY.json"
        if not p.exists():
            continue
        s = jread(p)
        if s.get("status") != "PASS":
            continue
        m, sd = s["seed_mean"], s["seed_std"]
        rows.append({
            "experiment_id": mid,
            "StrictPair_mean": m["StrictPair"], "StrictPair_std": sd["StrictPair"],
            "Edge_mean": m["Edge"], "Edge_std": sd["Edge"],
            "AUPRC_mean": m["AUPRC"], "AUPRC_std": sd["AUPRC"],
            "AUROC_mean": m["AUROC"], "AUROC_std": sd["AUROC"],
            "balanced_accuracy_mean": m["balanced_accuracy"],
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["StrictPair_mean", "Edge_mean", "AUPRC_mean"], ascending=False)
    df.to_csv(OUT / "FORMAL_BASELINE_COMPARISON.csv", index=False, encoding="utf-8-sig")
    return df


def base06_vs_base05_transitions():
    p5 = model_dir("BASE-05") / "PAIR_OOF.parquet"
    p6 = model_dir("BASE-06") / "PAIR_OOF.parquet"
    if not p5.exists() or not p6.exists():
        return None
    a = pd.read_parquet(p5)
    b = pd.read_parquet(p6)
    rows = []
    for seed in SEEDS:
        x = a[a.seed == seed].sort_values("pair_set_id", kind="stable").reset_index(drop=True)
        y = b[b.seed == seed].sort_values("pair_set_id", kind="stable").reset_index(drop=True)
        if not x.pair_set_id.equals(y.pair_set_id):
            raise RuntimeError("BASE05_BASE06_PAIR_ALIGNMENT")
        cx = x.pair_concordant.to_numpy(np.int8)
        cy = y.pair_concordant.to_numpy(np.int8)
        ex = x[["gap_1", "gap_2"]].to_numpy() > 0
        ey = y[["gap_1", "gap_2"]].to_numpy() > 0
        rows.append({
            "seed": seed,
            "discordant_to_concordant_pairs": int(((cx == 0) & (cy == 1)).sum()),
            "concordant_to_discordant_pairs": int(((cx == 1) & (cy == 0)).sum()),
            "net_concordant_pair_gain": int(cy.sum() - cx.sum()),
            "net_correct_edge_gain": int(ey.sum() - ex.sum()),
            "BASE05_StrictPair": float(cx.mean()),
            "BASE06_StrictPair": float(cy.mean()),
            "BASE05_Edge": float(ex.mean()),
            "BASE06_Edge": float(ey.mean()),
        })
    out = pd.DataFrame(rows)
    path = OUT / "BASE06_VS_BASE05_TRANSITIONS.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path


# ============================================================================
# Main
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="BASE-05", help="BASE-05, BASE-06, or BASE-05,BASE-06")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="rerun model finalization; completed candidates are reused")
    ap.add_argument("--restart-incomplete", action="store_true")
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    ap.add_argument("--torch-threads", type=int, default=1)
    args = ap.parse_args()

    if args.bootstrap < 0 or args.torch_threads < 1:
        raise ValueError("invalid bootstrap/torch-threads")
    torch.set_num_threads(args.torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_REQUESTED_BUT_NOT_AVAILABLE")
        device = torch.device("cuda")
    elif args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    models = [x.strip().upper() for x in args.models.split(",") if x.strip()]
    invalid = [x for x in models if x not in {"BASE-05", "BASE-06"}]
    if invalid:
        raise ValueError("23 supports BASE-05/BASE-06 only: " + ",".join(invalid))

    OUT.mkdir(parents=True, exist_ok=True)
    log("23 repaired-V2 neural baseline stage started")
    log(f"python={sys.executable}")
    log(f"torch={torch.__version__}; device={device}; torch_threads={args.torch_threads}")

    b, cfgs = preflight(device)
    log(
        f"PREFLIGHT PASS: BASE05 candidates={len(cfgs['BASE-05'])}, "
        f"BASE06 candidates={len(cfgs['BASE-06'])}"
    )

    if args.preflight_only:
        final = {
            "status": "PASS_NEURAL_PREFLIGHT_ONLY",
            "requested": models,
            "next_recommended": "RUN_BASE05_REPAIRED_V2",
        }
        jwrite(OUT / "23_FINAL_STATUS.json", final)
        print("FINAL_STATUS=PASS_NEURAL_PREFLIGHT_ONLY", flush=True)
        print(OUT / "23_NEURAL_PREFLIGHT.json", flush=True)
        print(OUT / "23_FROZEN_NEURAL_CANDIDATES.json", flush=True)
        return 0

    summaries = {}
    for mid in models:
        if mid == "BASE-06" and not (model_dir("BASE-05") / "COMPLETED.flag").exists():
            raise RuntimeError("BASE06_BLOCKED_UNTIL_REPAIRED_V2_BASE05_COMPLETES")
        summaries[mid] = run_model(b, cfgs, mid, args, device)
        comp = update_comparison()
        if not comp.empty:
            print("\nCURRENT BASELINE COMPARISON", flush=True)
            print(
                comp[["experiment_id", "StrictPair_mean", "Edge_mean", "AUPRC_mean", "AUROC_mean"]].to_string(index=False),
                flush=True,
            )

    transition_path = base06_vs_base05_transitions()
    completed = [m for m in ["BASE-05", "BASE-06"] if (model_dir(m) / "COMPLETED.flag").exists()]
    final = {
        "status": "PASS_NEURAL_BASELINES_REQUEST_COMPLETED",
        "requested": models,
        "completed_neural": completed,
        "comparison": str(OUT / "FORMAL_BASELINE_COMPARISON.csv"),
        "BASE06_vs_BASE05_transitions": str(transition_path) if transition_path else None,
    }
    jwrite(OUT / "23_FINAL_STATUS.json", final)

    print("\n" + "=" * 112, flush=True)
    print("23 FORMAL REPAIRED-V2 NEURAL BASELINE STAGE COMPLETE", flush=True)
    print("=" * 112, flush=True)
    print("FINAL_STATUS:", final["status"], flush=True)
    print("Please send:", flush=True)
    print(OUT / "FORMAL_BASELINE_COMPARISON.csv", flush=True)
    print(OUT / "23_FINAL_STATUS.json", flush=True)
    for mid in models:
        print(model_dir(mid) / "MODEL_SUMMARY.json", flush=True)
        print(model_dir(mid) / "FOLD_SEED_METRICS.csv", flush=True)
        print(model_dir(mid) / "SELECTED_CONFIGS.csv", flush=True)
    print("=" * 112, flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        tb = traceback.format_exc()
        print(tb, flush=True)
        jwrite(OUT / "23_FINAL_STATUS.json", {
            "status": "FAIL_NEURAL_BASELINE_STAGE",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": tb,
        })
        raise
