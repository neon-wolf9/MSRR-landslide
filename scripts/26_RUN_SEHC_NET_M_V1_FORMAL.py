#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Formal repaired-V2 five-fold runner for S-EHC-Net-M v1.

Training framework authority:
  scripts/24_RUN_BASE06_REPAIRED_V2.py ->
  scripts/23D_RUN_NEURAL_BASELINES_REPAIRED_V2.py

The BASE06 data loader, frozen spatial split, fold-only preprocessing, complete
pair batching, AdamW optimizer, ReduceLROnPlateau scheduler, validation key,
gradient clipping, atomic checkpointing and one-pass outer-test protocol are
retained. Only the model, objective and trigger interpretation outputs differ.

This module does not start training when imported. Run it explicitly to launch
the 5 folds x 3 seeds formal experiment.
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
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT = Path(__file__).resolve().parents[1]
BASE06_LAUNCHER = PROJECT / "scripts" / "24_RUN_BASE06_REPAIRED_V2.py"
BASE06_RUNNER = PROJECT / "scripts" / "23D_RUN_NEURAL_BASELINES_REPAIRED_V2.py"
MODEL_FILE = PROJECT / "models" / "sehc_net_m_v1.py"
LOSS_FILE = PROJECT / "models" / "loss_sehc.py"
TRIGGER_SCHEMA = (
    PROJECT / "modeling" / "MODEL-00" / "00C_model_experiment_contract"
    / "run_20260805_151152" / "MODEL00C_TRIGGER_LIBRARY.csv"
)
OUTPUT = PROJECT / "experiments" / "SEHC_NET_M_V1_FORMAL_V1"

MODEL_NAME = "S-EHC-Net-M-v1"
EXPERIMENT_ID = "SEHC_NET_M_V1_FORMAL_V1"
SEEDS = [7, 11, 21]
FOLDS = list(range(5))
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
MIN_EPOCHS = 1

# Frozen BASE06 candidate values (candidate grid member C09).
PAIRS_PER_BATCH = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0
DROPOUT = 0.10
SCHEDULER_PATIENCE = 5

LAMBDA_RANK = 1.0
LAMBDA_BCE = 2.0
LAMBDA_SPARSE = 0.005
RANK_TEMPERATURE = 1.0
ACTIVE_TRIGGER_THRESHOLD = 0.5
TRIGGER_COUNT = 83
CATEGORY_COUNT = 14

if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from models.loss_sehc import SEHCNetLoss
from models.sehc_net_m_v1 import SEHCNetM


def import_base06_runner():
    """Load the formal repaired-V2 BASE06 implementation without running it."""
    spec = importlib.util.spec_from_file_location("sehc_formal_base06_authority", BASE06_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import BASE06 runner: {BASE06_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def log(message: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with (OUTPUT / "FORMAL_RUN.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def fairness_banner() -> None:
    print("MODEL=S-EHC-Net-M-v1", flush=True)
    print("DATA_PROTOCOL=SAME_AS_BASE06", flush=True)
    print("SPLIT=SAME_SPATIAL_5FOLD", flush=True)
    print("SEED=[7,11,21]", flush=True)
    print("LOSS=lambda_rank*rank+", flush=True)
    print("lambda_bce*bce+", flush=True)
    print("lambda_sparse*sparse", flush=True)


def validate_required_assets() -> pd.DataFrame:
    required = {
        "BASE06 formal launcher": BASE06_LAUNCHER,
        "BASE06 formal runner": BASE06_RUNNER,
        "S-EHC-Net-M v1 model": MODEL_FILE,
        "S-EHC loss": LOSS_FILE,
        "frozen trigger schema": TRIGGER_SCHEMA,
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required formal assets:\n" + "\n".join(missing))
    schema = pd.read_csv(TRIGGER_SCHEMA)
    if len(schema) != TRIGGER_COUNT or schema["trigger_id"].nunique() != TRIGGER_COUNT:
        raise RuntimeError("trigger schema must contain exactly 83 unique trigger_id values")
    return schema


def build_static_92(xs91: np.ndarray, category_index: np.ndarray) -> np.ndarray:
    """Preserve BASE06 preprocessing and bridge its 91+category model interface."""
    category_channel = np.asarray(category_index, dtype=np.float32) / float(CATEGORY_COUNT - 1)
    static92 = np.column_stack([xs91, category_channel]).astype(np.float32, copy=False)
    if static92.shape[1] != 92 or not np.isfinite(static92).all():
        raise RuntimeError(f"static adapter expected [N,92], observed {static92.shape}")
    return static92


def run_dir(outer_fold: int, seed: int) -> Path:
    return OUTPUT / f"fold{outer_fold + 1}_seed{seed}"


def make_model(seed: int, device: torch.device) -> SEHCNetM:
    set_seed(seed)
    model = SEHCNetM(dropout=DROPOUT, init_gate_logit=-1.5).to(device)
    return model


def config_payload(outer_fold: int, seed: int, train_pairs: int, validation_pairs: int) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "model": MODEL_NAME,
        "outer_fold": outer_fold,
        "human_fold": outer_fold + 1,
        "seed": seed,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "min_epochs": MIN_EPOCHS,
        "pairs_per_batch": PAIRS_PER_BATCH,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "ReduceLROnPlateau(mode=max, patience=5)",
        "gradient_clip": GRAD_CLIP,
        "loss": {
            "lambda_rank": LAMBDA_RANK,
            "lambda_bce": LAMBDA_BCE,
            "lambda_sparse": LAMBDA_SPARSE,
            "rank_temperature": RANK_TEMPERATURE,
        },
        "train_pairs": train_pairs,
        "validation_pairs": validation_pairs,
        "outer_test_used_for_selection": False,
        "refit": False,
        "data_protocol": "SAME_AS_BASE06_REPAIRED_V2",
    }


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def atomic_save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    extra: dict[str, Any],
    *,
    required: bool,
) -> str | None:
    """BASE06-compatible Windows-safe atomic checkpoint writer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        **extra,
    }
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    torch.save(payload, temporary)
    last_error: Exception | None = None
    for attempt in range(12):
        try:
            os.replace(temporary, path)
            return str(path)
        except (PermissionError, OSError) as exc:
            last_error = exc
            time.sleep(min(0.15 * (attempt + 1), 1.5))
    if not required:
        epoch = int(extra.get("epoch", -1))
        fallback = path.with_name(f"{path.stem}_fallback_epoch{epoch:04d}_{os.getpid()}.pt")
        try:
            os.replace(temporary, fallback)
            log(f"WARNING latest checkpoint locked; saved {fallback.name}")
            return str(fallback)
        except Exception as fallback_error:
            if temporary.exists():
                try:
                    temporary.unlink()
                except Exception:
                    pass
            log(f"WARNING latest checkpoint skipped: {last_error}; fallback: {fallback_error}")
            return None
    if temporary.exists():
        try:
            temporary.unlink()
        except Exception:
            pass
    raise PermissionError(f"required checkpoint could not be committed: {path}; {last_error}")


def forward_pair_batch(
    model: SEHCNetM,
    flat_indices: torch.Tensor,
    static92: torch.Tensor,
    rain: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    static_x = static92[flat_indices]
    rain_seq = rain[flat_indices]
    labels = torch.zeros(len(flat_indices), dtype=torch.float32, device=flat_indices.device)
    labels[::3] = 1.0
    roles = labels.bool()
    outputs = model(static_x, rain_seq, pair_role=roles)
    required = {
        "pair_score", "global_risk", "trigger_weights", "event_embedding", "sparse_activation"
    }
    missing = required.difference(outputs)
    if missing:
        raise RuntimeError(f"model output missing: {sorted(missing)}")
    return outputs, labels, roles


@torch.no_grad()
def predict_pair_rows(
    model: SEHCNetM,
    pair_rows: np.ndarray,
    bundle: Any,
    static92: torch.Tensor,
    rain: torch.Tensor,
    device: torch.device,
    *,
    collect_triggers: bool,
    batch_pairs: int = 512,
) -> dict[str, np.ndarray]:
    """Same complete-pair prediction traversal as BASE06; optionally collect triggers."""
    model.eval()
    n = int(bundle.sample.shape[0])
    pair_score = np.full(n, np.nan, dtype=np.float64)
    global_risk = np.full(n, np.nan, dtype=np.float64)
    seen = np.zeros(n, dtype=np.int8)
    trigger_weights = (
        np.full((n, TRIGGER_COUNT), np.nan, dtype=np.float32) if collect_triggers else None
    )
    for start in range(0, len(pair_rows), batch_pairs):
        rows = pair_rows[start : start + batch_pairs]
        flat_np = bundle.pt[rows].reshape(-1)
        flat = torch.as_tensor(flat_np, dtype=torch.long, device=device)
        outputs, _, _ = forward_pair_batch(model, flat, static92, rain)
        scores = outputs["pair_score"].detach().cpu().numpy().reshape(-1)
        risks = outputs["global_risk"].detach().cpu().numpy().reshape(-1)
        if not np.isfinite(scores).all() or not np.isfinite(risks).all():
            raise RuntimeError("nonfinite prediction")
        pair_score[flat_np] = scores
        global_risk[flat_np] = risks
        seen[flat_np] += 1
        if trigger_weights is not None:
            trigger_weights[flat_np] = outputs["trigger_weights"].detach().cpu().numpy()
    used = bundle.pt[pair_rows].reshape(-1)
    if not np.all(seen[used] == 1):
        raise RuntimeError("prediction coverage mismatch")
    result = {"pair_score": pair_score, "global_risk": global_risk}
    if trigger_weights is not None:
        if not np.isfinite(trigger_weights[used]).all():
            raise RuntimeError("nonfinite trigger weights")
        result["trigger_weights"] = trigger_weights
    return result


def metric_frames(
    predictions: dict[str, np.ndarray],
    pair_rows: np.ndarray,
    bundle: Any,
    base: Any,
    seed: int,
    outer_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    pair_frame, edge_frame = base.pair_tables(
        bundle, pair_rows, predictions["pair_score"], EXPERIMENT_ID, seed, outer_fold
    )
    used = bundle.pt[pair_rows].reshape(-1)
    labels = bundle.sample.y_pair.to_numpy(np.int8)[used]
    probability = np.clip(predictions["global_risk"][used], 1e-7, 1.0 - 1e-7)
    logit = np.log(probability / (1.0 - probability))
    predicted = (probability >= 0.5).astype(np.int8)
    metrics = base.metrics(pair_frame, edge_frame, labels, logit, probability, predicted)
    return pair_frame, edge_frame, metrics


def validation_metrics(
    model: SEHCNetM,
    pair_rows: np.ndarray,
    bundle: Any,
    static92: torch.Tensor,
    rain: torch.Tensor,
    base: Any,
    device: torch.device,
) -> dict[str, float]:
    predictions = predict_pair_rows(
        model, pair_rows, bundle, static92, rain, device, collect_triggers=False
    )
    _, _, metrics = metric_frames(predictions, pair_rows, bundle, base, 0, -1)
    return metrics


def validation_key(metrics: dict[str, float], tie: int) -> tuple[float, ...]:
    """Unchanged BASE06 validation-only selection priority."""
    return (
        float(metrics["pair_set_concordance"]),
        float(metrics["edge_wise_ranking_accuracy"]),
        float(metrics["min_gap_median"]),
        float(metrics["min_gap_mean"]),
        float(metrics["AUPRC"]),
        float(tie),
    )


def restore_latest(
    directory: Path,
    expected_hash: str,
    model: SEHCNetM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
) -> tuple[int, tuple[float, ...] | None, int, int, int, list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = []
    latest = directory / "checkpoint_latest.pt"
    if latest.exists():
        candidates.append(latest)
    candidates.extend(sorted(directory.glob("checkpoint_latest_fallback_epoch*.pt")))
    chosen = None
    chosen_path = None
    chosen_epoch = -1
    for path in candidates:
        try:
            candidate = torch.load(path, map_location="cpu")
            if candidate.get("config_sha256") != expected_hash:
                continue
            epoch = int(candidate.get("epoch", -1))
            if epoch > chosen_epoch:
                chosen, chosen_path, chosen_epoch = candidate, path, epoch
        except Exception as exc:
            log(f"WARNING unreadable resume checkpoint ignored: {path.name}: {exc}")
    if chosen is None:
        return 1, None, 0, 0, 0, [], []
    model.load_state_dict(chosen["model_state"])
    optimizer.load_state_dict(chosen["optimizer_state"])
    scheduler.load_state_dict(chosen["scheduler_state"])
    train_log_path = directory / "train_log.csv"
    val_log_path = directory / "val_metrics.csv"
    train_history = pd.read_csv(train_log_path).to_dict("records") if train_log_path.exists() else []
    val_history = pd.read_csv(val_log_path).to_dict("records") if val_log_path.exists() else []
    train_history = [row for row in train_history if int(row["epoch"]) <= chosen_epoch]
    val_history = [row for row in val_history if int(row["epoch"]) <= chosen_epoch]
    log(f"RESUME {directory.name} from {chosen_path.name} at epoch={chosen_epoch + 1}")
    key = tuple(chosen["best_key"]) if chosen.get("best_key") is not None else None
    return (
        chosen_epoch + 1,
        key,
        int(chosen.get("best_epoch", 0)),
        int(chosen.get("bad", 0)),
        int(chosen.get("steps", 0)),
        train_history,
        val_history,
    )


def train_fold_seed(
    outer_fold: int,
    seed: int,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    bundle: Any,
    static92: torch.Tensor,
    rain: torch.Tensor,
    base: Any,
    device: torch.device,
) -> dict[str, Any]:
    directory = run_dir(outer_fold, seed)
    directory.mkdir(parents=True, exist_ok=True)
    complete_path = directory / "COMPLETE.json"
    config = config_payload(outer_fold, seed, len(train_rows), len(validation_rows))
    digest = config_hash(config)
    write_json(directory / "CONFIG.json", {**config, "config_sha256": digest})
    if complete_path.exists():
        complete = json.loads(complete_path.read_text(encoding="utf-8-sig"))
        if complete.get("config_sha256") != digest:
            raise RuntimeError(f"completed config mismatch: {directory}")
        log(f"reuse complete {directory.name}")
        return complete

    model = make_model(seed, device)
    criterion = SEHCNetLoss(
        lambda_rank=LAMBDA_RANK,
        lambda_bce=LAMBDA_BCE,
        lambda_sparse=LAMBDA_SPARSE,
        rank_temperature=RANK_TEMPERATURE,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=SCHEDULER_PATIENCE
    )
    start_epoch, best_key, best_epoch, bad, steps, train_history, val_history = restore_latest(
        directory, digest, model, optimizer, scheduler
    )
    best_path = directory / "checkpoint_best.pt"
    latest_path = directory / "checkpoint_latest.pt"
    stop_reason = "MAX_EPOCH_REACHED"
    started = time.perf_counter()

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        epoch_seed = seed * 1_000_003 + outer_fold * 10_007 + epoch
        torch.manual_seed(epoch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(epoch_seed)
        model.train()
        generator = torch.Generator().manual_seed(epoch_seed)
        order = torch.randperm(len(train_rows), generator=generator).numpy()
        accumulated = {"train_loss": 0.0, "rank_loss": 0.0, "bce_loss": 0.0, "sparse_loss": 0.0}
        seen_pairs = 0
        maximum_gradient_norm = 0.0

        for start in range(0, len(order), PAIRS_PER_BATCH):
            rows = train_rows[order[start : start + PAIRS_PER_BATCH]]
            flat = torch.as_tensor(bundle.pt[rows].reshape(-1), dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            outputs, labels, roles = forward_pair_batch(model, flat, static92, rain)
            losses = criterion(outputs, labels, pair_role=roles)
            total_loss = losses["loss_total"]
            if not torch.isfinite(total_loss):
                raise RuntimeError("NONFINITE_LOSS")
            total_loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise RuntimeError("NONFINITE_GRADIENT")
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
            if not np.isfinite(gradient_norm):
                raise RuntimeError("NONFINITE_GRAD_NORM")
            optimizer.step()
            steps += 1
            pair_count = len(rows)
            seen_pairs += pair_count
            maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
            accumulated["train_loss"] += float(total_loss.detach().cpu()) * pair_count
            accumulated["rank_loss"] += float(losses["loss_rank"].detach().cpu()) * pair_count
            accumulated["bce_loss"] += float(losses["loss_bce"].detach().cpu()) * pair_count
            accumulated["sparse_loss"] += float(losses["loss_sparse"].detach().cpu()) * pair_count

        train_row = {
            "epoch": epoch,
            **{name: value / seen_pairs for name, value in accumulated.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
            "max_gradient_norm_before_clip": maximum_gradient_norm,
            "optimizer_steps": steps,
        }
        metrics = validation_metrics(
            model, validation_rows, bundle, static92, rain, base, device
        )
        key = validation_key(metrics, -epoch)
        is_best = best_key is None or key > best_key
        if is_best:
            best_key, best_epoch, bad = key, epoch, 0
            atomic_save_checkpoint(
                best_path, model, optimizer, scheduler,
                {
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_key": list(best_key),
                    "bad": bad,
                    "steps": steps,
                    "config_sha256": digest,
                    "validation_metrics": metrics,
                },
                required=True,
            )
        else:
            bad += 1
        scheduler.step(metrics["pair_set_concordance"])
        train_row.update({"is_best": is_best, "best_epoch_so_far": best_epoch, "early_stop_counter": bad})
        validation_row = {
            "epoch": epoch,
            "StrictPair": metrics["StrictPair"],
            "Edge": metrics["Edge"],
            "AUROC": metrics["AUROC"],
            "AUPRC": metrics["AUPRC"],
            "pair_set_concordance": metrics["pair_set_concordance"],
            "edge_wise_ranking_accuracy": metrics["edge_wise_ranking_accuracy"],
            "min_gap_median": metrics["min_gap_median"],
            "min_gap_mean": metrics["min_gap_mean"],
            "is_best": is_best,
        }
        train_history.append(train_row)
        val_history.append(validation_row)
        pd.DataFrame(train_history).to_csv(directory / "train_log.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(val_history).to_csv(directory / "val_metrics.csv", index=False, encoding="utf-8-sig")
        atomic_save_checkpoint(
            latest_path, model, optimizer, scheduler,
            {
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_key": list(best_key),
                "bad": bad,
                "steps": steps,
                "config_sha256": digest,
            },
            required=False,
        )
        log(
            f"FOLD{outer_fold+1} s{seed} e{epoch:03d} "
            f"loss={train_row['train_loss']:.6f} valPair={metrics['StrictPair']:.4f} "
            f"valEdge={metrics['Edge']:.4f} AP={metrics['AUPRC']:.4f} "
            f"best={best_epoch} bad={bad}/{EARLY_STOPPING_PATIENCE}"
        )
        if epoch >= MIN_EPOCHS and bad >= EARLY_STOPPING_PATIENCE:
            stop_reason = "EARLY_STOPPING"
            break

    if not best_path.exists():
        raise RuntimeError(f"best checkpoint missing: {directory}")
    best_checkpoint = torch.load(best_path, map_location="cpu")
    result = {
        **config,
        "config_sha256": digest,
        "best_epoch": int(best_checkpoint["best_epoch"]),
        "stop_epoch": int(train_history[-1]["epoch"]),
        "stop_reason": stop_reason,
        "optimizer_steps": steps,
        "training_seconds_this_process": time.perf_counter() - started,
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256(best_path),
        "status": "PASS_TRAINING_COMPLETED",
    }
    return result


def evaluate_outer_test(
    training_result: dict[str, Any],
    outer_fold: int,
    seed: int,
    test_rows: np.ndarray,
    bundle: Any,
    static92: torch.Tensor,
    rain: torch.Tensor,
    base: Any,
    trigger_schema: pd.DataFrame,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    directory = run_dir(outer_fold, seed)
    model = make_model(seed, device)
    checkpoint = torch.load(Path(training_result["best_checkpoint"]), map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    predictions = predict_pair_rows(
        model, test_rows, bundle, static92, rain, device, collect_triggers=True
    )
    pair_frame, edge_frame, metrics = metric_frames(
        predictions, test_rows, bundle, base, seed, outer_fold
    )
    test_indices = base.srows(bundle, [outer_fold])
    probability = np.clip(predictions["global_risk"][test_indices], 1e-7, 1.0 - 1e-7)
    logit = np.log(probability / (1.0 - probability))
    predicted = (probability >= 0.5).astype(np.int8)
    sample_frame = base.sample_frame(
        bundle, test_indices, EXPERIMENT_ID, seed, outer_fold,
        logit, probability, predicted,
        {"best_epoch": int(training_result["best_epoch"]), "outer_test_forward_count": 1},
    )
    sample_frame["pair_score"] = predictions["pair_score"][test_indices]

    weights = predictions["trigger_weights"][test_indices]
    active_counts = (weights >= ACTIVE_TRIGGER_THRESHOLD).sum(axis=1).astype(np.int16)
    trigger_output = sample_frame[
        ["sample_index", "unit_id", "pair_set_id", "sample_role", "y_true", "outer_fold", "seed"]
    ].copy()
    trigger_output["trigger_weights"] = [
        json.dumps(row.tolist(), separators=(",", ":")) for row in weights
    ]
    trigger_output["active_trigger_count"] = active_counts
    trigger_output["sparsity_ratio"] = 1.0 - active_counts / float(TRIGGER_COUNT)
    trigger_output["active_threshold"] = ACTIVE_TRIGGER_THRESHOLD
    for index, trigger_id in enumerate(trigger_schema["trigger_id"].astype(str)):
        trigger_output[f"weight_{trigger_id}"] = weights[:, index]
    trigger_output.to_csv(directory / "trigger_activation.csv", index=False, encoding="utf-8-sig")

    test_payload = {
        "experiment_id": EXPERIMENT_ID,
        "model": MODEL_NAME,
        "human_fold": outer_fold + 1,
        "outer_fold": outer_fold,
        "seed": seed,
        "best_epoch": int(training_result["best_epoch"]),
        "stop_epoch": int(training_result["stop_epoch"]),
        "outer_test_forward_count": 1,
        "outer_test_used_for_selection": False,
        "refit_performed": False,
        "StrictPair": float(metrics["StrictPair"]),
        "Edge": float(metrics["Edge"]),
        "AUROC": float(metrics["AUROC"]),
        "AUPRC": float(metrics["AUPRC"]),
        "all_metrics": metrics,
        "test_samples": int(len(test_indices)),
        "test_pairs": int(len(test_rows)),
        "mean_active_trigger_count": float(active_counts.mean()),
        "mean_sparsity_ratio": float((1.0 - active_counts / float(TRIGGER_COUNT)).mean()),
    }
    write_json(directory / "test_metrics.json", test_payload)
    sample_frame.to_parquet(directory / "test_sample_predictions.parquet", index=False)
    pair_frame.to_parquet(directory / "test_pair_predictions.parquet", index=False)
    edge_frame.to_parquet(directory / "test_edge_predictions.parquet", index=False)
    write_json(directory / "COMPLETE.json", {**training_result, "test_metrics": test_payload})
    return sample_frame, pair_frame, edge_frame, test_payload


def load_completed_outer_test(
    outer_fold: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    directory = run_dir(outer_fold, seed)
    metrics = json.loads((directory / "test_metrics.json").read_text(encoding="utf-8-sig"))
    return (
        pd.read_parquet(directory / "test_sample_predictions.parquet"),
        pd.read_parquet(directory / "test_pair_predictions.parquet"),
        pd.read_parquet(directory / "test_edge_predictions.parquet"),
        metrics,
    )


def write_formal_config(device: torch.device) -> None:
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "model": MODEL_NAME,
        "data_protocol": "SAME_AS_BASE06",
        "split": "SAME_SPATIAL_5FOLD",
        "folds": [1, 2, 3, 4, 5],
        "seeds": SEEDS,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "pairs_per_batch": PAIRS_PER_BATCH,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
        },
        "scheduler": {"name": "ReduceLROnPlateau", "mode": "max", "patience": SCHEDULER_PATIENCE},
        "loss": {
            "lambda_rank": LAMBDA_RANK,
            "lambda_bce": LAMBDA_BCE,
            "lambda_sparse": LAMBDA_SPARSE,
        },
        "device": str(device),
        "source_authority": {
            "BASE06_launcher": {"path": str(BASE06_LAUNCHER), "sha256": sha256(BASE06_LAUNCHER)},
            "BASE06_runner": {"path": str(BASE06_RUNNER), "sha256": sha256(BASE06_RUNNER)},
            "model": {"path": str(MODEL_FILE), "sha256": sha256(MODEL_FILE)},
            "loss": {"path": str(LOSS_FILE), "sha256": sha256(LOSS_FILE)},
            "trigger_schema": {"path": str(TRIGGER_SCHEMA), "sha256": sha256(TRIGGER_SCHEMA)},
        },
    }
    write_json(OUTPUT / "FORMAL_CONFIG.json", payload)


def run_formal(device: torch.device, torch_threads: int) -> int:
    fairness_banner()
    trigger_schema = validate_required_assets()
    runner = import_base06_runner()
    base = runner.base
    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_formal_config(device)
    bundle = base.preflight()
    sample_outputs: list[pd.DataFrame] = []
    pair_outputs: list[pd.DataFrame] = []
    edge_outputs: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for outer_fold in FOLDS:
        validation_fold = (outer_fold + 1) % 5
        train_folds = sorted(set(FOLDS) - {outer_fold, validation_fold})
        train_indices = base.srows(bundle, train_folds)
        validation_indices = base.srows(bundle, [validation_fold])
        test_indices = base.srows(bundle, [outer_fold])
        train_pairs = base.prows(bundle, train_folds)
        validation_pairs = base.prows(bundle, [validation_fold])
        test_pairs = base.prows(bundle, [outer_fold])
        log(
            f"FOLD{outer_fold+1}: train={train_folds}, validation=FOLD{validation_fold+1}, "
            f"outer-test=FOLD{outer_fold+1}; fitting BASE06 fold-only preprocessor"
        )
        preprocessor = runner.fit_preprocessor(bundle, train_indices)
        preprocessor_path = OUTPUT / "preprocessors" / f"fold{outer_fold+1}.json"
        write_json(preprocessor_path, preprocessor)
        xs91, category_index, rain_np = runner.transform_all(bundle, preprocessor)
        static92_np = build_static_92(xs91, category_index)
        static92 = torch.as_tensor(static92_np, dtype=torch.float32, device=device)
        rain = torch.as_tensor(rain_np, dtype=torch.float32, device=device)
        if len(validation_indices) != len(bundle.pt[validation_pairs].reshape(-1)):
            raise RuntimeError("validation sample/pair coverage mismatch")
        if len(test_indices) != len(bundle.pt[test_pairs].reshape(-1)):
            raise RuntimeError("test sample/pair coverage mismatch")

        for seed in SEEDS:
            directory = run_dir(outer_fold, seed)
            completed_files = [
                directory / "COMPLETE.json",
                directory / "checkpoint_best.pt",
                directory / "train_log.csv",
                directory / "val_metrics.csv",
                directory / "test_metrics.json",
                directory / "trigger_activation.csv",
                directory / "test_sample_predictions.parquet",
                directory / "test_pair_predictions.parquet",
                directory / "test_edge_predictions.parquet",
            ]
            if all(path.exists() for path in completed_files):
                sample_frame, pair_frame, edge_frame, test_metrics = load_completed_outer_test(
                    outer_fold, seed
                )
                log(f"reuse formal output {directory.name}")
            else:
                training_result = train_fold_seed(
                    outer_fold, seed, train_pairs, validation_pairs,
                    bundle, static92, rain, base, device,
                )
                sample_frame, pair_frame, edge_frame, test_metrics = evaluate_outer_test(
                    training_result, outer_fold, seed, test_pairs,
                    bundle, static92, rain, base, trigger_schema, device,
                )
            sample_outputs.append(sample_frame)
            pair_outputs.append(pair_frame)
            edge_outputs.append(edge_frame)
            summary_rows.append({
                "experiment_id": EXPERIMENT_ID,
                "fold": outer_fold + 1,
                "outer_fold": outer_fold,
                "validation_fold": validation_fold,
                "training_folds": "|".join(map(str, train_folds)),
                "seed": seed,
                "best_epoch": test_metrics["best_epoch"],
                "stop_epoch": test_metrics["stop_epoch"],
                "StrictPair": test_metrics["StrictPair"],
                "Edge": test_metrics["Edge"],
                "AUROC": test_metrics["AUROC"],
                "AUPRC": test_metrics["AUPRC"],
                "mean_active_trigger_count": test_metrics["mean_active_trigger_count"],
                "mean_sparsity_ratio": test_metrics["mean_sparsity_ratio"],
            })

        del static92, rain, static92_np, xs91, category_index, rain_np
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    sample_oof = pd.concat(sample_outputs, ignore_index=True)
    pair_oof = pd.concat(pair_outputs, ignore_index=True)
    edge_oof = pd.concat(edge_outputs, ignore_index=True)
    if sample_oof.duplicated(["seed", "sample_index"]).any():
        raise RuntimeError("duplicate sample OOF predictions")
    if pair_oof.duplicated(["seed", "pair_set_id"]).any():
        raise RuntimeError("duplicate pair OOF predictions")
    if edge_oof.duplicated(["seed", "pair_set_id", "control_rank"]).any():
        raise RuntimeError("duplicate edge OOF predictions")
    for seed in SEEDS:
        if len(sample_oof[sample_oof.seed == seed]) != base.N:
            raise RuntimeError(f"sample OOF coverage mismatch for seed {seed}")
        if len(pair_oof[pair_oof.seed == seed]) != base.NP:
            raise RuntimeError(f"pair OOF coverage mismatch for seed {seed}")
        if len(edge_oof[edge_oof.seed == seed]) != base.NC:
            raise RuntimeError(f"edge OOF coverage mismatch for seed {seed}")

    sample_oof.to_parquet(OUTPUT / "SAMPLE_OOF.parquet", index=False)
    pair_oof.to_parquet(OUTPUT / "PAIR_OOF.parquet", index=False)
    edge_oof.to_parquet(OUTPUT / "EDGE_OOF.parquet", index=False)
    summary = pd.DataFrame(summary_rows).sort_values(["fold", "seed"], kind="stable")
    summary.to_csv(
        OUTPUT / "SEHC_NET_M_V1_FORMAL_SUMMARY.csv", index=False, encoding="utf-8-sig"
    )
    write_json(OUTPUT / "FORMAL_COMPLETE.json", {
        "status": "PASS_SEHC_NET_M_V1_FORMAL_COMPLETED",
        "fold_seed_runs": len(summary),
        "sample_oof_rows": len(sample_oof),
        "pair_oof_rows": len(pair_oof),
        "edge_oof_rows": len(edge_oof),
        "summary": str(OUTPUT / "SEHC_NET_M_V1_FORMAL_SUMMARY.csv"),
    })
    print("PASS_SEHC_NET_M_V1_FORMAL_COMPLETED", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.torch_threads < 1:
        raise ValueError("torch-threads must be >= 1")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_REQUESTED_BUT_NOT_AVAILABLE")
        device = torch.device("cuda")
    elif args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    return run_formal(device, args.torch_threads)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "FAIL_SEHC_NET_M_V1_FORMAL",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(OUTPUT / "FORMAL_FAILURE.json", failure)
        print(failure["traceback"], flush=True)
        raise
