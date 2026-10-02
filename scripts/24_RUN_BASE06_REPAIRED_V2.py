                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      #0 -*- coding: utf-8 -*-
r"""
24_RUN_BASE06_REPAIRED_V2.py
============================

Formal launcher for BASE-06 Pairwise RankNet on repaired-V2.

It does NOT:
- rerun BASE-05;
- delete checkpoints;
- use --force;
- use --restart-incomplete;
- modify repaired-V2 data.

It DOES:
- confirm BASE-05 has completed;
- confirm 23D exists;
- call 23D with BASE-06 only;
- preserve resume/reuse behavior already implemented in 23D.

Run:
python ^
<PROJECT_ROOT>\scripts\24_RUN_BASE06_REPAIRED_V2.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]

PYTHON = Path(sys.executable)

RUNNER_23D = (
    PROJECT
    / "scripts"
    / "23D_RUN_NEURAL_BASELINES_REPAIRED_V2.py"
)

EXPERIMENT_ROOT = (
    PROJECT
    / "experiments"
    / "FORMAL_BASELINES_REPAIRED_V2_V1"
)

BASE05_FLAG = (
    EXPERIMENT_ROOT
    / "BASE05"
    / "COMPLETED.flag"
)

BASE06_DIR = (
    EXPERIMENT_ROOT
    / "BASE06"
)


def fail(message: str, code: int = 1) -> int:
    print("\n" + "=" * 100)
    print("BASE-06 LAUNCH BLOCKED")
    print("=" * 100)
    print(message)
    print("=" * 100)
    return code


def main() -> int:
    print("=" * 100)
    print("BASE-06 PAIRWISE RANKNET — REPAIRED-V2 FORMAL RUN")
    print("=" * 100)

    print(f"Project : {PROJECT}")
    print(f"Python  : {PYTHON}")
    print(f"23D     : {RUNNER_23D}")
    print()

    if not PYTHON.exists():
        return fail(
            f"Python executable not found:\n{PYTHON}",
            2,
        )

    if not RUNNER_23D.exists():
        return fail(
            f"23D runner not found:\n{RUNNER_23D}",
            3,
        )

    if not BASE05_FLAG.exists():
        return fail(
            "BASE-05 formal completion flag is missing.\n"
            "BASE-06 must not start before BASE-05 completes.\n\n"
            f"Expected:\n{BASE05_FLAG}",
            4,
        )

    flag_text = BASE05_FLAG.read_text(
        encoding="utf-8-sig",
        errors="replace",
    ).strip()

    if "PASS" not in flag_text.upper():
        return fail(
            "BASE-05 COMPLETED.flag exists but does not contain PASS.\n\n"
            f"File:\n{BASE05_FLAG}\n\n"
            f"Content:\n{flag_text}",
            5,
        )

    print("[PASS] BASE-05 completion confirmed.")
    print()
    print("Formal BASE-06 protocol:")
    print("  model      : BASE-06 Pairwise RankNet")
    print("  data       : repaired-V2 frozen dataset")
    print("  outer CV   : 5 spatial folds")
    print("  seeds      : 7, 11, 21")
    print("  candidates : 16 frozen candidates")
    print("  bootstrap  : 2000")
    print("  device     : CPU")
    print("  torch CPU threads : 1")
    print("  resume/reuse      : ENABLED")
    print("  --force           : NOT USED")
    print("  --restart-incomplete : NOT USED")
    print()

    # Keep the same conservative runtime environment used for BASE-05.
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"

    command = [
        str(PYTHON),
        str(RUNNER_23D),
        "--models",
        "BASE-06",
        "--bootstrap",
        "2000",
        "--device",
        "cpu",
        "--torch-threads",
        "1",
    ]

    print("Command:")
    print(" ".join(f'"{x}"' if " " in x else x for x in command))
    print()
    print("=" * 100)
    print("STARTING BASE-06")
    print("=" * 100)
    print()

    # No capture_output: epoch progress is shown live in PyCharm/terminal.
    completed = subprocess.run(
        command,
        cwd=str(PROJECT),
        env=env,
    )

    rc = int(completed.returncode)

    print()
    print("=" * 100)
    print("BASE-06 PROCESS FINISHED")
    print("=" * 100)
    print(f"Exit code: {rc}")
    print()

    if rc == 0:
        print("[PASS] BASE-06 formal run completed.")
        print()
        print("Main outputs:")
        print(BASE06_DIR / "MODEL_SUMMARY.json")
        print(BASE06_DIR / "FOLD_SEED_METRICS.csv")
        print(BASE06_DIR / "SELECTED_CONFIGS.csv")
        print(EXPERIMENT_ROOT / "FORMAL_BASELINE_COMPARISON.csv")
        print(EXPERIMENT_ROOT / "BASE06_VS_BASE05_PAIRED_EFFECT.csv")
        print(EXPERIMENT_ROOT / "BASE06_VS_BASE05_TRANSITIONS.csv")
    else:
        print("[FAILED / INTERRUPTED]")
        print("Do NOT delete candidate_runs or checkpoints.")
        print("Simply run this same Python launcher again.")
        print("23D will reuse completed candidates and resume valid checkpoints.")

    print("=" * 100)

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
