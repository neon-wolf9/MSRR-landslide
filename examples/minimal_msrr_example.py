#!/usr/bin/env python
"""Run the redistribution-safe demonstration of the canonical MSRR transform."""
from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    namespace = runpy.run_path(str(ROOT / "scripts" / "demo_msrr.py"))
    return int(namespace["main"]())


if __name__ == "__main__":
    raise SystemExit(main())
