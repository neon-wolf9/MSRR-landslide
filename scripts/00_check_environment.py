#!/usr/bin/env python
"""Report the runtime used for a package verification or scientific rerun."""
from __future__ import annotations

import platform
import subprocess
import sys


REQUIRED = {
    "numpy": "1.24.0",
    "pandas": "2.0.0",
    "scipy": "1.15.3",
    "sklearn": "1.5.2",
    "torch": "2.1.0",
    "xgboost": "3.2.0",
    "pyarrow": "16.1.0",
    "geopandas": "1.1.3",
    "matplotlib": "3.10.8",
    "yaml": "6.0",
    "shapely": "2.1.2",
    "pyogrio": "0.12.1",
    "rasterio": "1.4.4",
    "fiona": "1.10.1",
    "affine": "2.4.0",
    "pyproj": "3.7.1",
    "imblearn": "0.12.4",
    "joblib": "1.6.0",
    "PIL": "12.1.1",
    "pytest": "7.4.0",
}


def main() -> int:
    print(f"python={platform.python_version()} executable={sys.executable}")
    missing: list[str] = []
    for name, reference in REQUIRED.items():
        # Import each compiled dependency in an isolated process. On Windows,
        # Fiona, Rasterio, Pyogrio, and GeoPandas may load different bundled
        # GDAL DLLs; importing every backend into one diagnostic process can
        # terminate that process even when each repository workflow imports a
        # valid backend combination. Isolation makes this checker diagnostic
        # without changing any scientific execution path.
        code = (
            "import importlib; "
            f"m=importlib.import_module({name!r}); "
            "print(getattr(m, '__version__', 'unknown'))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if completed.returncode == 0:
            observed = completed.stdout.strip().splitlines()[-1]
            print(f"{name}={observed} reference={reference}")
        else:
            missing.append(name)
            print(f"{name}=IMPORT_FAILED (exit={completed.returncode})")
    if missing:
        print("FAIL missing=" + ",".join(missing))
        return 1
    print("PASS environment imports complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
