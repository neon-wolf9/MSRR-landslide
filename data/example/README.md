# Synthetic MSRR example

`synthetic_msrr_demo.npz` contains eight synthetic matched sets with shape `[8, 3, 792]` and deterministic synthetic candidate scores. It contains no Hiroshima, Kyushu, EO/GIS, inventory, or personal data.

Run `python scripts/demo_msrr.py` from the repository root. The demo loads the canonical experiment-120B implementation, constructs the 2,446-dimensional representation, checks every output block numerically, and verifies StrictPair and Edge.
