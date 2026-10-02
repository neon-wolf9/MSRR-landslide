# Minimal example

Run from the repository root:

```bash
python examples/minimal_msrr_example.py
```

The entry point delegates to `scripts/demo_msrr.py`, which imports the canonical manuscript implementation from experiment 120B. The input contains eight synthetic three-member matched sets. It verifies the shared mean, signed deviation, absolute deviation, seven rainfall-residual summaries, output dimension, StrictPair, and Edge calculations.

This synthetic example demonstrates the MSRR transformation only and is not a reproduction of the landslide experiments.
