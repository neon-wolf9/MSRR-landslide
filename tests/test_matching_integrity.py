from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching


ROOT = Path(__file__).resolve().parents[1]


def test_replayed_hiroshima_sets_are_complete_and_unique():
    path = ROOT / "reference_results" / "hiroshima" / "HIROSHIMA_REPLAY_MATCHED_SETS.csv"
    frame = pd.read_csv(path)
    assert len(frame) == 5056
    assert frame["matched_set_id"].nunique() == 5056
    assert (frame["control_low_id"].astype(str) != frame["control_high_id"].astype(str)).all()
    controls = pd.concat([frame["control_low_id"], frame["control_high_id"]], ignore_index=True).astype(str)
    assert len(controls) == 2 * len(frame)
    assert controls.nunique() == len(controls)  # capacity = 1 / no reuse


def test_replayed_kyushu_sets_are_complete_and_unique():
    path = ROOT / "reference_results" / "hiroshima" / "KYUSHU_REPLAY_MATCHED_SETS.csv"
    frame = pd.read_csv(path)
    assert len(frame) == 1692
    assert frame["matched_set_id"].nunique() == 1692
    assert (frame["control_low_id"].astype(str) != frame["control_high_id"].astype(str)).all()
    controls = pd.concat([frame["control_low_id"], frame["control_high_id"]], ignore_index=True).astype(str)
    assert len(controls) == 2 * len(frame)
    assert controls.nunique() == len(controls)


def test_legal_edge_graph_enforces_two_controls_and_capacity_one():
    # Two demand rows per positive; absent sparse entries are illegal edges.
    # Legal edges: P0->{C0,C1,C2}; P1->{C2,C3,C4}. P0->C4 is deliberately illegal.
    rows = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    cols = np.array([0, 1, 2, 0, 1, 2, 2, 3, 4, 2, 3, 4])
    costs = np.array([1, 2, 9, 1, 2, 9, 9, 1, 2, 9, 1, 2], dtype=float)
    graph = csr_matrix((costs, (rows, cols)), shape=(4, 5))
    demand_rows, selected_controls = min_weight_full_bipartite_matching(graph)
    assert len(demand_rows) == 4
    assert len(np.unique(selected_controls)) == 4  # capacity = 1
    assert set(selected_controls[:2]).issubset({0, 1, 2})
    assert set(selected_controls[2:]).issubset({2, 3, 4})
    assert selected_controls[0] != 4 and selected_controls[1] != 4
