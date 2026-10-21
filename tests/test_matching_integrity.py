from pathlib import Path
import importlib.util

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


def test_canonical_common_support_interface_on_synthetic_edges():
    """Exercise the published solver with invented IDs, not event observations."""
    path = ROOT / "src" / "benchmark" / "hard_control_01e_common_support_protocol.py"
    spec = importlib.util.spec_from_file_location("synthetic_common_support", path)
    assert spec is not None and spec.loader is not None
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    protocol.N_POS, protocol.N_CAND = 2, 3

    positives = pd.DataFrame({"unit_id": ["SYN_P0", "SYN_P1"]})
    candidates = pd.DataFrame({"unit_id": ["SYN_C0", "SYN_C1", "SYN_C2"]})
    # P1 has only one legal edge, so the all-or-none rule must exclude it.
    legal = pd.DataFrame({
        "positive_index": [0, 0, 1],
        "candidate_index": [0, 1, 1],
        "composite_distance": [1.0, 2.0, 1.0],
        "geology_parent": ["SYN_G"] * 3,
        "matching_missingness_pattern_id": ["SYN_M"] * 3,
        "soil_completeness_block": ["SYN_S"] * 3,
    })
    selected, feasibility = protocol.exact_all_or_none(legal, positives)
    assert selected.tolist() == [0]
    assert feasibility.loc[0, "all_or_none_upper_bound"] == 1

    assignments, certificate = protocol.mincost_certificate(
        legal, selected, positives, candidates
    )
    assert len(assignments) == 2
    assert set(assignments["candidate_unit_id"]) == {"SYN_C0", "SYN_C1"}
    assert assignments["candidate_unit_id"].is_unique
    assert set(zip(assignments["positive_index"], assignments["candidate_index"])) <= set(
        zip(legal["positive_index"], legal["candidate_index"])
    )
    assert assignments["TRAINING_DATA"].eq("NO").all()
    assert certificate.loc[0, "solver_status"] == "OPTIMAL"
