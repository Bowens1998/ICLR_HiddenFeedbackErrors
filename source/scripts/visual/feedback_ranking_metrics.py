"""Outcome metrics for a fixed action-candidate set, with exact-tie semantics.

This module has no model or simulator dependencies. ``evaluate_case`` accepts
predicted costs in branch order free/act/don/reset and realized candidate costs.
Its primary selection estimand is the average realized cost over *all* exact
predicted-cost minimizers (uniform tie resolution). The deterministic lowest
index choice is also retained. Rank correlations are null when undefined, never
NaN. ``aggregate_by_objective`` first averages fixed model/context rows within
each goal, then averages goals; it does not construct confidence intervals or
label a development result as confirmation.
"""
from collections import defaultdict
import math

import numpy as np


BRANCHES = ("free", "act", "don", "reset")
COST_SPACES = ("pose", "latent")
CONTRASTS = (("act", "free"), ("act", "don"), ("don", "free"),
             ("reset", "free"), ("act", "reset"))


def _vector(values, name):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or len(result) < 2 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector with at least two candidates")
    return result


def physical_goal_cost(states, goal_state):
    """Existing PushT cost: block xy squared error + 900 * wrapped angle^2.

    State arrays may retain extra simulator fields after the first five.
    """
    states = np.asarray(states, dtype=np.float64)
    goal = np.asarray(goal_state, dtype=np.float64)
    if states.ndim < 1 or states.shape[-1] < 5 or goal.ndim != 1 or goal.size < 5:
        raise ValueError("Expected simulator states with at least five coordinates")
    if not np.isfinite(states).all() or not np.isfinite(goal).all():
        raise ValueError("Nonfinite physical state")
    angle = np.arctan2(np.sin(states[..., 4] - goal[4]),
                       np.cos(states[..., 4] - goal[4]))
    return np.square(states[..., 2:4] - goal[2:4]).sum(-1) + 900 * angle**2


def _average_ranks(values):
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        result[order[start:stop]] = (start + stop - 1) / 2 + 1
        start = stop
    return result


def ranking_metrics(predicted_cost, realized_cost):
    """Return selection, regret, exact-tie counts, Kendall tau-b and Spearman."""
    predicted = _vector(predicted_cost, "predicted_cost")
    realized = _vector(realized_cost, "realized_cost")
    if predicted.shape != realized.shape:
        raise ValueError("Predicted and realized candidate vectors must align")
    best = np.flatnonzero(predicted == predicted.min())
    pool_best = float(realized.min())
    selected_mean = float(realized[best].mean())
    chosen = int(best[0])
    ii, jj = np.triu_indices(len(predicted), k=1)
    px = np.sign(predicted[ii] - predicted[jj])
    py = np.sign(realized[ii] - realized[jj])
    concordant = int(np.count_nonzero(px * py > 0))
    discordant = int(np.count_nonzero(px * py < 0))
    tied_predicted = int(np.count_nonzero(px == 0))
    tied_realized = int(np.count_nonzero(py == 0))
    tied_both = int(np.count_nonzero((px == 0) & (py == 0)))
    total = len(ii)
    denominator = math.sqrt((total - tied_predicted) * (total - tied_realized))
    tau = (concordant - discordant) / denominator if denominator else None
    rx, ry = _average_ranks(predicted), _average_ranks(realized)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    rank_norm = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    rho = float(np.dot(rx, ry) / rank_norm) if rank_norm else None
    return {
        "candidate_count": len(predicted), "argmin_indices": best.tolist(),
        "argmin_count": len(best), "selected_index_lowest": chosen,
        "predicted_minimum": float(predicted.min()), "pool_best_cost": pool_best,
        "selected_cost": selected_mean, "regret": selected_mean - pool_best,
        "selected_cost_lowest": float(realized[chosen]),
        "regret_lowest": float(realized[chosen]) - pool_best,
        "kendall_tau_b": tau, "spearman": rho,
        "unique_predicted_costs": len(np.unique(predicted)),
        "unique_realized_costs": len(np.unique(realized)),
        "pair_counts": {"total": total, "concordant": concordant,
                        "discordant": discordant, "tied_predicted": tied_predicted,
                        "tied_realized": tied_realized, "tied_both": tied_both},
    }


def evaluate_case(predicted_costs, realized_cost):
    """Costs are (4,N), ordered by BRANCHES; return branches and paired contrasts."""
    values = np.asarray(predicted_costs, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(BRANCHES):
        raise ValueError("Expected four branch-by-candidate cost rows")
    branches = {b: ranking_metrics(values[i], realized_cost) for i, b in enumerate(BRANCHES)}
    contrasts = {}
    for lhs, rhs in CONTRASTS:
        left, right = branches[lhs], branches[rhs]
        a, b = set(left["argmin_indices"]), set(right["argmin_indices"])
        contrasts[f"{lhs}_minus_{rhs}"] = {
            "selected_cost": left["selected_cost"] - right["selected_cost"],
            "regret": left["regret"] - right["regret"],
            "selected_cost_lowest": left["selected_cost_lowest"] - right["selected_cost_lowest"],
            "selection_set_changed": a != b,
            "selection_sets_disjoint": a.isdisjoint(b),
            "lowest_index_changed": left["selected_index_lowest"] != right["selected_index_lowest"],
            "kendall_tau_b": (left["kendall_tau_b"] - right["kendall_tau_b"]
                              if left["kendall_tau_b"] is not None and right["kendall_tau_b"] is not None else None),
            "spearman": (left["spearman"] - right["spearman"]
                         if left["spearman"] is not None and right["spearman"] is not None else None),
        }
    return {"branches": branches, "contrasts": contrasts}


def _mean_valid(values):
    valid = [float(v) for v in values if v is not None]
    return {"mean": float(np.mean(valid)) if valid else None,
            "valid_count": len(valid), "total_count": len(values)}


def _mean_case_metrics(items, already_aggregated=False):
    result = {"branches": {}, "contrasts": {}}
    for branch in BRANCHES:
        fields = ("selected_cost", "regret", "selected_cost_lowest", "regret_lowest",
                  "kendall_tau_b", "spearman", "argmin_count")
        result["branches"][branch] = {}
        for field in fields:
            values = [x["branches"][branch][field] for x in items]
            if already_aggregated:
                values = [x["mean"] for x in values]
            result["branches"][branch][field] = _mean_valid(values)
    for lhs, rhs in CONTRASTS:
        key = f"{lhs}_minus_{rhs}"
        result["contrasts"][key] = {}
        for field in ("selected_cost", "regret", "selected_cost_lowest", "selection_set_changed",
                      "selection_sets_disjoint", "lowest_index_changed", "kendall_tau_b", "spearman"):
            values = [x["contrasts"][key][field] for x in items]
            if already_aggregated:
                values = [x["mean"] for x in values]
            result["contrasts"][key][field] = _mean_valid(values)
    return result


def aggregate_by_objective(rows):
    """Aggregate rows with objective, goal_index and metrics[pose|latent].

    One row is one model/context combination. All fixed rows receive equal
    weight within each goal, and each goal receives equal weight overall.
    Missing rank correlations retain explicit valid/total counts. No imputation.
    """
    if not rows:
        raise ValueError("Cannot aggregate an empty experiment")
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["objective"]][int(row["goal_index"])].append(row)
    output = {}
    for objective, goals in sorted(grouped.items()):
        per_goal = []
        for goal, goal_rows in sorted(goals.items()):
            per_goal.append({"goal_index": goal, "model_context_rows": len(goal_rows),
                             "metrics": {space: _mean_case_metrics([r["metrics"][space] for r in goal_rows])
                                         for space in COST_SPACES}})
        output[objective] = {
            "goal_count": len(per_goal), "model_context_rows": sum(len(r) for r in goals.values()),
            "per_goal": per_goal,
            "equal_goal_means": {space: _mean_case_metrics([r["metrics"][space] for r in per_goal], True)
                                 for space in COST_SPACES},
        }
    return output
