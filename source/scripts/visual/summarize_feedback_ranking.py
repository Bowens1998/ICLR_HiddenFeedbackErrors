"""Summarize accepted shared-prefix ranking development runs across groups.

Example: python scripts/visual/summarize_feedback_ranking.py --runs RUN_ROOT
--groups 0 1 --output summary.json. Intervals are omitted by default. Supplying
--draws 20000 adds exploratory paired-goal percentile intervals for actual-minus-
free and actual-minus-donor; this is not an independent confirmation protocol.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from feedback_ranking_metrics import BRANCHES, COST_SPACES, aggregate_by_objective, physical_goal_cost


OBJECTIVES = {2: "unit_latent", 3: "unit_decoded_teacher", 4: "unit_physical_labels"}
PRIMARY_CONTRASTS = ("act_minus_free", "act_minus_don")
BOOTSTRAP_SEED = 1379001


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _bound_file(root, relative, expected_hash):
    path = (root / relative).resolve()
    _require(path.is_relative_to(root.resolve()), f"Path escapes artifact directory: {relative}")
    _require(path.is_file() and sha(path) == expected_hash, f"Artifact hash mismatch: {relative}")
    return path


def load_accepted_groups(runs, groups):
    """Validate complete accepted artifacts and a common case/goal/seed roster."""
    runs = Path(runs)
    _require(bool(groups) and len(groups) == len(set(groups)), "Groups must be nonempty and unique")
    _require(all(g in range(6) for g in groups), "Group indices must be in 0..5")
    rows, provenance, reference_roster = [], [], None
    reference_protocol = None
    for group in sorted(groups):
        bank, scores = runs / f"bank_{group}", runs / f"scores_{group}"
        acceptance_path = runs / f"acceptance_{group}.json"
        acceptance = _read(acceptance_path)
        manifest, report = _read(bank / "manifest.json"), _read(scores / "report.json")
        _require(acceptance["status"] == "PASS_DEVELOPMENT_FEEDBACK_RANKING_NUMPY_ACCEPTANCE"
                 and acceptance["phase"] == "development", "Group has no accepted development result")
        _require(acceptance["group"] == manifest["group"] == report["group"] == group, "Group binding mismatch")
        _require(acceptance["bank_manifest_sha256"] == report["bank_manifest_sha256"] == sha(bank / "manifest.json"),
                 "Bank manifest hash mismatch")
        _require(acceptance["bank_acceptance_sha256"] == sha(bank / "acceptance.json"), "Bank acceptance hash mismatch")
        _require(acceptance["score_report_sha256"] == sha(scores / "report.json"), "Score report hash mismatch")
        _require(acceptance["acceptor_sha256"] == sha(Path(__file__).with_name("accept_feedback_ranking.py")),
                 "Acceptance code version mismatch")
        _require(acceptance["metrics_sha256"] == sha(Path(__file__).with_name("feedback_ranking_metrics.py")),
                 "Metric code version mismatch")
        _require(acceptance["branches"] == list(BRANCHES) and acceptance["horizons"] == [10, 15, 20, 25],
                 "Accepted branch/horizon schema mismatch")
        _require(acceptance["candidate_count"] == manifest["candidates"] == 32, "Candidate count mismatch")
        protocol = acceptance["metric_protocol"]
        if reference_protocol is None:
            reference_protocol = protocol
        _require(protocol == reference_protocol, "Metric protocols differ across groups")
        case_items = {int(item["index"]): item for item in manifest["cases"]}
        _require(len(case_items) == len(manifest["cases"]) == acceptance["case_count"], "Case count or duplicate mismatch")
        binding_items = {int(item["index"]): item for item in acceptance["bank_bindings"]}
        _require(set(binding_items) == set(case_items), "Accepted bank bindings are incomplete")
        roster = sorted((index, int(item.get("goal_index", index)), int(item["seed"]))
                        for index, item in case_items.items())
        if reference_roster is None:
            reference_roster = roster
        _require(roster == reference_roster, "Case/goal/seed roster differs across groups")
        for index, item in case_items.items():
            for kind in ("input", "outcome"):
                _require(binding_items[index][f"{kind}_sha256"] == item[f"{kind}_sha256"], "Accepted file binding mismatch")
                _bound_file(bank, item[f"{kind}_file"], item[f"{kind}_sha256"])

        expected_models = [8 * group + slot for slot in OBJECTIVES]
        models = {int(entry["model_index"]): entry for entry in report["models"]}
        _require(report["expected_model_indices"] == expected_models and sorted(models) == expected_models
                 and len(report["models"]) == acceptance["model_count"] == 3, "Incomplete model roster")
        expected_pairs, score_hashes = set(), {}
        for mi, entry in models.items():
            _require(entry["objective"] == OBJECTIVES[mi % 8], "Objective binding mismatch")
            _bound_file(scores, entry["head_file"], entry["head_sha256"])
            found = [int(case["index"]) for case in entry["cases"]]
            _require(len(found) == len(case_items) and sorted(found) == sorted(case_items), "Incomplete scored cases")
            for case in entry["cases"]:
                index = int(case["index"])
                _require(case["input_sha256"] == case_items[index]["input_sha256"], "Scored input binding mismatch")
                _bound_file(scores, case["file"], case["sha256"])
                expected_pairs.add((mi, index))
                score_hashes[(mi, index)] = case["sha256"]
        group_rows = acceptance["rows"]
        observed_pairs = [(int(row["model_index"]), int(row["case_index"])) for row in group_rows]
        _require(len(observed_pairs) == len(expected_pairs) and set(observed_pairs) == expected_pairs,
                 "Accepted metric rows omit or duplicate a model/case")
        for row in group_rows:
            mi, index = int(row["model_index"]), int(row["case_index"])
            item = case_items[index]
            _require(row["group"] == group and row["objective"] == OBJECTIVES[mi % 8], "Metric row model mismatch")
            _require(row["seed"] == item["seed"] and row["goal_index"] == item.get("goal_index", index),
                     "Metric row goal/seed mismatch")
            _require(row["score_sha256"] == score_hashes[(mi, index)], "Metric row score binding mismatch")
        _require(aggregate_by_objective(group_rows) == acceptance["objectives"], "Accepted row/summary inconsistency")
        rows.extend(group_rows)
        provenance.append({"group": group, "acceptance_file": str(acceptance_path.resolve()),
                           "acceptance_sha256": sha(acceptance_path),
                           "bank_manifest_sha256": acceptance["bank_manifest_sha256"],
                           "score_report_sha256": acceptance["score_report_sha256"],
                           "model_indices": expected_models, "case_count": len(case_items)})
    return rows, provenance, reference_roster


def _compact_objective(aggregated, resamples=None):
    compact = {"goal_count": aggregated["goal_count"], "model_context_rows": aggregated["model_context_rows"],
               "score_spaces": {}}
    for space in COST_SPACES:
        means = aggregated["equal_goal_means"][space]
        free_cost = means["branches"]["free"]["selected_cost"]["mean"]
        contrasts = {}
        for name, value in means["contrasts"].items():
            delta = value["selected_cost"]["mean"]
            contrasts[name] = dict(value,
                                  relative_gain_percent_of_free=(-100 * delta / free_cost if free_cost > 0 else None))
            if resamples is not None and name in PRIMARY_CONTRASTS:
                by_goal = np.asarray([goal["metrics"][space]["contrasts"][name]["selected_cost"]["mean"]
                                      for goal in aggregated["per_goal"]], dtype=np.float64)
                draws = by_goal[resamples].mean(axis=1)
                contrasts[name]["exploratory_selected_cost_ci95"] = np.quantile(draws, [.025, .975]).tolist()
        compact["score_spaces"][space] = {
            "role": "primary" if space == "pose" else "secondary",
            "free_selected_cost": free_cost,
            "branches": means["branches"], "contrasts": contrasts,
        }
    return compact


def summarize_rows(rows, draws=0, seed=BOOTSTRAP_SEED):
    """Average all fixed groups within each goal before optional goal resampling."""
    _require(isinstance(draws, int) and draws >= 0, "Bootstrap draws must be a nonnegative integer")
    aggregate = aggregate_by_objective(rows)
    _require(set(aggregate) == set(OBJECTIVES.values()), "All three objectives must be represented")
    rosters = [[goal["goal_index"] for goal in aggregate[name]["per_goal"]] for name in sorted(aggregate)]
    _require(all(roster == rosters[0] for roster in rosters), "Objective goal rosters differ")
    goals = rosters[0]
    resamples = np.random.default_rng(seed).integers(0, len(goals), size=(draws, len(goals))) if draws else None
    per_model = []
    for model in sorted({int(row["model_index"]) for row in rows}):
        selected = [row for row in rows if row["model_index"] == model]
        objective = selected[0]["objective"]
        _require(all(row["objective"] == objective for row in selected), "Model has multiple objective labels")
        model_aggregate = aggregate_by_objective(selected)[objective]
        # Individual model points are descriptive; pooled bootstrap conditions
        # on the full fixed roster and never resamples models as independent data.
        per_model.append({"model_index": model, "group": selected[0]["group"], "objective": objective,
                          **_compact_objective(model_aggregate)})
    return {"goal_indices": goals, "goal_count": len(goals), "model_count": len(per_model),
            "objectives": {name: _compact_objective(value, resamples) for name, value in aggregate.items()},
            "per_model": per_model, "goal_aggregation": aggregate,
            "bootstrap": {"draws": draws, "seed": seed if draws else None,
                          "interval_level": .95 if draws else None,
                          "unit": "goal; fixed model/group rows averaged before resampling",
                          "shared_draws": bool(draws), "contrasts": list(PRIMARY_CONTRASTS) if draws else [],
                          "method": "paired-goal percentile" if draws else None,
                          "interpretation": "Exploratory development intervals; no familywise or confirmatory claim" if draws else "Descriptive point estimates only"},
            "relative_gain_definition": "-100 * mean paired selected-cost difference / pooled free selected cost; positive means improvement. Both contrasts use the same free denominator; zero denominator yields null."}


def summarize_pools(runs, groups):
    """Physical pool baselines and visibility/contact descriptives; no model input."""
    runs = Path(runs)
    fields = ("uniform_random_expected_cost", "best_candidate_cost", "zero_suffix_cost",
              "original_selected_suffix_cost", "candidate_out_of_view_fraction", "candidate_contact_fraction")
    rows = []
    for group in sorted(groups):
        bank = runs / f"bank_{group}"
        manifest = _read(bank / "manifest.json")
        for item in manifest["cases"]:
            outcome_path = _bound_file(bank, item["outcome_file"], item["outcome_sha256"])
            with np.load(outcome_path, allow_pickle=False) as saved:
                costs = physical_goal_cost(saved["terminal_states"], saved["goal_state"])
                out_of_view = saved["out_of_view"]
                contact_points = saved["contact_points"]
            _require(costs.shape == (32,), "Unexpected pool-cost vector")
            _require(out_of_view.shape == contact_points.shape == (32, 20), "Unexpected visibility/contact arrays")
            _require(np.isfinite(contact_points).all() and (contact_points >= 0).all(), "Invalid contact-point counts")
            _require(np.isin(out_of_view, [False, True]).all(), "Invalid out-of-view indicators")
            input_path = _bound_file(bank, item["input_file"], item["input_sha256"])
            with np.load(input_path, allow_pickle=False) as inputs:
                _require(np.all(inputs["suffix_actions"][1] == 0), "Zero-action baseline is not candidate one")
            rows.append({"group": group, "case_index": int(item["index"]),
                         "goal_index": int(item.get("goal_index", item["index"])), "seed": int(item["seed"]),
                         "outcome_sha256": item["outcome_sha256"],
                         "uniform_random_expected_cost": float(costs.mean()), "best_candidate_cost": float(costs.min()),
                         "zero_suffix_cost": float(costs[1]), "original_selected_suffix_cost": float(costs[0]),
                         "candidate_out_of_view_fraction": float(np.any(out_of_view, axis=1).mean()),
                         "candidate_contact_fraction": float(np.any(contact_points > 0, axis=1).mean())})
    per_goal = []
    for goal in sorted({row["goal_index"] for row in rows}):
        selected = [row for row in rows if row["goal_index"] == goal]
        per_goal.append({"goal_index": goal, "group_count": len(selected),
                         **{field: float(np.mean([row[field] for row in selected])) for field in fields}})
    per_group = []
    for group in sorted(groups):
        selected = [row for row in rows if row["group"] == group]
        per_group.append({"group": group, "goal_count": len(selected), "per_goal": selected,
                          "equal_goal_means": {field: float(np.mean([row[field] for row in selected])) for field in fields}})
    return {"equal_goal_means": {field: float(np.mean([row[field] for row in per_goal])) for field in fields},
            "per_goal": per_goal, "per_group": per_group,
            "definitions": {"uniform_random_expected_cost": "Mean realized cost over the fixed 32-candidate pool",
                            "best_candidate_cost": "Minimum realized cost within each group/goal pool, averaged after taking each pool minimum",
                            "zero_suffix_cost": "Realized cost of candidate index 1, twenty zero actions",
                            "original_selected_suffix_cost": "Realized cost of candidate index 0, the original selected sequence's suffix",
                            "candidate_out_of_view_fraction": "Fraction of candidates with an out-of-view flag at any of twenty suffix steps",
                            "candidate_contact_fraction": "Fraction of candidates with at least one contact point at any of twenty suffix steps",
                            "aggregation": "Equal group weighting within goal, then equal goal weighting; no candidate filtering"}}


def summarize(runs, groups, draws=0):
    rows, provenance, roster = load_accepted_groups(runs, groups)
    return {"status": "ACCEPTED_DEVELOPMENT_FEEDBACK_RANKING_SUMMARY", "phase": "development",
            "groups": sorted(groups), **summarize_rows(rows, draws),
            "pool_baselines": summarize_pools(runs, groups),
            "case_goal_seed_roster": [{"case_index": case, "goal_index": goal, "seed": seed}
                                      for case, goal, seed in roster],
            "sources": provenance, "source_sha256": sha(__file__),
            "metrics_sha256": sha(Path(__file__).with_name("feedback_ranking_metrics.py")),
            "scope": "Complete requested model roster and paired goals; fixed candidate action-sequence selection after a common observed prefix. Development data, not independent confirmation or a closed-loop control result."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--groups", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--draws", type=int, default=0, help="0 omits intervals; use 20000 for exploratory intervals")
    args = parser.parse_args()
    result = summarize(args.runs, args.groups, args.draws)
    with Path(args.output).open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "groups": result["groups"],
                      "goals": result["goal_count"], "models": result["model_count"], "draws": args.draws}), flush=True)


if __name__ == "__main__":
    main()
