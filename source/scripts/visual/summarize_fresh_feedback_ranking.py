"""Summarize the frozen 512-goal feedback-ranking confirmation without selecting models.

The full summary requires all six groups, all 18 models and 512 unique goals.
Four primary pose-score selected-cost contrasts receive two-sided 98.75%
percentile intervals with nominal Bonferroni familywise 95% coverage. Other
declared intervals are secondary 95%, never substitutes for the primary tests.
The 8-case engineering mode produces descriptive summaries without intervals.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from feedback_ranking_metrics import BRANCHES, COST_SPACES, aggregate_by_objective, physical_goal_cost


OBJECTIVES = {2: "unit_latent", 3: "unit_decoded_teacher", 4: "unit_physical_labels"}
PRIMARY_CONTRASTS = ("act_minus_free", "act_minus_don")
BOOTSTRAP_SEED = 1389001
BOOTSTRAP_DRAWS = 20000
PRIMARY_OBJECTIVES = ("unit_decoded_teacher", "unit_physical_labels")
PRIMARY_LEVEL = .9875


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


def load_accepted_groups(runs, groups, protocol, expected_cases=512):
    """Validate complete accepted artifacts and a common case/goal/seed roster."""
    runs = Path(runs)
    _require(bool(groups) and len(groups) == len(set(groups)), "Groups must be nonempty and unique")
    _require(all(g in range(6) for g in groups), "Group indices must be in 0..5")
    _require(expected_cases in (8, 512), "Expected 8 engineering or 512 confirmation goals")
    if expected_cases == 512:
        _require(sorted(groups) == list(range(6)), "Confirmation requires all six fixed groups")
    phase = "confirmation" if expected_cases == 512 else "engineering"
    rows, provenance, reference_roster = [], [], None
    reference_protocol = None
    for group in sorted(groups):
        bank, scores = runs / f"bank_{group}", runs / f"scores_{group}"
        acceptance_path = runs / f"acceptance_{group}.json"
        acceptance = _read(acceptance_path)
        manifest, report = _read(bank / "manifest.json"), _read(scores / "report.json")
        _require(acceptance["status"] == "PASS_FRESH_FEEDBACK_RANKING_NUMPY_ACCEPTANCE"
                 and acceptance["phase"] == phase, "Group is not accepted in the requested phase")
        _require(acceptance["expected_cases"] == acceptance["case_count"] == expected_cases, "Incomplete requested goal roster")
        _require(acceptance["protocol_sha256"] == report["protocol_sha256"] == sha(protocol), "Frozen protocol hash mismatch")
        for key in ("roots_report_sha256", "fibers_report_sha256"):
            _require(acceptance["input_provenance"][key] == report[key], "Accepted fresh-input provenance mismatch")
        _require(acceptance["group"] == manifest["group"] == report["group"] == group, "Group binding mismatch")
        _require(acceptance["bank_manifest_sha256"] == report["bank_manifest_sha256"] == sha(bank / "manifest.json"),
                 "Bank manifest hash mismatch")
        _require(acceptance["bank_acceptance_sha256"] == sha(bank / "acceptance.json"), "Bank acceptance hash mismatch")
        _require(acceptance["score_report_sha256"] == sha(scores / "report.json"), "Score report hash mismatch")
        _require(acceptance["acceptor_sha256"] == sha(Path(__file__).with_name("accept_fresh_feedback_ranking.py")),
                 "Acceptance code version mismatch")
        _require(acceptance["metrics_sha256"] == sha(Path(__file__).with_name("feedback_ranking_metrics.py")),
                 "Metric code version mismatch")
        _require(acceptance["branches"] == list(BRANCHES) and acceptance["horizons"] == [10, 15, 20, 25],
                 "Accepted branch/horizon schema mismatch")
        _require(acceptance["candidate_count"] == manifest["candidates"] == 32, "Candidate count mismatch")
        metric_protocol = acceptance["metric_protocol"]
        if reference_protocol is None:
            reference_protocol = metric_protocol
        _require(metric_protocol == reference_protocol, "Metric protocols differ across groups")
        case_items = {int(item["index"]): item for item in manifest["cases"]}
        _require(len(case_items) == len(manifest["cases"]) == acceptance["case_count"], "Case count or duplicate mismatch")
        binding_items = {int(item["index"]): item for item in acceptance["bank_bindings"]}
        _require(set(binding_items) == set(case_items), "Accepted bank bindings are incomplete")
        roster = sorted((index, int(item.get("goal_index", index)), int(item["seed"]))
                        for index, item in case_items.items())
        _require([r[0] for r in roster] == list(range(expected_cases)), "Incomplete case index roster")
        _require(len({r[1] for r in roster}) == len({r[2] for r in roster}) == expected_cases, "Duplicate goals or recipient seeds")
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


def interval(values, resamples, level):
    """Percentile interval for a paired goal mean; rows are already goal means."""
    values = np.asarray(values, dtype=np.float64)
    _require(values.ndim == 1 and np.isfinite(values).all(), "Interval requires finite per-goal values")
    draws = values[resamples].mean(axis=1)
    tail = (1 - level) / 2
    return np.quantile(draws, [tail, 1 - tail]).tolist()


def _compact_objective(objective, aggregated, resamples=None):
    compact = {"goal_count": aggregated["goal_count"], "model_context_rows": aggregated["model_context_rows"],
               "score_spaces": {}}
    for space in COST_SPACES:
        means = aggregated["equal_goal_means"][space]
        free_cost = means["branches"]["free"]["selected_cost"]["mean"]
        contrasts = {}
        for name, value in means["contrasts"].items():
            is_primary = space == "pose" and objective in PRIMARY_OBJECTIVES and name in PRIMARY_CONTRASTS
            delta = value["selected_cost"]["mean"]
            result = dict(value, inferential_role="primary" if is_primary else "secondary",
                          relative_gain_percent_of_free=(-100 * delta / free_cost if free_cost > 0 else None))
            if resamples is not None:
                per_goal = [g["metrics"][space]["contrasts"][name] for g in aggregated["per_goal"]]
                level = PRIMARY_LEVEL if is_primary else .95
                result["selected_cost_interval"] = {
                    "level": level, "bounds": interval([g["selected_cost"]["mean"] for g in per_goal], resamples, level),
                    "role": "primary" if is_primary else "secondary",
                    "familywise_nominal_coverage": .95 if is_primary else None,
                }
                # Tie-rule sensitivity and changed decisions are secondary descriptions.
                for field in ("selection_set_changed", "selected_cost_lowest"):
                    result[f"{field}_secondary_ci95"] = interval([g[field]["mean"] for g in per_goal], resamples, .95)
            contrasts[name] = result
        compact["score_spaces"][space] = {"score_role": "primary" if space == "pose" else "secondary",
                                           "free_selected_cost": free_cost, "branches": means["branches"],
                                           "contrasts": contrasts}
    return compact


def summarize_rows(rows, expected_cases=512, groups=tuple(range(6))):
    """Enforce the fixed roster, then average groups before shared goal resampling."""
    _require(expected_cases in (8, 512), "Expected 8 engineering or 512 confirmation goals")
    _require(bool(groups) and len(set(groups)) == len(groups) and all(g in range(6) for g in groups), "Invalid group roster")
    if expected_cases == 512:
        _require(sorted(groups) == list(range(6)), "Confirmation requires all six fixed groups")
    expected_models = {8 * group + slot for group in groups for slot in OBJECTIVES}
    expected_pairs = {(mi, goal) for mi in expected_models for goal in range(expected_cases)}
    observed_pairs = [(int(row["model_index"]), int(row["goal_index"])) for row in rows]
    _require(len(observed_pairs) == len(expected_pairs) and set(observed_pairs) == expected_pairs,
             "Missing or duplicate fixed model/goal rows")
    for row in rows:
        mi = int(row["model_index"])
        _require(row["group"] == mi // 8 and row["objective"] == OBJECTIVES[mi % 8], "Row model/objective/group binding mismatch")
    aggregate = aggregate_by_objective(rows)
    goals = list(range(expected_cases))
    for objective in aggregate.values():
        _require([g["goal_index"] for g in objective["per_goal"]] == goals, "Unpaired objective goals")
        _require(all(g["model_context_rows"] == len(groups) for g in objective["per_goal"]), "Incomplete within-goal model roster")
    confirmation = expected_cases == 512
    resamples = np.random.default_rng(BOOTSTRAP_SEED).integers(0, expected_cases,
                    size=(BOOTSTRAP_DRAWS, expected_cases)) if confirmation else None
    objectives = {name: _compact_objective(name, value, resamples) for name, value in aggregate.items()}
    per_model = []
    for mi in sorted(expected_models):
        model_rows = [row for row in rows if row["model_index"] == mi]
        name = OBJECTIVES[mi % 8]
        per_model.append({"model_index": mi, "group": mi // 8, "objective": name,
                          **_compact_objective(name, aggregate_by_objective(model_rows)[name])})
    primary_tests = []
    if confirmation:
        for objective in PRIMARY_OBJECTIVES:
            for contrast in PRIMARY_CONTRASTS:
                value = objectives[objective]["score_spaces"]["pose"]["contrasts"][contrast]
                bounds = value["selected_cost_interval"]["bounds"]
                primary_tests.append({"objective": objective, "score": "pose", "contrast": contrast,
                                      "estimate": value["selected_cost"]["mean"], "interval_level": PRIMARY_LEVEL,
                                      "interval": bounds,
                                      "direction": "decrease" if bounds[1] < 0 else "increase" if bounds[0] > 0 else "unresolved"})
    interactions = []
    for space in ("pose",):
        for contrast in ("act_minus_free",):
            by_objective = {name: np.asarray([g["metrics"][space]["contrasts"][contrast]["selected_cost"]["mean"]
                                             for g in aggregate[name]["per_goal"]]) for name in PRIMARY_OBJECTIVES}
            values = by_objective["unit_physical_labels"] - by_objective["unit_decoded_teacher"]
            interactions.append({"score": space, "contrast": contrast, "role": "secondary",
                                 "definition": "physical-label intervention effect minus coordinate-teacher intervention effect",
                                 "estimate": float(values.mean()),
                                 "interval_level": .95 if confirmation else None,
                                 "interval": interval(values, resamples, .95) if confirmation else None})
    return {"goal_indices": goals, "goal_count": expected_cases, "model_count": len(expected_models),
            "objectives": objectives, "per_model": per_model, "goal_aggregation": aggregate,
            "primary_tests": primary_tests, "paired_objective_interactions": interactions,
            "bootstrap": {"draws": BOOTSTRAP_DRAWS if confirmation else 0,
                          "seed": BOOTSTRAP_SEED if confirmation else None,
                          "primary_interval_level": PRIMARY_LEVEL if confirmation else None,
                          "primary_test_count": 4 if confirmation else 0,
                          "nominal_familywise_coverage": .95 if confirmation else None,
                          "secondary_interval_level": .95 if confirmation else None,
                          "unit": "goal; fixed model/group rows averaged before resampling",
                          "shared_draws": confirmation, "method": "paired-goal percentile" if confirmation else None,
                          "interpretation": "Four primary contrasts, each interpreted separately; numerical acceptance does not require efficacy. Secondary intervals have no familywise claim." if confirmation else "Engineering smoke; descriptive values only"},
            "relative_gain_definition": "-100 * mean paired selected-cost difference / pooled free selected cost. The denominator is free even for actual-minus-donor; regret differences are identical evidence."}


def summarize(runs, protocol, groups=tuple(range(6)), expected_cases=512):
    from summarize_feedback_ranking import summarize_pools
    rows, provenance, roster = load_accepted_groups(runs, groups, protocol, expected_cases)
    phase = "confirmation" if expected_cases == 512 else "engineering"
    return {"status": "ACCEPTED_FRESH_FEEDBACK_RANKING_SUMMARY", "phase": phase, "groups": sorted(groups),
            **summarize_rows(rows, expected_cases, groups), "pool_baselines": summarize_pools(Path(runs), groups),
            "case_goal_seed_roster": [{"case_index": case, "goal_index": goal, "seed": seed} for case, goal, seed in roster],
            "sources": provenance, "source_sha256": sha(__file__), "protocol_sha256": sha(protocol),
            "metrics_sha256": sha(Path(__file__).with_name("feedback_ranking_metrics.py")),
            "pool_summary_source_sha256": sha(Path(__file__).with_name("summarize_feedback_ranking.py")),
            "scope": "One decision over a fixed suffix pool after a shared observed prefix; fixed model roster and donor bank. No closed-loop success claim. Engineering smoke cannot establish confirmation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runs", "protocol", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--groups", type=int, nargs="+", default=list(range(6)))
    parser.add_argument("--expected-cases", type=int, choices=(8, 512), default=512)
    args = parser.parse_args()
    result = summarize(args.runs, args.protocol, args.groups, args.expected_cases)
    with Path(args.output).open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "phase": result["phase"], "goals": result["goal_count"],
                      "models": result["model_count"], "primary_tests": len(result["primary_tests"])}), flush=True)


if __name__ == "__main__":
    main()
