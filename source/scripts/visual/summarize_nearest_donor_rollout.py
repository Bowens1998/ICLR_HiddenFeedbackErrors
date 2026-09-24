"""Goal-paired development summary of four matched feedback sources.

The complete result requires 128 goals and the fixed 18-model roster. All
intervals are exploratory 95% with no familywise or independent-confirmation
claim. Eight-case engineering outputs omit intervals.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from accept_nearest_donor_rollout import BRANCHES, HORIZONS, OBJECTIVES, bound, read, require, sha


BOOTSTRAP_SEED = 1390001
BOOTSTRAP_DRAWS = 20000
PLANNED_CONTRASTS = (("actual", "pose_nearest"), ("actual", "latent_nearest"),
                     ("actual", "random"), ("actual", "free"))
CONTRASTS = PLANNED_CONTRASTS + (("pose_nearest", "free"), ("latent_nearest", "free"),
                                  ("random", "free"), ("reset", "free"))


def paired_interval(values, samples):
    return np.quantile(np.asarray(values, dtype=np.float64)[samples].mean(axis=1), [.025, .975]).tolist()


def summarize_rows(rows, expected_cases=128, groups=tuple(range(6))):
    require(expected_cases in (8, 128), "Expected eight smoke or 128 development goals")
    require(bool(groups) and len(set(groups)) == len(groups) and set(groups) <= set(range(6)), "Invalid group roster")
    if expected_cases == 128:
        require(sorted(groups) == list(range(6)), "Complete development summary requires all six groups")
    models = {8 * group + slot for group in groups for slot in OBJECTIVES}
    expected = {(mi, i) for mi in models for i in range(expected_cases)}
    found = [(r["model_index"], r["goal_index"]) for r in rows]
    require(len(found) == len(expected) and set(found) == expected, "Missing or duplicate model-goal rows")
    lookup = {(r["model_index"], r["goal_index"]): r for r in rows}
    goal_seeds = {}
    for row in rows:
        mi, i = row["model_index"], row["goal_index"]
        require(row["group"] == mi // 8 and row["objective"] == OBJECTIVES[mi % 8] and row["case_index"] == i, "Row identity mismatch")
        if i in goal_seeds:
            require(goal_seeds[i] == row["seed"], "Unpaired goal seed across groups")
        goal_seeds[i] = row["seed"]
        values = np.asarray(row["position_mse"])
        require(values.shape == (6, 4) and np.isfinite(values).all() and np.all(values >= 0), "Invalid prediction error matrix")
    require(len(set(goal_seeds.values())) == expected_cases, "Duplicate goal seeds")
    full = expected_cases == 128
    samples = np.random.default_rng(BOOTSTRAP_SEED).integers(0, expected_cases,
                  size=(BOOTSTRAP_DRAWS, expected_cases)) if full else None
    objectives, per_model, endpoints = {}, [], []
    for slot, objective in OBJECTIVES.items():
        # Models are fixed repeated contexts: first average them within each goal.
        tensor = np.asarray([[lookup[(8 * g + slot, i)]["position_mse"] for i in range(expected_cases)] for g in sorted(groups)], dtype=np.float64)
        per_goal = tensor.mean(axis=0)
        means = per_goal.mean(axis=0)
        contrasts = {}
        for left, right in CONTRASTS:
            li, ri = BRANCHES.index(left), BRANCHES.index(right)
            key = f"{left}_minus_{right}"
            difference = per_goal[:, li] - per_goal[:, ri]
            contrasts[key] = []
            for hi, horizon in enumerate(HORIZONS):
                contrast = {"horizon": horizon, "estimate": float(difference[:, hi].mean()),
                            "interval": paired_interval(difference[:, hi], samples) if full else None,
                            "interval_level": .95 if full else None,
                            "role": "planned_endpoint" if horizon == 25 and (left, right) in PLANNED_CONTRASTS else "supporting",
                            "relative_gain_percent_of_free": (-100 * float(difference[:, hi].mean()) / means[0, hi] if means[0, hi] > 0 else None)}
                contrasts[key].append(contrast)
                if horizon == 25 and (left, right) in PLANNED_CONTRASTS:
                    endpoints.append({"objective": objective, "contrast": key, **contrast})
        objectives[objective] = {"goal_count": expected_cases, "fixed_group_count": len(groups),
                                  "branches": {b: {"position_mse": means[bi].tolist()} for bi, b in enumerate(BRANCHES)},
                                  "contrasts": contrasts,
                                  "per_goal": [{"goal_index": i, "seed": goal_seeds[i], "position_mse": per_goal[i].tolist()} for i in range(expected_cases)]}
        for gi, group in enumerate(sorted(groups)):
            model_means = tensor[gi].mean(axis=0)
            per_model.append({"model_index": 8 * group + slot, "group": group, "objective": objective,
                              "position_mse": model_means.tolist(),
                              "endpoint_contrasts": {f"{left}_minus_{right}": float(model_means[BRANCHES.index(left), -1] - model_means[BRANCHES.index(right), -1])
                                                     for left, right in CONTRASTS}})
    norms = np.asarray([[lookup[(8 * g + 2, i)]["common_norm"] for i in range(expected_cases)] for g in sorted(groups)])
    require(np.isfinite(norms).all() and np.all(norms >= 0), "Invalid common norms")
    for g in sorted(groups):
        for i in range(expected_cases):
            require(all(lookup[(8 * g + s, i)]["common_norm"] == lookup[(8 * g + 2, i)]["common_norm"] for s in (3, 4)), "Within-group common norm differs by objective")
    return {"goal_count": expected_cases, "model_count": len(models), "branches": list(BRANCHES), "horizons": list(HORIZONS),
            "objectives": objectives, "planned_endpoint_contrasts": endpoints, "per_model": per_model,
            "correction_norms": {"mean": float(norms.mean()), "median": float(np.median(norms)),
                                 "minimum": float(norms.min()), "maximum": float(norms.max()),
                                 "zero_group_goal_count": int(np.count_nonzero(norms == 0)),
                                 "group_goal_count": int(norms.size), "per_group_goal": norms.tolist()},
            "bootstrap": {"draws": BOOTSTRAP_DRAWS if full else 0, "seed": BOOTSTRAP_SEED if full else None,
                          "level": .95 if full else None, "unit": "goal; fixed groups averaged before paired resampling",
                          "shared_draws": full, "method": "percentile" if full else None,
                          "interpretation": "Exploratory development intervals, no familywise correction and no independent confirmation" if full else "Engineering smoke, descriptive values only"},
            "relative_gain_definition": "Negative mean paired MSE difference divided by free MSE times 100; free is the denominator for every contrast."}


def load_groups(runs, protocol, groups, expected_cases):
    runs = Path(runs)
    rows, sources, reference_roster = [], [], None
    for group in sorted(groups):
        a = read(runs / f"acceptance_{group}.json")
        scores, bank = runs / f"scores_{group}", runs / f"bank_{group}"
        report, manifest = read(scores / "report.json"), read(bank / "manifest.json")
        require(a["status"] == "PASS_NEAREST_DONOR_ROLLOUT_ACCEPTANCE", "Nearest-donor scores not accepted")
        require(a["phase"] == ("development" if expected_cases == 128 else "engineering"), "Analysis phase differs")
        require(a["group"] == report["group"] == manifest["group"] == group, "Group mismatch")
        require(a["expected_cases"] == a["goal_count"] == report["expected_cases"] == expected_cases, "Case count mismatch")
        require(a["protocol_sha256"] == report["protocol_sha256"] == sha(protocol), "Protocol changed")
        require(a["source_sha256"] == sha(Path(__file__).with_name("accept_nearest_donor_rollout.py")), "Acceptance implementation changed")
        require(a["score_report_sha256"] == sha(scores / "report.json"), "Accepted score report changed")
        require(a["bank_manifest_sha256"] == report["bank_manifest_sha256"] == sha(bank / "manifest.json"), "Physical bank changed")
        require(a["fibers_report_sha256"] == report["fibers_report_sha256"] == sha(scores / "fibers/report.json"), "Matched fibers changed")
        for key in ("geometry_report_sha256", "anchor_score_report_sha256", "old_fibers_report_sha256"):
            require(a[key] == report[key], f"Accepted provenance changed: {key}")
        require(a["branches"] == list(BRANCHES) and a["horizons"] == list(HORIZONS), "Accepted array axes changed")
        items = sorted(manifest["cases"], key=lambda x: x["index"])
        require(len(items) == 128 and [x["index"] for x in items] == list(range(128)), "Incomplete full source bank")
        roster = [(i["index"], i["seed"]) for i in items[:expected_cases]]
        require(len({x[1] for x in roster}) == expected_cases, "Duplicate recipient seeds")
        if reference_roster is None:
            reference_roster = roster
        require(roster == reference_roster, "Unpaired goals across groups")
        bindings = {x["index"]: x for x in a["bank_bindings"]}
        require(len(a["bank_bindings"]) == len(bindings) == expected_cases, "Missing accepted physical bindings")
        for item in items[:expected_cases]:
            for kind in ("input", "outcome"):
                require(bindings[item["index"]][f"{kind}_sha256"] == item[f"{kind}_sha256"], "Accepted physical outcome binding mismatch")
                bound(bank, item[f"{kind}_file"], item[f"{kind}_sha256"])
        models = {m["model_index"]: m for m in report["models"]}
        expected_models = [8 * group + slot for slot in OBJECTIVES]
        require(len(report["models"]) == 3 and sorted(models) == report["expected_model_indices"] == expected_models, "Incomplete model roster")
        expected_rows, hashes = set(), {}
        for mi, model in models.items():
            bound(scores, model["head_file"], model["head_sha256"])
            require(model["objective"] == OBJECTIVES[mi % 8], "Model objective changed")
            require(len(model["cases"]) == expected_cases and sorted(x["index"] for x in model["cases"]) == list(range(expected_cases)), "Scored cases incomplete")
            for case in model["cases"]:
                i = case["index"]
                require(case["seed"] == items[i]["seed"] and case["input_sha256"] == items[i]["input_sha256"], "Case binding mismatch")
                bound(scores, case["file"], case["sha256"])
                expected_rows.add((mi, i)); hashes[(mi, i)] = case["sha256"]
        found = [(r["model_index"], r["case_index"]) for r in a["rows"]]
        require(len(found) == len(expected_rows) and set(found) == expected_rows, "Accepted metric rows incomplete")
        for row in a["rows"]:
            require(row["score_sha256"] == hashes[(row["model_index"], row["case_index"])], "Accepted score binding mismatch")
            require(row["seed"] == items[row["case_index"]]["seed"], "Accepted goal identity mismatch")
        rows.extend(a["rows"])
        sources.append({"group": group, "acceptance_sha256": sha(runs / f"acceptance_{group}.json"),
                        "score_report_sha256": a["score_report_sha256"], "geometry_report_sha256": a["geometry_report_sha256"],
                        "donor_reuse": a["donor_reuse"], "maximum_errors": a["maximum_errors"]})
    return rows, sources


def summarize(runs, protocol, expected_cases=128, groups=tuple(range(6))):
    require(expected_cases in (8, 128), "Expected eight smoke or 128 development goals")
    require(bool(groups) and len(set(groups)) == len(groups) and set(groups) <= set(range(6)), "Invalid group roster")
    if expected_cases == 128:
        require(sorted(groups) == list(range(6)), "Complete development summary requires all six groups")
    rows, sources = load_groups(runs, protocol, groups, expected_cases)
    return {"status": "ACCEPTED_NEAREST_DONOR_ROLLOUT_SUMMARY", "phase": "development" if expected_cases == 128 else "engineering",
            "groups": sorted(groups), **summarize_rows(rows, expected_cases, groups), "sources": sources,
            "protocol_sha256": sha(protocol), "source_sha256": sha(__file__),
            "acceptor_sha256": sha(Path(__file__).with_name("accept_nearest_donor_rollout.py")),
            "scope": "Fixed-action prediction-error sensitivity to four matched guidance sources on reused goals. No new action-selection outcome, independent confirmation, equivalence claim or physical-variable identification."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("runs", "protocol", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--groups", type=int, nargs="+", default=list(range(6)))
    p.add_argument("--expected-cases", type=int, choices=(8, 128), default=128)
    a = p.parse_args()
    result = summarize(a.runs, a.protocol, a.expected_cases, a.groups)
    with Path(a.output).open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False); handle.write("\n")
    print(result["status"], result["phase"], result["goal_count"], result["model_count"], flush=True)


if __name__ == "__main__":
    main()
