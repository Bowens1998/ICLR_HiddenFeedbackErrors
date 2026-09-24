"""Validate fresh feedback-ranking scores against roots, fibers and held-out outcomes.

Usage: python scripts/visual/accept_fresh_feedback_ranking.py --bank BANK --scores
SCORES --roots ROOTS --fibers FIBERS --protocol FILE --output acceptance.json. Output contains per-model/case metrics and
equal-goal summaries for each of the three continuation objectives. This
acceptor reconstructs readouts, costs and decisions in NumPy; it does not rerun
the neural dynamics or claim an independent physical simulator reproduction.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from feedback_ranking_metrics import (BRANCHES, aggregate_by_objective,
                                      evaluate_case, physical_goal_cost)


OBJECTIVES = {2: "unit_latent", 3: "unit_decoded_teacher", 4: "unit_physical_labels"}
HORIZONS = [10, 15, 20, 25]


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


def _arrays(path):
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key].copy() for key in values.files}


def decode_pose(tokens, head, return_normalized=False):
    """Full frozen ReLU head, reconstructed with NumPy float64 arithmetic."""
    values = (np.asarray(tokens, dtype=np.float64) - head["mean"]) / head["scale"]
    for layer in (0, 2, 4):
        values = values @ head[f"{layer}.weight"].astype(np.float64).T + head[f"{layer}.bias"].astype(np.float64)
        if layer < 4:
            values = np.maximum(values, 0)
    if return_normalized:
        return values
    return values * head["target_scale"] + head["target_mean"]


def pose_goal_cost(endpoint_pose, goal_pose):
    angle = np.arctan2(endpoint_pose[..., 4], endpoint_pose[..., 5]) - np.arctan2(goal_pose[4], goal_pose[5])
    angle = np.arctan2(np.sin(angle), np.cos(angle))
    return np.square(endpoint_pose[..., 2:4] - goal_pose[2:4]).sum(-1) + 900 * angle**2


def region_margins(head, original, replacement):
    """Independently check both original ReLU activation-region inequalities."""
    x = (original.astype(np.float64) - head["mean"]) / head["scale"]
    d = (replacement.astype(np.float64) - original) / head["scale"]
    w0, w1 = head["0.weight"], head["2.weight"]
    first = w0 @ x + head["0.bias"]
    second = w1 @ np.maximum(first, 0) + head["2.bias"]
    second_map = w1 @ ((first >= 0)[:, None] * w0)
    signs = np.where(np.r_[first, second] >= 0, 1., -1.)
    return signs * (np.r_[first, second] + np.r_[w0, second_map] @ d)


def validate_inputs(roots, fibers, score_report, bank_manifest, protocol, count):
    """Bind each scored root and correction to its fresh, case-aligned input."""
    rr, fr = _read(roots / "report.json"), _read(fibers / "report.json")
    _require(rr["status"] == "FRESH_FEEDBACK_ROOTS_READY", "Unexpected fresh-root stage")
    _require(fr["status"] == "FRESH_FEEDBACK_FIBERS_READY", "Unexpected fresh-fiber stage")
    root_hash, fiber_hash = sha(roots / "report.json"), sha(fibers / "report.json")
    _require(score_report["roots_report_sha256"] == fr["roots_report_sha256"] == root_hash,
             "Fresh root report binding mismatch")
    _require(score_report["fibers_report_sha256"] == fiber_hash, "Fresh fiber report binding mismatch")
    for value in (rr, fr):
        _require(value["group"] == bank_manifest["group"], "Fresh input group mismatch")
        _require(value["case_count"] == value["expected_cases"] == count, "Fresh input case count mismatch")
        _require(value["protocol_sha256"] == sha(protocol), "Fresh input protocol mismatch")
    for value, name in ((rr, "extract_fresh_feedback_roots.py"), (fr, "prepare_fresh_feedback_fibers.py")):
        _require(value["source_sha256"] == sha(Path(__file__).with_name(name)), f"Fresh input source changed: {name}")
    _require(fr["projection_source_sha256"] == sha(Path(__file__).with_name("readout_fiber.py")),
             "Projection implementation changed")
    _require(rr["plan_sha256"] == score_report["plan_sha256"], "Root plan mismatch")
    _require(rr["bank_manifest_sha256"] == score_report["bank_manifest_sha256"], "Root bank mismatch")
    expected_models = score_report["expected_model_indices"]
    root_models = {int(row["model_index"]): row for row in rr["models"]}
    fiber_models = {int(row["model_index"]): row for row in fr["models"]}
    for rows, models in ((rr["models"], root_models), (fr["models"], fiber_models)):
        _require(len(rows) == 3 and sorted(models) == expected_models, "Incomplete fresh input model roster")
    case_roster = sorted((int(x["index"]), int(x["seed"]), x["input_sha256"]) for x in bank_manifest["cases"])
    _require(sorted((int(x["index"]), int(x["seed"]), x["input_sha256"]) for x in rr["cases"]) == case_roster,
             "Root recipient input roster mismatch")
    recipient_seeds = np.asarray([x[1] for x in case_roster], dtype=np.int64)
    root_arrays, fiber_arrays, bindings = {}, {}, []
    for scored in score_report["models"]:
        mi = int(scored["model_index"])
        r, f = root_models[mi], fiber_models[mi]
        _require(r["objective"] == f["objective"] == scored["objective"], "Fresh input objective mismatch")
        _require(r["head_sha256"] == f["head_sha256"] == scored["head_sha256"], "Fresh input head mismatch")
        _require(scored["roots_sha256"] == f["root_file_sha256"] == r["sha256"], "Root array hash mismatch")
        _require(scored["fiber_sha256"] == f["sha256"], "Fiber array hash mismatch")
        _require(r["model_weights_sha256"] == scored["model_weights_sha256"], "Root/scorer model weights differ")
        _require(r["initial_state_hash"] == scored["state_hash"], "Root/scorer model tensor state differs")
        root = _arrays(_bound_file(roots, r["file"], r["sha256"]))
        fiber = _arrays(_bound_file(fibers, f["file"], f["sha256"]))
        _bound_file(roots, r["head_file"], r["head_sha256"])
        expected_shapes = {"predicted": (count, 192), "observed": (count, 192),
                           "donor": (count, 192), "initial_history": (count, 3, 192),
                           "goal_tokens": (count, 192), "seeds": (count,),
                           "donor_indices": (count,), "donor_seeds": (count,)}
        for key, shape in expected_shapes.items():
            _require(root[key].shape == shape and np.isfinite(root[key]).all(), f"Invalid root {key}")
        np.testing.assert_array_equal(root["seeds"], recipient_seeds)
        _require(not set(root["donor_seeds"].tolist()) & set(recipient_seeds.tolist()),
                 "Donor and fresh recipient seeds overlap")
        for key in ("predicted", "observed", "donor", "constrained", "shuffled", "full", "shuffled_full"):
            _require(fiber[key].shape == (count, 192) and np.isfinite(fiber[key]).all(), f"Invalid fiber {key}")
        for key in ("predicted", "observed", "donor"):
            np.testing.assert_array_equal(fiber[key], root[key])
        _require(fiber["target_norm"].shape == (count,) and np.isfinite(fiber["target_norm"]).all()
                 and np.all(fiber["target_norm"] >= 0), "Invalid matched displacement norms")
        _require(fiber["displacement_norms"].shape == (count, 2)
                 and np.isfinite(fiber["displacement_norms"]).all(), "Invalid recorded displacement norms")
        np.testing.assert_allclose(fiber["displacement_norms"], np.repeat(fiber["target_norm"][:, None], 2, axis=1),
                                   rtol=1e-6, atol=1e-6)
        root_arrays[mi], fiber_arrays[mi] = root, fiber
        bindings.append({"model_index": mi, "root_file": r["file"], "root_sha256": r["sha256"],
                         "fiber_file": f["file"], "fiber_sha256": f["sha256"], "head_sha256": r["head_sha256"]})
    return root_arrays, fiber_arrays, {"roots_report_sha256": root_hash, "fibers_report_sha256": fiber_hash,
                                      "models": bindings, "donor_recipient_seed_overlap": False}


def accept(bank, scores, roots, fibers, protocol, expected_cases=512):
    _require(expected_cases in (8, 512), "Expected 8 engineering cases or the frozen 512 confirmation cases")
    phase = "confirmation" if expected_cases == 512 else "engineering"
    bank, scores, roots, fibers = map(Path, (bank, scores, roots, fibers))
    manifest, report = _read(bank / "manifest.json"), _read(scores / "report.json")
    bank_acceptance = _read(bank / "acceptance.json")
    _require(manifest["status"] == "ENGINEERING_SHARED_PREFIX_RANKING_BANK", "Unexpected bank stage")
    _require(bank_acceptance["status"] == "PASS_SHARED_PREFIX_SIMULATOR_BANK", "Bank anchors not accepted")
    _require(bank_acceptance["manifest_sha256"] == sha(bank / "manifest.json"), "Stale bank acceptance")
    _require(report["status"] == "CONFIRMATION_SCORES_REQUIRE_OUTCOME_ACCEPTANCE", "Unexpected scoring stage")
    _require(report["phase"] == "confirmation", "Only fresh confirmation-source scores are accepted")
    _require(report["expected_cases"] == expected_cases, "Scorer expected-case count differs")
    _require(report["execution_scope"] == ("frozen_confirmation" if expected_cases == 512 else "engineering_smoke"), "Scoring execution scope differs")
    _require(report["protocol_sha256"] == sha(protocol), "Scoring protocol hash mismatch")
    _require(report["bank_manifest_sha256"] == sha(bank / "manifest.json"), "Scoring bank mismatch")
    _require(report["group"] == manifest["group"], "Model/bank group mismatch")
    _require(report["plan_sha256"] == manifest["bindings"]["plan_sha256"], "Plan binding mismatch")
    _require(report["branches"] == list(BRANCHES) and report["horizons"] == HORIZONS, "Branch/horizon order mismatch")
    group = int(manifest["group"])
    expected_models = [8 * group + slot for slot in OBJECTIVES]
    _require(report["expected_model_indices"] == expected_models, "Declared model roster is incomplete or incorrect")
    actual_models = [row["model_index"] for row in report["models"]]
    _require(len(actual_models) == 3 and sorted(actual_models) == expected_models, "Missing or duplicate model results")
    _require(manifest["candidates"] == 32 and manifest["executed_actions"] == 5 and manifest["future_actions"] == 20,
             "Unexpected shared-prefix experiment dimensions")
    items = manifest["cases"]
    indices = [item["index"] for item in items]
    _require(len(indices) == expected_cases and sorted(indices) == list(range(expected_cases)), "Missing, duplicate or unexpected bank cases")
    _require(len({int(item["seed"]) for item in items}) == expected_cases, "Recipient seeds must be unique")
    _require(len({int(item.get("goal_index", item["index"])) for item in items}) == expected_cases, "Recipient goal identities must be unique")
    _require(bank_acceptance["cases"] == len(items), "Bank case count mismatch")
    _require(len({row["head_sha256"] for row in report["models"]}) == 1, "Readout differs across objectives")
    for name, key in (("score_fresh_feedback_ranking.py", "source_sha256"), ("feedback_suffix_rollout.py", "helper_sha256")):
        _require(report[key] == sha(Path(__file__).with_name(name)), f"Scoring source changed: {name}")

    root_inputs, fiber_inputs, input_provenance = validate_inputs(roots, fibers, report, manifest, protocol, expected_cases)

    physical, bank_bindings = {}, []
    for item in items:
        inp = _bound_file(bank, item["input_file"], item["input_sha256"])
        outcome = _bound_file(bank, item["outcome_file"], item["outcome_sha256"])
        x, y = _arrays(inp), _arrays(outcome)
        _require(int(x["index"]) == int(y["index"]) == item["index"], "Candidate case identity mismatch")
        _require(int(x["seed"]) == int(y["seed"]) == item["seed"], "Candidate seed mismatch")
        _require(x["suffix_actions"].shape == (32, 20, 2), "Unexpected suffix action dimensions")
        _require(np.isfinite(x["suffix_actions"]).all(), "Nonfinite candidate actions")
        _require(not ({"terminal_states", "goal_state", "suffix_states", "current_state"} & set(x)),
                 "Physical outcome leaked into scorer-input archive")
        _require(y["terminal_states"].shape == (32, 7) and y["goal_state"].shape == (7,), "Unexpected physical state dimensions")
        _require(y["suffix_states"].shape == (32, 21, 7), "Unexpected candidate physical trajectories")
        np.testing.assert_array_equal(y["suffix_states"][:, -1], y["terminal_states"])
        np.testing.assert_array_equal(y["suffix_states"][:, 0], np.broadcast_to(y["current_state"], (32, 7)))
        physical[item["index"]] = physical_goal_cost(y["terminal_states"], y["goal_state"])
        bank_bindings.append({"index": item["index"], "input_sha256": item["input_sha256"],
                              "outcome_sha256": item["outcome_sha256"]})

    rows, displacements, full_norms, target_norms = [], {}, [], []
    maxima = {"normalized_readout_error": 0., "pose_reconstruction_error": 0.,
              "pose_cost_reconstruction_error": 0., "latent_cost_reconstruction_error": 0.}
    for entry in sorted(report["models"], key=lambda row: row["model_index"]):
        mi = entry["model_index"]
        objective = entry["objective"]
        _require(objective == OBJECTIVES[mi % 8], "Model objective/slot mismatch")
        case_rows = entry["cases"]
        found = [row["index"] for row in case_rows]
        _require(len(found) == len(indices) and sorted(found) == sorted(indices), "Missing or duplicate scored cases")
        head_path = _bound_file(scores, entry["head_file"], entry["head_sha256"])
        head = _arrays(head_path)
        _require(head["mean"].shape == head["scale"].shape == (192,), "Wrong readout input dimension")
        _require(head["target_mean"].shape == head["target_scale"].shape == (6,), "Wrong readout output dimension")
        _require(all(np.isfinite(value).all() for value in head.values()), "Nonfinite readout parameters")
        _require(np.all(head["scale"] > 0) and np.all(head["target_scale"] > 0), "Nonpositive readout scales")
        head = {key: value.astype(np.float64) for key, value in head.items()}
        root, fiber = root_inputs[mi], fiber_inputs[mi]
        full_norms.append(np.stack([np.linalg.norm((fiber[key].astype(np.float64) - fiber["predicted"]) / head["scale"], axis=-1)
                                   for key in ("full", "shuffled_full")], axis=-1))
        target_norms.append(fiber["target_norm"])
        indexed = {row["index"]: row for row in case_rows}
        for item in items:
            index, case = item["index"], indexed[item["index"]]
            _require(case["input_sha256"] == item["input_sha256"] and case["seed"] == item["seed"], "Scored input binding mismatch")
            path = _bound_file(scores, case["file"], case["sha256"])
            z = _arrays(path)
            np.testing.assert_array_equal(z["branches"], BRANCHES)
            np.testing.assert_array_equal(z["horizons"], HORIZONS)
            shapes = {"initial_tokens": (4, 192), "tokens": (4, 4, 32, 192), "pose": (4, 4, 32, 6),
                      "goal_token": (192,), "goal_pose": (6,), "pose_cost": (4, 32), "latent_cost": (4, 32)}
            for key, shape in shapes.items():
                _require(z[key].shape == shape and np.isfinite(z[key]).all(), f"Invalid {key} array in model {mi}, case {index}")
            _require(z["initial_tokens"].dtype == z["tokens"].dtype == np.float32,
                     "Feedback and rollout tokens must retain model float32 precision")
            expected_initial = np.stack([fiber[key][index] for key in ("predicted", "constrained", "shuffled", "observed")])
            np.testing.assert_array_equal(z["initial_tokens"], expected_initial)
            np.testing.assert_array_equal(z["goal_token"], root["goal_tokens"][index])
            initial_pose = decode_pose(z["initial_tokens"], head, True)
            error = float(np.max(np.abs(initial_pose[1:3] - initial_pose[0])))
            _require(error <= 1e-6, "Pose-preserving feedback failed complete nonlinear readout check")
            norms = np.linalg.norm((z["initial_tokens"][1:3].astype(np.float64) - z["initial_tokens"][0]) / head["scale"], axis=-1)
            np.testing.assert_allclose(norms[0], norms[1], rtol=2e-6, atol=2e-6)
            np.testing.assert_allclose(norms, fiber["target_norm"][index], rtol=1e-6, atol=1e-6)
            for replacement in z["initial_tokens"][1:3]:
                _require(float(np.min(region_margins(head, z["initial_tokens"][0], replacement))) >= -1e-6, "Replacement left original ReLU region")
            displacements.setdefault(index, []).extend(norms.tolist())
            pose = decode_pose(z["tokens"], head)
            goal_pose = decode_pose(z["goal_token"], head)
            pose_cost = pose_goal_cost(pose[:, -1], goal_pose)
            latent_cost = np.square(z["tokens"][:, -1].astype(np.float64) - z["goal_token"].astype(np.float64)).sum(-1)
            for key, expected in (("pose", pose), ("goal_pose", goal_pose), ("pose_cost", pose_cost), ("latent_cost", latent_cost)):
                np.testing.assert_allclose(z[key], expected, rtol=1e-10, atol=1e-8)
            maxima["normalized_readout_error"] = max(maxima["normalized_readout_error"], error)
            for key, expected, saved in (("pose_reconstruction_error", pose, z["pose"]),
                                         ("pose_cost_reconstruction_error", pose_cost, z["pose_cost"]),
                                         ("latent_cost_reconstruction_error", latent_cost, z["latent_cost"])):
                maxima[key] = max(maxima[key], float(np.max(np.abs(expected - saved))))
            rows.append({"model_index": mi, "objective": objective, "group": group,
                         "case_index": index, "goal_index": int(item.get("goal_index", index)), "seed": item["seed"],
                         "score_sha256": case["sha256"], "normalized_readout_error": error,
                         "standardized_displacement_norms": norms.tolist(),
                         "metrics": {space: evaluate_case(cost, physical[index])
                                     for space, cost in (("pose", pose_cost), ("latent", latent_cost))}})
    common = np.min(np.stack(full_norms), axis=(0, 2))
    for norms in target_norms:
        np.testing.assert_allclose(norms, common, rtol=1e-12, atol=1e-12,
                                   err_msg="Target displacement is not the minimum of all six full corrections")
    for index, norms in displacements.items():
        np.testing.assert_allclose(norms, np.repeat(norms[0], 6), rtol=2e-6, atol=2e-6,
                                   err_msg=f"Cross-objective/source norm mismatch for case {index}")
    return {"status": "PASS_FRESH_FEEDBACK_RANKING_NUMPY_ACCEPTANCE", "phase": phase,
            "expected_cases": expected_cases, "protocol_sha256": sha(protocol), "input_provenance": input_provenance,
            "group": group, "goal_count": len({row["goal_index"] for row in rows}),
            "case_count": len(items), "model_count": len(report["models"]), "candidate_count": 32,
            "branches": list(BRANCHES), "horizons": HORIZONS, "rows": rows,
            "objectives": aggregate_by_objective(rows), "maximum_errors": maxima,
            "metric_protocol": {"primary_score": "pose", "secondary_score": "latent",
                                "selection": "Uniform exact-argmin tie expected realized cost; deterministic lowest index also logged",
                                "physical_cost": "Squared block xy distance + 900 * squared wrapped orientation error",
                                "regret": "Selected realized cost minus best realized candidate cost",
                                "rank_ties": "Exact equality; average ranks for Spearman; Kendall tau-b",
                                "undefined_correlations": "null with valid/total counts; no imputation",
                                "aggregation": "Average fixed model/context rows within goal, then equal-weight goals, separately by objective",
                                "intervals": "None at group acceptance; fixed-roster confirmation summary applies the frozen inference protocol"},
            "bank_bindings": bank_bindings, "bank_manifest_sha256": sha(bank / "manifest.json"),
            "bank_acceptance_sha256": sha(bank / "acceptance.json"), "score_report_sha256": sha(scores / "report.json"),
            "acceptor_sha256": sha(__file__), "metrics_sha256": sha(Path(__file__).with_name("feedback_ranking_metrics.py")),
            "scope": "Independent NumPy readout, score, tie handling and physical-outcome metric reconstruction. Creation-time physical anchors are inherited; neural transitions and the simulator are not independently rerun here. Full 512-case groups are eligible for confirmation aggregation; 8-case engineering smoke is never confirmation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("bank", "scores", "roots", "fibers", "protocol", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--expected-cases", type=int, default=512, choices=(8, 512))
    args = parser.parse_args()
    result = accept(args.bank, args.scores, args.roots, args.fibers, args.protocol, args.expected_cases)
    with Path(args.output).open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(result["status"], result["model_count"], result["case_count"], flush=True)


if __name__ == "__main__":
    main()
