"""Independently decode saved feedback rollouts and join held-out outcomes.

Usage: python scripts/visual/accept_feedback_ranking.py --bank BANK --scores
SCORES --output acceptance.json. Output contains per-model/case metrics and
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


def accept(bank, scores):
    bank, scores = Path(bank), Path(scores)
    manifest, report = _read(bank / "manifest.json"), _read(scores / "report.json")
    bank_acceptance = _read(bank / "acceptance.json")
    _require(manifest["status"] == "ENGINEERING_SHARED_PREFIX_RANKING_BANK", "Unexpected bank stage")
    _require(bank_acceptance["status"] == "PASS_SHARED_PREFIX_SIMULATOR_BANK", "Bank anchors not accepted")
    _require(bank_acceptance["manifest_sha256"] == sha(bank / "manifest.json"), "Stale bank acceptance")
    _require(report["status"] == "DEVELOPMENT_SCORES_REQUIRE_OUTCOME_ACCEPTANCE", "Unexpected scoring stage")
    _require(report["phase"] == "development", "This protocol only accepts development results")
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
    _require(bool(indices) and len(set(indices)) == len(indices), "Empty or duplicate bank cases")
    _require(bank_acceptance["cases"] == len(items), "Bank case count mismatch")
    _require(len({row["head_sha256"] for row in report["models"]}) == 1, "Readout differs across objectives")
    for name, key in (("score_feedback_ranking.py", "source_sha256"), ("feedback_suffix_rollout.py", "helper_sha256")):
        _require(report[key] == sha(Path(__file__).with_name(name)), f"Scoring source changed: {name}")

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

    rows, displacements = [], {}
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
            initial_pose = decode_pose(z["initial_tokens"], head, True)
            error = float(np.max(np.abs(initial_pose[1:3] - initial_pose[0])))
            _require(error <= 1e-6, "Pose-preserving feedback failed complete nonlinear readout check")
            norms = np.linalg.norm((z["initial_tokens"][1:3].astype(np.float64) - z["initial_tokens"][0]) / head["scale"], axis=-1)
            np.testing.assert_allclose(norms[0], norms[1], rtol=2e-6, atol=2e-6)
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
    for index, norms in displacements.items():
        np.testing.assert_allclose(norms, np.repeat(norms[0], 6), rtol=2e-6, atol=2e-6,
                                   err_msg=f"Cross-objective/source norm mismatch for case {index}")
    return {"status": "PASS_DEVELOPMENT_FEEDBACK_RANKING_NUMPY_ACCEPTANCE", "phase": "development",
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
                                "intervals": "None: engineering development, not independent confirmation"},
            "bank_bindings": bank_bindings, "bank_manifest_sha256": sha(bank / "manifest.json"),
            "bank_acceptance_sha256": sha(bank / "acceptance.json"), "score_report_sha256": sha(scores / "report.json"),
            "acceptor_sha256": sha(__file__), "metrics_sha256": sha(Path(__file__).with_name("feedback_ranking_metrics.py")),
            "scope": "Independent NumPy readout, score, tie handling and physical-outcome metric reconstruction. Creation-time physical anchors are inherited; neural transitions and the simulator are not independently rerun here. Development results only."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("bank", "scores", "output"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    result = accept(args.bank, args.scores)
    with Path(args.output).open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(result["status"], result["model_count"], result["case_count"], flush=True)


if __name__ == "__main__":
    main()
