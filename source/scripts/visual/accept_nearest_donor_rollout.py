"""Independently reconstruct nearest-donor prediction errors and source bindings.

Only this acceptor reads suffix simulator outcomes. The six branches use one
fixed continuation, not six choices of action. Eight-case smoke is engineering;
128-case acceptance is development and never independent confirmation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


BRANCHES = ("free", "actual", "random", "pose_nearest", "latent_nearest", "reset")
SOURCES = BRANCHES[1:5]
HORIZONS = (10, 15, 20, 25)
OBJECTIVES = {2: "unit_latent", 3: "unit_decoded_teacher", 4: "unit_physical_labels"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bound(root, name, expected_hash):
    root = Path(root).resolve()
    path = (root / name).resolve()
    require(path.is_relative_to(root), "Artifact path escapes its directory")
    require(path.is_file() and sha(path) == expected_hash, f"Artifact hash mismatch: {name}")
    return path


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def decode(tokens, head, normalized=False):
    x = (np.asarray(tokens, dtype=np.float64) - head["mean"]) / head["scale"]
    for layer in (0, 2, 4):
        x = x @ head[f"{layer}.weight"].astype(np.float64).T + head[f"{layer}.bias"].astype(np.float64)
        if layer != 4:
            x = np.maximum(x, 0)
    return x if normalized else x * head["target_scale"] + head["target_mean"]


def region_margins(head, original, replacement):
    x = (original.astype(np.float64) - head["mean"]) / head["scale"]
    d = (replacement.astype(np.float64) - original) / head["scale"]
    w0, w1 = head["0.weight"].astype(np.float64), head["2.weight"].astype(np.float64)
    first = w0 @ x + head["0.bias"]
    second = w1 @ np.maximum(first, 0) + head["2.bias"]
    second_map = w1 @ ((first >= 0)[:, None] * w0)
    signs = np.where(np.r_[first, second] >= 0, 1., -1.)
    return signs * (np.r_[first, second] + np.r_[w0, second_map] @ d)


def match_corrections(head, predicted, full, common_norm):
    """Rebuild scaled FP32 tokens, including the zero-minimum identity branch."""
    predicted, full = np.asarray(predicted), np.asarray(full)
    require(predicted.shape == (192,) and full.shape == (4, 192), "Wrong correction dimensions")
    require(np.isfinite(predicted).all() and np.isfinite(full).all(), "Nonfinite correction")
    require(np.isfinite(common_norm) and common_norm >= 0, "Invalid common correction norm")
    directions = full.astype(np.float64) - predicted.astype(np.float64)
    norms = np.linalg.norm(directions / head["scale"], axis=-1)
    require(common_norm <= float(norms.min()) + 1e-10, "Common norm exceeds an available direction")
    alphas = np.divide(common_norm, norms, out=np.zeros_like(norms), where=norms > 0)
    matched = (predicted.astype(np.float64)[None] + alphas[:, None] * directions).astype(np.float32)
    return matched, norms, alphas


def validate_geometry(selection, head):
    """Recompute both current-observation selectors; ties use lowest bank index."""
    actual, donors = selection["actual"], selection["donors"]
    require(actual.shape == donors.shape == (128, 192), "Expected two 128-token geometry banks")
    require(np.isfinite(actual).all() and np.isfinite(donors).all(), "Nonfinite source geometry")
    require(selection["recipient_seeds"].shape == selection["donor_seeds"].shape == (128,), "Wrong geometry seed roster")
    require(len(set(selection["recipient_seeds"].tolist())) == len(set(selection["donor_seeds"].tolist())) == 128,
            "Repeated source-bank seed")
    require(not set(selection["recipient_seeds"].tolist()) & set(selection["donor_seeds"].tolist()), "Recipient/donor overlap")
    ax = (actual.astype(np.float64) - head["mean"]) / head["scale"]
    dx = (donors.astype(np.float64) - head["mean"]) / head["scale"]
    pd = np.linalg.norm(decode(actual, head, True)[:, None] - decode(donors, head, True)[None], axis=-1)
    ld = np.linalg.norm(ax[:, None] - dx[None], axis=-1)
    for name, distance in (("pose_nearest", pd), ("latent_nearest", ld)):
        np.testing.assert_array_equal(selection[f"{name}_indices"], np.argmin(distance, axis=1))
    np.testing.assert_array_equal(selection["random_indices"], np.random.default_rng(1368001).permutation(128))
    np.testing.assert_allclose(selection["pairwise_normalized_pose_distance"], pd, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(selection["pairwise_standardized_latent_distance"], ld, rtol=1e-12, atol=1e-12)
    return {name: int(len(np.unique(selection[f"{name}_indices"]))) for name in ("random", "pose_nearest", "latent_nearest")}


def truth_positions(outcome):
    """Suffix starts at action 5: action 10 is suffix-state index 5, not 10."""
    states = outcome["suffix_states"]
    require(states.shape == (32, 21, 7) and np.isfinite(states).all(), "Invalid suffix truth trajectory")
    np.testing.assert_array_equal(states[:, -1], outcome["terminal_states"])
    np.testing.assert_array_equal(states[:, 0], np.broadcast_to(outcome["current_state"], (32, 7)))
    return states[0, np.asarray(HORIZONS) - 5, 2:4].astype(np.float64)


def reconstruct_case(saved, head, truth, anchor, fiber, index):
    """Check numerical branches, exact execution anchors and block-position MSE."""
    np.testing.assert_array_equal(saved["branches"], BRANCHES)
    np.testing.assert_array_equal(saved["horizons"], HORIZONS)
    for key, shape in (("initial_tokens", (6, 192)), ("tokens", (6, 4, 192)), ("pose", (6, 4, 6))):
        require(saved[key].shape == shape and np.isfinite(saved[key]).all(), f"Invalid saved {key}")
    require(saved["initial_tokens"].dtype == saved["tokens"].dtype == np.float32, "Feedback precision changed")
    np.testing.assert_array_equal(anchor["branches"], ("free", "act", "don", "reset"))
    np.testing.assert_array_equal(anchor["horizons"], HORIZONS)
    np.testing.assert_array_equal(saved["tokens"][0], anchor["tokens"][0, :, 0])
    np.testing.assert_array_equal(saved["tokens"][5], anchor["tokens"][3, :, 0])
    np.testing.assert_array_equal(saved["initial_tokens"][0], anchor["initial_tokens"][0])
    np.testing.assert_array_equal(saved["initial_tokens"][5], anchor["initial_tokens"][3])
    np.testing.assert_array_equal(fiber["predicted"][index], anchor["initial_tokens"][0])
    np.testing.assert_array_equal(fiber["observed"][index], anchor["initial_tokens"][3])
    full, common = fiber["full_correction_tokens"][index], float(fiber["common_norm"][index])
    require(saved["common_norm"].shape == () and float(saved["common_norm"]) == common,
            "Saved common norm differs from matched fibers")
    matched, norms, alphas = match_corrections(head, fiber["predicted"][index], full, common)
    np.testing.assert_array_equal(fiber["matched_correction_tokens"][index], matched)
    np.testing.assert_array_equal(saved["initial_tokens"][1:5], matched)
    np.testing.assert_allclose(fiber["full_displacement_norms"][index], norms, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fiber["match_alphas"][index], alphas, rtol=1e-12, atol=1e-12)
    measured = np.linalg.norm((matched.astype(np.float64) - saved["initial_tokens"][0]) / head["scale"], axis=-1)
    np.testing.assert_allclose(measured, np.repeat(common, 4), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(fiber["matched_displacement_norms"][index], measured, rtol=1e-12, atol=1e-12)
    initial_pose = decode(saved["initial_tokens"], head, True)
    error = float(np.max(np.abs(initial_pose[1:5] - initial_pose[0])))
    require(error <= 1e-6, "Current nonlinear readout changed")
    minimum_margin = min(float(np.min(region_margins(head, saved["initial_tokens"][0], replacement)))
                         for replacement in np.concatenate([full, matched]))
    require(minimum_margin >= -1e-6, "Correction left original ReLU region")
    full_error = float(np.max(np.abs(decode(full, head, True) - initial_pose[0])))
    require(full_error <= 1e-6, "Full correction changed nonlinear readout")
    pose = decode(saved["tokens"], head)
    np.testing.assert_allclose(saved["pose"], pose, rtol=1e-10, atol=1e-8)
    mse = np.square(pose[..., 2:4] - truth[None]).sum(-1)
    return mse, {"readout_error": error, "full_readout_error": full_error,
                 "pose_reconstruction_error": float(np.max(np.abs(pose - saved["pose"]))),
                 "norm_error": float(np.max(np.abs(measured - common))),
                 "region_violation": max(0., -minimum_margin)}, norms


def accept(bank, scores, fibers, geometry, anchors, anchor_acceptance, old_fibers, protocol, expected_cases=128):
    require(expected_cases in (8, 128), "Expected eight smoke or 128 development goals")
    bank, scores, fibers, geometry, anchors, old_fibers = map(Path, (bank, scores, fibers, geometry, anchors, old_fibers))
    bm, sr, fr, gr, ar, old = [read(p / n) for p, n in ((bank, "manifest.json"), (scores, "report.json"),
        (fibers, "report.json"), (geometry, "report.json"), (anchors, "report.json"), (old_fibers, "report.json"))]
    ba, aa, oa = read(bank / "acceptance.json"), read(anchor_acceptance), read(old_fibers / "acceptance.json")
    group = int(sr["group"])
    require(group in range(6) and bm["group"] == fr["group"] == ar["group"] == group, "Group mismatch")
    require(sr["status"] == "NEAREST_DONOR_SCORES_REQUIRE_ACCEPTANCE" and sr["phase"] == "development", "Unexpected score stage")
    require(fr["status"] == "NEAREST_DONOR_MATCHED_FIBERS_READY", "Unexpected correction stage")
    require(sr["expected_cases"] == fr["expected_cases"] == expected_cases, "Incomplete expected cases")
    require(sr["case_count"] == expected_cases, "Score case count mismatch")
    require(bm["reference_route"] == 16 * group and bm["candidates"] == 32
            and bm["executed_actions"] == 5 and bm["future_actions"] == 20, "Physical bank design changed")
    require(sr["precision"] == ar["precision"], "Execution precision differs from accepted anchors")
    require(sr["protocol_sha256"] == fr["protocol_sha256"] == sha(protocol), "Protocol mismatch")
    require(sr["plan_sha256"] == ar["plan_sha256"] == bm["bindings"]["plan_sha256"], "Plan changed")
    require(sr["bank_manifest_sha256"] == fr["bank_manifest_sha256"] == ar["bank_manifest_sha256"] == sha(bank / "manifest.json"), "Bank binding mismatch")
    require(ba["status"] == "PASS_SHARED_PREFIX_SIMULATOR_BANK" and ba["manifest_sha256"] == sha(bank / "manifest.json"), "Bank not accepted")
    require(aa["status"] == "PASS_DEVELOPMENT_FEEDBACK_RANKING_NUMPY_ACCEPTANCE" and aa["phase"] == "development"
            and aa["score_report_sha256"] == sha(anchors / "report.json"), "Development anchor is not accepted")
    require(aa["bank_manifest_sha256"] == sha(bank / "manifest.json"), "Anchor physical bank differs")
    require(sr["anchor_score_report_sha256"] == sha(anchors / "report.json"), "Anchor score report mismatch")
    require(sr["fibers_report_sha256"] == sha(fibers / "report.json"), "Matched-fiber report mismatch")
    require(sr["geometry_report_sha256"] == fr["geometry_report_sha256"] == sha(geometry / "report.json"), "Geometry report mismatch")
    require(oa["status"] == "PASS1536_PAIRED_CONFIRMATION_FIBER_INPUTS" and oa["report_sha256"] == sha(old_fibers / "report.json"), "Original fibers not accepted")
    require(sr["old_fibers_report_sha256"] == fr["old_fibers_report_sha256"] == sha(old_fibers / "report.json"), "Original fiber report binding mismatch")
    require(gr["status"] == "OUTCOME_BLIND_NEAREST_DONOR_GEOMETRY_AUDIT" and gr["recipient_cases_per_group"] == 128, "Incomplete geometry audit")
    require(gr["sources"] == list(SOURCES), "Source order mismatch")
    for report, key, name in ((sr, "source_sha256", "score_nearest_donor_rollout.py"),
                              (sr, "helper_sha256", "feedback_suffix_rollout.py"),
                              (sr, "utility_source_sha256", "score_feedback_ranking.py"),
                              (fr, "source_sha256", "score_nearest_donor_rollout.py"),
                              (gr, "source_sha256", "audit_nearest_donor_geometry.py"),
                              (gr, "projection_source_sha256", "readout_fiber.py")):
        require(report[key] == sha(Path(__file__).with_name(name)), f"Source changed: {name}")
    np.testing.assert_array_equal(sr["branches"], BRANCHES)
    np.testing.assert_array_equal(sr["horizons"], HORIZONS)
    items = sorted(bm["cases"], key=lambda x: x["index"])
    require(len(items) == ba["cases"] == 128 and [x["index"] for x in items] == list(range(128)), "Bank must retain all 128 goals")
    require(len({x["seed"] for x in items}) == 128, "Duplicate recipient seed")
    selected_items = items[:expected_cases]
    expected_models = [8 * group + slot for slot in OBJECTIVES]
    require(sr["expected_model_indices"] == expected_models, "Declared model roster changed")
    indexed = []
    for entries in (sr["models"], fr["models"], ar["models"]):
        by_model = {x["model_index"]: x for x in entries}
        require(len(entries) == 3 and sorted(by_model) == expected_models, "Incomplete model roster")
        indexed.append(by_model)
    sm, fm, am = indexed
    groups = [x for x in gr["groups"] if x["group"] == group]
    require(len(groups) == 1, "Geometry group missing or duplicated")
    g = groups[0]
    require(g["fiber_report_sha256"] == sha(old_fibers / "report.json"), "Original fiber/geometry binding mismatch")
    selection = arrays(bound(geometry, g["file"], g["sha256"]))
    np.testing.assert_array_equal(selection["recipient_seeds"], [x["seed"] for x in items])
    gm = {x["model_index"]: x for x in gr["models"]}
    physical, bank_bindings = {}, []
    for item in selected_items:
        x = arrays(bound(bank, item["input_file"], item["input_sha256"]))
        y = arrays(bound(bank, item["outcome_file"], item["outcome_sha256"]))
        require(int(x["index"]) == int(y["index"]) == item["index"] and int(x["seed"]) == int(y["seed"]) == item["seed"], "Physical case identity mismatch")
        require(not ({"goal_state", "current_state", "terminal_states", "suffix_states"} & set(x)), "Scorer inputs contain physical outcomes")
        require(x["suffix_actions"].shape == (32, 20, 2), "Candidate arithmetic changed")
        physical[item["index"]] = truth_positions(y)
        bank_bindings.append({"index": item["index"], "seed": item["seed"], "input_sha256": item["input_sha256"],
                              "outcome_sha256": item["outcome_sha256"], "truth_positions": physical[item["index"]].tolist()})
    rows, all_norms, commons, model_bindings = [], [], [], []
    maxima = {k: 0. for k in ("readout_error", "full_readout_error", "pose_reconstruction_error", "norm_error", "region_violation")}
    donor_reuse = None
    for mi in expected_models:
        s, f, a, projection = sm[mi], fm[mi], am[mi], gm[mi]
        objective = OBJECTIVES[mi % 8]
        require(s["objective"] == f["objective"] == a["objective"] == objective, "Objective mismatch")
        require(s["head_sha256"] == f["head_sha256"] == a["head_sha256"] == g["head_sha256"], "Head changed")
        require(s["model_weights_sha256"] == a["model_weights_sha256"], "Model weights changed")
        require(s["state_hash"] == a["state_hash"], "Model tensors changed")
        require(s["fiber_sha256"] == f["sha256"], "Model fiber binding mismatch")
        require(s["geometry_projection_sha256"] == f["geometry_projection_sha256"] == projection["sha256"], "Projection binding mismatch")
        head = arrays(bound(scores, s["head_file"], s["head_sha256"]))
        require(head["mean"].shape == head["scale"].shape == (192,) and head["target_mean"].shape == head["target_scale"].shape == (6,), "Wrong head dimensions")
        require(all(np.isfinite(v).all() for v in head.values()) and np.all(head["scale"] > 0) and np.all(head["target_scale"] > 0), "Invalid frozen head")
        if donor_reuse is None:
            donor_reuse = validate_geometry(selection, head)
        geo = arrays(bound(geometry, projection["file"], projection["sha256"]))
        np.testing.assert_array_equal(geo["sources"], SOURCES)
        require(projection["cases"] == 128 and projection["attempts"] == projection["valid_attempts"] == 256, "Unresolved nearest projection cases")
        logs = read(bound(geometry, projection["log_file"], projection["log_sha256"]))
        require(len(logs) == 256 and {(x["source"], x["index"]) for x in logs} == {(q, i) for q in SOURCES[2:] for i in range(128)}, "Incomplete solver logs")
        require(all(x["valid"] and x["status"] == "solved" for x in logs), "Nearest projection solver failed")
        ob = [x for x in old["bindings"] if x.get("model_index") == mi]
        require(len(ob) == 1 and ob[0]["output_sha256"] == projection["old_fiber_sha256"], "Original projection provenance mismatch")
        require(s["old_fiber_sha256"] == f["old_fiber_sha256"] == ob[0]["output_sha256"], "Scored original projection binding mismatch")
        oz = arrays(bound(old_fibers, ob[0]["output_file"], ob[0]["output_sha256"]))
        np.testing.assert_array_equal(geo["predicted"], oz["predicted"][:128])
        np.testing.assert_array_equal(geo["corrections"][:, 0], oz["full"][:128])
        np.testing.assert_array_equal(geo["corrections"][:, 1], oz["shuffled_full"][:128])
        np.testing.assert_array_equal(selection["actual"], oz["observed"][:128])
        np.testing.assert_array_equal(selection["donors"][selection["random_indices"]], oz["donor"][:128])
        fiber = arrays(bound(fibers, f["file"], f["sha256"]))
        np.testing.assert_array_equal(fiber["sources"], SOURCES)
        np.testing.assert_array_equal(fiber["seeds"], [x["seed"] for x in selected_items])
        np.testing.assert_array_equal(fiber["predicted"], geo["predicted"][:expected_cases])
        np.testing.assert_array_equal(fiber["observed"], selection["actual"][:expected_cases])
        np.testing.assert_array_equal(fiber["full_correction_tokens"], geo["corrections"][:expected_cases])
        require(fiber["matched_correction_tokens"].shape == (expected_cases, 4, 192) and fiber["common_norm"].shape == (expected_cases,), "Incomplete matched fibers")
        cases = {x["index"]: x for x in s["cases"]}
        anchor_cases = {x["index"]: x for x in a["cases"]}
        require(len(s["cases"]) == len(cases) == expected_cases and sorted(cases) == list(range(expected_cases)), "Incomplete scored cases")
        require(len(a["cases"]) == len(anchor_cases) == 128 and sorted(anchor_cases) == list(range(128)), "Incomplete accepted anchor roster")
        full_norms = []
        for item in selected_items:
            i, c = item["index"], cases[item["index"]]
            ac = anchor_cases[i]
            require(c["seed"] == ac["seed"] == item["seed"] and c["input_sha256"] == ac["input_sha256"] == item["input_sha256"], "Scorer input identity changed")
            require(c["anchor_score_sha256"] == ac["sha256"], "Per-case anchor mismatch")
            saved = arrays(bound(scores, c["file"], c["sha256"]))
            anchor = arrays(bound(anchors, ac["file"], ac["sha256"]))
            mse, errors, norms = reconstruct_case(saved, head, physical[i], anchor, fiber, i)
            full_norms.append(norms)
            for key, value in errors.items():
                maxima[key] = max(maxima[key], value)
            rows.append({"model_index": mi, "group": group, "objective": objective, "goal_index": i,
                         "case_index": i, "seed": item["seed"], "position_mse": mse.tolist(),
                         "common_norm": float(fiber["common_norm"][i]), "score_sha256": c["sha256"],
                         "donor_indices": {name: int(selection[f"{name}_indices"][i]) for name in SOURCES[1:]}})
        all_norms.append(np.asarray(full_norms)); commons.append(fiber["common_norm"])
        model_bindings.append({"model_index": mi, "fiber_sha256": f["sha256"], "head_sha256": s["head_sha256"],
                               "geometry_projection_sha256": projection["sha256"], "old_fiber_sha256": ob[0]["output_sha256"]})
    minimum = np.min(np.stack(all_norms), axis=(0, 2))
    for common in commons:
        np.testing.assert_allclose(common, minimum, rtol=1e-12, atol=1e-12, err_msg="Incorrect twelve-direction minimum")
    return {"status": "PASS_NEAREST_DONOR_ROLLOUT_ACCEPTANCE", "phase": "development" if expected_cases == 128 else "engineering",
            "group": group, "expected_cases": expected_cases, "goal_count": expected_cases, "model_count": 3,
            "branches": list(BRANCHES), "horizons": list(HORIZONS), "rows": rows, "maximum_errors": maxima,
            "donor_reuse": donor_reuse, "bank_bindings": bank_bindings, "model_bindings": model_bindings,
            "protocol_sha256": sha(protocol), "bank_manifest_sha256": sha(bank / "manifest.json"),
            "score_report_sha256": sha(scores / "report.json"), "fibers_report_sha256": sha(fibers / "report.json"),
            "geometry_report_sha256": sha(geometry / "report.json"), "anchor_score_report_sha256": sha(anchors / "report.json"),
            "anchor_acceptance_sha256": sha(anchor_acceptance), "old_fibers_report_sha256": sha(old_fibers / "report.json"),
            "source_sha256": sha(__file__),
            "metric": "Squared Euclidean decoded block-position error; no coordinate division, no goal cost, suffix zero only",
            "scope": "Independent donor selection, correction scaling, nonlinear readout and fixed-action error reconstruction; development inference only. No new independent confirmation or physical-variable identification."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "scores", "fibers", "geometry", "anchors", "anchor-acceptance", "old-fibers", "protocol", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--expected-cases", type=int, choices=(8, 128), default=128)
    a = p.parse_args()
    result = accept(a.bank, a.scores, a.fibers, a.geometry, a.anchors, a.anchor_acceptance, a.old_fibers, a.protocol, a.expected_cases)
    with Path(a.output).open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False); handle.write("\n")
    print(result["status"], result["phase"], result["goal_count"], flush=True)


if __name__ == "__main__":
    main()
