#!/usr/bin/env python3
"""Draw a readout-scoped training comparison from accepted, complete-roster data.

Portable redraw (NumPy and Matplotlib):
  python draw_training_readout_sensitivity.py
Rebuild and verify bundled plotting data against repository evidence:
  python draw_training_readout_sensitivity.py --rebuild-from-repo /path/to/repo

No models, interventions, statistical analyses, or scientific results are changed.
The display uses descriptive points; paired confidence intervals are not drawn.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
NAME = "training_readout_sensitivity"
OBJECTIVES = ("decoded_teacher", "physical_labels")
CONDITIONS = ("T0", "T1", "T2")
COLORS = {"decoded_teacher": "#087F83", "physical_labels": "#C27A35"}
MARKERS = {"T0": "s", "T1": "o", "T2": "^"}
INK, MUTED, LINE = "#203247", "#596B7C", "#D7E0E8"
SCORES_HASH = "f628068b8a0bc9cef347576d72f31888d8caf5c03bbcf946426c99a629e27b2a"
METADATA_HASH = "8ce5b281b49f7b2aa1efa2ddcf28e882b0062eba13b0695c734e023305d3b91e"
SUMMARY_HASH = "15e131d0df76f70728d97249684c2cfd9283a1673be191f2f3cc74cd95012e6d"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_data(repo):
    base = Path("strengthening/presubmission_v8_20260922")
    paths = {
        "g_A_plotting_data": Path("paper/iclr2027/figures/result_figure_data.json"),
        "g_eval_scores": base / "verification/readout_score_inputs_v1/scores.npz",
        "g_eval_metadata": base / "independent_readout/reports/scores_v1/scores_metadata.json",
        "g_eval_accepted_summary": base / "independent_readout/reports/statistics_v1/summary.json",
        "root_acceptance": base / "verification/G_EVAL_FINAL_ACCEPTANCE.lock.json",
    }
    for key, expected in (("g_eval_scores", SCORES_HASH), ("g_eval_metadata", METADATA_HASH),
                          ("g_eval_accepted_summary", SUMMARY_HASH)):
        assert sha(repo / paths[key]) == expected, (key, "source hash mismatch")
    original = json.loads((repo / paths["g_A_plotting_data"]).read_text())["training"]
    metadata = json.loads((repo / paths["g_eval_metadata"]).read_text())
    summary = json.loads((repo / paths["g_eval_accepted_summary"]).read_text())
    acceptance = json.loads((repo / paths["root_acceptance"]).read_text())
    assert acceptance["status"] == "ROOT_ACCEPTED_COMPLETE_G_EVAL_RESULT"
    assert metadata["scores_sha256"] == SCORES_HASH
    source_rows = metadata["cell_metadata"]
    records, aggregates = [], []
    assert len(original) == 6
    for row in original:
        assert len(row["pool_free"]) == len(row["pool_G"]) == 3
        for pool in range(3):
            records.append({"readout": "g_A", "objective": row["objective"],
                            "condition": row["condition"], "pool": pool,
                            "free_mse": row["pool_free"][pool], "G": row["pool_G"][pool]})
        for field, expected in (("pool_free", row["free_mean"]), ("pool_G", row["G_mean"])):
            assert np.isclose(np.mean(row[field]), expected, rtol=0, atol=1e-9)
    with np.load(repo / paths["g_eval_scores"], allow_pickle=False) as scores:
        assert scores["model_keys"].tolist() == [row["key"] for row in source_rows]
        assert scores["branch_names"].tolist() == ["free", "actual", "donor", "reset"]
        assert scores["horizons_primitive"].tolist() == [5, 10, 15, 20, 25]
        assert scores["goal_ids"].tolist() == list(range(256))
        assert scores["stream_ids"].tolist() == list(range(4))
        assert scores["complete"].all() and scores["qualified"].all()
        errors = scores["block_error"]
        assert errors.shape == (48, 256, 4, 4, 5) and errors.dtype == np.float64
        assert np.isfinite(errors).all()
        c_rows = [(i, row) for i, row in enumerate(source_rows) if row["study"] == "C"]
        assert len(c_rows) == 18
        assert {(row["pool"], row["objective"], row["condition"]) for _, row in c_rows} == {
            (pool, objective, condition) for pool in range(3)
            for objective in OBJECTIVES for condition in CONDITIONS}
        for i, row in c_rows:
            assert row["constraint"] == "single_A" and row["architecture"] == "transformer_jepa"
            assert row["group"] == 2 * row["pool"]
            free, actual = errors[i, :, :, 0, -1], errors[i, :, :, 1, -1]
            records.append({"readout": "g_eval", "objective": row["objective"],
                            "condition": row["condition"], "pool": row["pool"],
                            "model_key": row["key"], "group": row["group"],
                            "free_mse": float(free.mean()), "G": float((free - actual).mean())})
    for readout in ("g_A", "g_eval"):
        for objective in OBJECTIVES:
            for condition in CONDITIONS:
                cells = [r for r in records if (r["readout"], r["objective"], r["condition"]) ==
                         (readout, objective, condition)]
                assert len(cells) == 3 and sorted(r["pool"] for r in cells) == [0, 1, 2]
                aggregates.append({"readout": readout, "objective": objective, "condition": condition,
                                   "free_mse": float(np.mean([r["free_mse"] for r in cells])),
                                   "G": float(np.mean([r["G"] for r in cells]))})
    residuals = []
    for contrast in summary["C_secondary"]:
        left, right = contrast["comparison"].split("-")
        endpoint = "free_mse" if contrast["measure"] == "free_error" else "G"
        rows = {r["condition"]: r for r in aggregates if r["readout"] == "g_eval" and
                r["objective"] == contrast["objective"]}
        checks = [rows[left][endpoint] - contrast["left_mean"],
                  rows[right][endpoint] - contrast["right_mean"],
                  rows[left][endpoint] - rows[right][endpoint] - contrast["difference"]]
        assert max(abs(v) for v in checks) < 1e-9
        residuals.extend(checks)
    return {
        "title": "Training comparisons depend on the evaluation readout",
        "schema": "readout_training_joint_points_v1",
        "endpoint": "Terminal block-position squared Euclidean error at primitive action 25",
        "units": "pixels squared; no division by two",
        "G_definition": "Mean free-rollout error minus mean actual-guided error under the named readout",
        "population": "All three original fixed Transformer pools, both coordinate objectives, all T0/T1/T2 conditions; 256 original goals and four reference streams",
        "scope": "Matched C branches only. g_eval is post-hoc measurement sensitivity on the original goals. Its predictions enter scoring only. No full-feasible C branches are mixed into this display.",
        "aggregation": "Within each pool: mean over all 256 goals and four streams; large points: equal-weight mean over the three pools",
        "display": "Objective colors, training-condition marker shapes, small faint pool trajectories, large aggregate trajectories. All coordinates use the named readout. No confidence regions or intervals are drawn; no cross-readout absolute-accuracy comparison is claimed.",
        "sources": {key: {"path": str(path), "sha256": sha(repo / path)} for key, path in paths.items()},
        "generator_sha256": sha(Path(__file__)),
        "checks": {"all_18_C_cells_retained": True, "all_256_goals_and_4_streams_retained": True,
                   "all_source_readouts_qualified": True, "all_source_branches_complete": True,
                   "source_hashes_match_accepted": True,
                   "g_A_original_pool_means_match": True,
                   "g_eval_eight_accepted_contrasts_match": True,
                   "maximum_summary_reconstruction_absolute_error": max(abs(v) for v in residuals)},
        "pool_points": records, "aggregate_points": aggregates,
        "suggested_caption": "Training outcomes under two evaluation readouts, at the original common standardized displacement norm. Colors identify objectives and marker shapes identify T0, T1 and T2. Large points average all three fixed Transformer pools; small points retain each pool. Lines connect T0 to T1 to T2 within pools and for their means. Each panel uses its named readout to compute both free-rollout MSE and G, the remaining improvement available to actual-guided correction. Moving left lowers free-rollout MSE; moving down lowers G. The additional T2 free-rollout improvement is resolved under g_eval, while the original g_A T2–T1 comparisons are unresolved. These are descriptive points, not joint confidence regions; the separate paired intervals are reported in the appendices. The evaluation-only analysis reuses the original goals."
    }


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.3,
        "axes.labelsize": 9.3, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": LINE, "axes.linewidth": .7,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "svg.hashsalt": NAME,
        "pdf.fonttype": 42, "savefig.facecolor": "white",
        "mathtext.fontset": "dejavusans",
    })


def draw(data, output):
    setup()
    fig = plt.figure(figsize=(7.5, 2.88))
    axes = [fig.add_axes([.103, .295, .382, .49]), fig.add_axes([.587, .295, .382, .49])]
    labels = [("g_A", r"Preserved readout $g_A$"), ("g_eval", r"Evaluation-only readout $g_{\mathrm{eval}}$")]
    for panel, (ax, (readout, title)) in enumerate(zip(axes, labels)):
        fig.text(ax.get_position().x0 - .032, .965, "AB"[panel], fontweight="bold", fontsize=11, color=COLORS["decoded_teacher"], va="top")
        fig.text(ax.get_position().x0, .965, title, fontweight="bold", fontsize=10.5, va="top")
        for objective in OBJECTIVES:
            color = COLORS[objective]
            for pool in range(3):
                rows = {r["condition"]: r for r in data["pool_points"] if
                        (r["readout"], r["objective"], r["pool"]) == (readout, objective, pool)}
                ax.plot([rows[c]["free_mse"] for c in CONDITIONS], [rows[c]["G"] for c in CONDITIONS],
                        color=color, alpha=.22, linewidth=.7, zorder=1)
                for condition in CONDITIONS:
                    r = rows[condition]
                    ax.scatter(r["free_mse"], r["G"], marker=MARKERS[condition], s=17,
                               facecolors="white", edgecolors=color, linewidths=.6, alpha=.4, zorder=2)
            rows = {r["condition"]: r for r in data["aggregate_points"] if
                    (r["readout"], r["objective"]) == (readout, objective)}
            ax.plot([rows[c]["free_mse"] for c in CONDITIONS], [rows[c]["G"] for c in CONDITIONS],
                    color=color, linewidth=1.7, zorder=3)
            for condition in CONDITIONS:
                r = rows[condition]
                ax.scatter(r["free_mse"], r["G"], marker=MARKERS[condition], s=42,
                           color=color, edgecolors="white", linewidths=.7, zorder=4)
        ax.set_xlim(4200, 11200)
        ax.set_ylim(170, 1110)
        ax.set_xticks([5000, 7000, 9000, 11000])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k"))
        ax.set_yticks([250, 500, 750, 1000])
        ax.tick_params(length=2.5, width=.6, pad=3, labelsize=8.5)
        ax.grid(color=LINE, linewidth=.55, alpha=.62)
        ax.set_axisbelow(True)
        ax.set_xlabel(r"Free-rollout MSE (pixels$^2$)", labelpad=4)
    axes[0].set_ylabel(r"$G$: free $-$ actual (pixels$^2$)", labelpad=4)
    objectives = [Line2D([], [], color=COLORS[o], linewidth=1.7, label=l) for o, l in
                  zip(OBJECTIVES, ("Coordinate teacher", "Physical labels"))]
    fig.legend(handles=objectives, loc="upper center", bbox_to_anchor=(.5, .89), ncol=2,
               frameon=False, fontsize=9.1, handlelength=1.7, columnspacing=2.2, borderaxespad=0)
    conditions = [Line2D([], [], linestyle="none", marker=MARKERS[c], color=INK,
                        markerfacecolor=INK, markeredgecolor="white", markersize=6,
                        label=label) for c, label in zip(CONDITIONS,
                        ("T0: observed history", "T1: recursive", "T2: + latent anchoring"))]
    fig.legend(handles=conditions, loc="lower center", bbox_to_anchor=(.5, .055), ncol=3,
               frameon=False, fontsize=9, handletextpad=.3, columnspacing=1.55, borderaxespad=0)
    fig.text(.5, .016, "Large points: three-pool means. Small points: all pools. Lines connect T0 → T1 → T2.",
             ha="center", va="bottom", fontsize=8.4, color=MUTED)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{NAME}.pdf", metadata={"Creator": "Scientific vector figure source",
               "Title": data["title"], "CreationDate": None, "ModDate": None})
    fig.savefig(output / f"{NAME}.svg", metadata={"Creator": "Scientific vector figure source",
               "Title": data["title"], "Date": None})
    fig.savefig(output / f"{NAME}.png", dpi=250, metadata={"Title": data["title"]})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-from-repo", type=Path)
    parser.add_argument("--data", type=Path, default=HERE / f"{NAME}.json")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    if args.rebuild_from_repo:
        data = build_data(args.rebuild_from_repo.resolve())
        args.data.parent.mkdir(parents=True, exist_ok=True)
        args.data.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    else:
        data = json.loads(args.data.read_text())
    assert len(data["pool_points"]) == 36 and len(data["aggregate_points"]) == 12
    draw(data, args.output_dir)
    print(json.dumps({"figure": NAME, "checks": data["checks"], "artifacts": {
        f"{NAME}.{ext}": sha(args.output_dir / f"{NAME}.{ext}") for ext in ("pdf", "svg", "png")}}, indent=2))


if __name__ == "__main__":
    main()
