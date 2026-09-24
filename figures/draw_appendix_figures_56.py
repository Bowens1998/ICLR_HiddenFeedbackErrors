"""Redraw Figures 5–6 from frozen display data; no training or statistics.

Run from any directory. Requires NumPy and Matplotlib. The adjacent JSON binds
all plotted coordinates to the original accepted reports by SHA-256 checksum.
Only figure assets are written. SVG text remains editable.
"""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
COLORS = ["#B76B20", "#3977A8", "#087E78"]
OBJECTIVES = {
    "latent": "Latent",
    "decoded_teacher": "Coordinate\nteacher",
    "physical_labels": "Physical labels",
}


def configure():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.labelcolor": "#25394F", "text.color": "#25394F",
    })


def keep_axes(ax, axis):
    """Retain the original scientific range and numerical tick locations."""
    ax.set_xticks(axis["xticks"])
    ax.set_yticks(axis["yticks"])
    ax.set_xlim(axis["xlim"])
    ax.set_ylim(axis["ylim"])


def head_learning_curves(data):
    fig, axs = plt.subplots(3, 4, figsize=(7.2, 6.4), sharex=True)
    for panel, ax in zip(data["head_panels"], axs.flat):
        for fit in panel["curves"]:
            color = COLORS[0 if fit["size"] == 5000 else 1 if fit["size"] == 20000 else 2]
            ax.plot(fit["steps"], fit["relative_mse"], color=color,
                    lw=1.4 if fit["width"] == 512 else .7,
                    ls="-" if fit["seed_role"] == "A" else "--", alpha=.8)
        ax.axhline(.9, color="#65758A", ls=":", lw=.8)
        ax.set_title(f'Pool {panel["pool"]} / {panel["architecture"]}\n{panel["domain"]}', fontsize=9)
        ax.grid(alpha=.12)
        if panel["pool"] == 2:
            ax.set_xlabel("Readout updates", fontsize=8)
        ax.tick_params(labelsize=8)
        keep_axes(ax, panel["axis"])
    fig.supylabel("Validation six-output MSE / original-readout MSE", fontsize=10)
    fig.suptitle("All 72 fits: orange 5k, blue 20k, green available max\n"
                 "Solid seed A; dashed seed B; thick width 512, thin width 256", fontsize=9)
    fig.tight_layout(rect=(.02, 0, 1, .96))
    return fig


def decision_decomposition(data):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.65), gridspec_kw={"width_ratios": [1.35, 1]})
    for i, row in enumerate(data["decision_rows"]):
        for offset, key, color in [(-.12, "cost_mse", "#3977A8"), (.12, "bias_squared", "#087E78")]:
            x = row["metrics"][key]
            v = x["actual_minus_free"] / 1e6
            lo, hi = np.asarray(x["ci95"]) / 1e6
            axs[0].errorbar(v, i + offset, xerr=[[v - lo], [hi - v]],
                           fmt="o", c=color, capsize=2, markersize=4,
                           label={"cost_mse": "Cost MSE", "bias_squared": "Shared bias squared"}[key] if i == 0 else None)
        x = row["metrics"]["centered_variance"]
        v = x["actual_minus_free"] / 1e6
        lo, hi = np.asarray(x["ci95"]) / 1e6
        axs[1].errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt="o", c="#B76B20", capsize=2, markersize=4)
    labels = [OBJECTIVES[row["objective"]] for row in data["decision_rows"]]
    for ax, axis in zip(axs, data["decision_axes"]):
        ax.set_yticks(range(3), labels)
        ax.invert_yaxis()
        ax.axvline(0, c="#8494A5", lw=.8, ls="--")
        ax.grid(axis="x", alpha=.15)
        ax.set_xlabel("Actual − free (10⁶ cost²)")
        keep_axes(ax, axis)
        ax.set_yticklabels(labels)
    axs[0].set_title("Cost-estimation error", loc="left", fontweight="bold", fontsize=10)
    axs[0].legend(frameon=False, fontsize=8, loc="upper left")
    axs[1].set_title("Centered cost-error\nvariance", loc="left", fontweight="bold", fontsize=10)
    fig.tight_layout(w_pad=1.6)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure()
    data = json.loads((HERE / "appendix_figure_56_data.json").read_text())
    for name, draw in [("head_learning_curves", head_learning_curves), ("decision_decomposition", decision_decomposition)]:
        fig = draw(data)
        for suffix in ["pdf", "svg", "png"]:
            fig.savefig(args.output_dir / f"{name}.{suffix}", bbox_inches="tight", dpi=190,
                        metadata={"Creator": "Scientific figure source"} if suffix == "pdf" else None)
        plt.close(fig)


if __name__ == "__main__":
    main()
