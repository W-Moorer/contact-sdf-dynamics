from __future__ import annotations
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "validation_summary.json"
PAPER = ROOT / "paper"


def load_rows() -> list[dict]:
    if not RESULTS.exists():
        raise FileNotFoundError("Run scripts/build_and_validate.py before generating paper figures.")
    with RESULTS.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    rows = load_rows()
    labels = ["ellipsoid", "hex prism", "cone"]
    x = np.arange(len(rows))
    width = 0.24

    colors = {
        "grid": "#6c757d",
        "uniform": "#2a9d8f",
        "adaptive": "#e76f51",
    }

    fig, axes = plt.subplots(1, 3, figsize=(8.8, 2.7), constrained_layout=True)

    gap_grid = [r["grid_gap_rmse"] for r in rows]
    gap_uniform = [r["atlas_gap_rmse"] for r in rows]
    gap_adapt = [r["adaptive_gap_rmse"] for r in rows]
    axes[0].bar(x - width, gap_grid, width, label="grid SDF", color=colors["grid"])
    axes[0].bar(x, gap_uniform, width, label="uniform atlas", color=colors["uniform"])
    axes[0].bar(x + width, gap_adapt, width, label="feature-adaptive", color=colors["adaptive"])
    axes[0].set_title("(a) gap accuracy")
    axes[0].set_ylabel("gap RMSE")

    n_grid = [r["grid_normal_mean_deg"] for r in rows]
    n_uniform = [r["atlas_best_normal_mean_deg"] for r in rows]
    n_adapt = [r["adaptive_best_normal_mean_deg"] for r in rows]
    axes[1].bar(x - width, n_grid, width, color=colors["grid"])
    axes[1].bar(x, n_uniform, width, color=colors["uniform"])
    axes[1].bar(x + width, n_adapt, width, color=colors["adaptive"])
    axes[1].set_title("(b) normal accuracy")
    axes[1].set_ylabel("mean normal error (deg)")

    speed = [r["speedup_projection_vs_adaptive"] for r in rows]
    axes[2].bar(x, speed, 0.42, color=colors["adaptive"])
    axes[2].set_title("(c) runtime speedup")
    axes[2].set_ylabel("projection/adaptive time")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    axes[0].legend(loc="upper right", fontsize=7, frameon=False)
    fig.savefig(PAPER / "fig_results.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
