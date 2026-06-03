from __future__ import annotations
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "validation_summary.json"
SUPPLEMENTAL_RESULTS = ROOT / "results" / "supplemental_validation_summary.json"
PAPER = ROOT / "paper"


def load_rows() -> list[dict]:
    if not RESULTS.exists():
        raise FileNotFoundError("Run scripts/build_and_validate.py before generating paper figures.")
    with RESULTS.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_supplemental_rows() -> tuple[dict[str, dict], dict]:
    if not SUPPLEMENTAL_RESULTS.exists():
        raise FileNotFoundError("Run scripts/supplemental_validation.py before generating supplemental figures.")
    with SUPPLEMENTAL_RESULTS.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {row["case"]: row for row in data["rows"]}, data.get("rigid_transform", {})


def style_axis(ax) -> None:
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def generate_main_figure() -> None:
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
        style_axis(ax)

    axes[0].legend(loc="upper right", fontsize=7, frameon=False)
    fig.savefig(PAPER / "fig_results.pdf", bbox_inches="tight")


def line_pair(ax, x, y_uniform, y_adaptive, *, xlabel: str, ylabel: str,
              title: str, xticklabels=None, ylim=None) -> None:
    colors = {"uniform": "#2a9d8f", "adaptive": "#e76f51"}
    ax.plot(x, y_uniform, marker="o", linewidth=1.5, markersize=4,
            color=colors["uniform"], label="uniform atlas")
    ax.plot(x, y_adaptive, marker="s", linewidth=1.5, markersize=4,
            color=colors["adaptive"], label="feature-adaptive")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xticklabels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels)
    if ylim is not None:
        ax.set_ylim(*ylim)
    style_axis(ax)


def generate_supplemental_figure() -> None:
    rows, rigid = load_supplemental_rows()
    plt.rcParams.update({
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
    })
    fig, axes = plt.subplots(2, 3, figsize=(8.8, 5.8), constrained_layout=True)
    ax = axes.ravel()

    res = np.array([7, 9, 13])
    sphere_rows = [rows[f"ablation_resolution_sphere_res{r}"] for r in res]
    line_pair(
        ax[0], res,
        [r["uniform_normal_mean_deg"] for r in sphere_rows],
        [r["adaptive_normal_mean_deg"] for r in sphere_rows],
        xlabel="uniform resolution", ylabel="mean normal error (deg)",
        title="(a) exact sphere normals", ylim=(0, 4.6)
    )

    line_pair(
        ax[1], res,
        [r["uniform_hessian_rmse"] for r in sphere_rows],
        [r["adaptive_hessian_rmse"] for r in sphere_rows],
        xlabel="uniform resolution", ylabel="Hessian RMSE",
        title="(b) exact sphere Hessian", ylim=(0, 0.95)
    )

    zones = ["side", "rim", "apex"]
    cone_rows = [rows[f"cone_zones:cone_{z}"] for z in zones]
    line_pair(
        ax[2], np.arange(len(zones)),
        [r["uniform_cone_hit_rate"] for r in cone_rows],
        [r["adaptive_cone_hit_rate"] for r in cone_rows],
        xlabel="cone region", ylabel="normal-cone hit rate",
        title="(c) cone-zone sectors", xticklabels=zones, ylim=(0, 1.05)
    )

    depths = np.array([1, 2, 3])
    cyl_depth_rows = [rows[f"ablation_feature_depth_cylinder_d{d}"] for d in depths]
    line_pair(
        ax[3], depths,
        [r["uniform_cone_hit_rate"] for r in cyl_depth_rows],
        [r["adaptive_cone_hit_rate"] for r in cyl_depth_rows],
        xlabel="feature max depth", ylabel="normal-cone hit rate",
        title="(d) cylinder feature depth", ylim=(0.65, 0.95)
    )

    segs = np.array([12, 24, 48])
    cyl_mesh_rows = [rows[f"mesh_consistency_cylinder_{s}"] for s in segs]
    line_pair(
        ax[4], segs,
        [r["uniform_cone_hit_rate"] for r in cyl_mesh_rows],
        [r["adaptive_cone_hit_rate"] for r in cyl_mesh_rows],
        xlabel="cylinder segments", ylabel="normal-cone hit rate",
        title="(e) mesh refinement", ylim=(0.35, 1.05)
    )

    noise = np.array([2, 5, 10])
    noise_rows = [rows[f"noisy_sphere_normals_{n:g}deg"] for n in noise]
    line_pair(
        ax[5], noise,
        [r["uniform_normal_mean_deg"] for r in noise_rows],
        [r["adaptive_normal_mean_deg"] for r in noise_rows],
        xlabel="normal noise (deg)", ylabel="mean normal error (deg)",
        title="(f) noisy normal stress", ylim=(0, 9.5)
    )
    ax[5].text(0.03, 0.92, "limitation\ncase", transform=ax[5].transAxes,
               ha="left", va="top", fontsize=6.5, color="#6c757d")

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.04))
    fig.savefig(PAPER / "fig_supplemental_validation.pdf", bbox_inches="tight")
    fig.savefig(PAPER / "fig_supplemental_validation.svg", bbox_inches="tight")


def main() -> None:
    generate_main_figure()
    generate_supplemental_figure()


if __name__ == "__main__":
    main()
