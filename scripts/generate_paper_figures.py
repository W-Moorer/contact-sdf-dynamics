from __future__ import annotations
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "validation_summary.json"
SUPPLEMENTAL_RESULTS = ROOT / "results" / "supplemental_validation_summary.json"
PAPER = ROOT / "paper"
FIGURES = PAPER / "figures" / "08_numerical_validation"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.3,
    "ytick.labelsize": 6.3,
    "legend.fontsize": 6.2,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})

COLORS = {
    "projection": "#272727",
    "trilinear": "#666A70",
    "tricubic": "#9AA6B8",
    "uniform": "#2B8C84",
    "adaptive": "#D75A45",
}


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
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.55)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.6)


def panel_label(ax, label: str) -> None:
    ax.text(-0.18, 1.06, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8, fontweight="bold")


def grouped_bars(ax, x, series, width, labels=None) -> None:
    offsets = (np.arange(len(series)) - 0.5 * (len(series) - 1)) * width
    for idx, (name, values) in enumerate(series.items()):
        label = labels.get(name, name) if labels else name
        ax.bar(x + offsets[idx], values, width, color=COLORS[name], label=label)


def generate_main_figure() -> None:
    rows = load_rows()
    labels = ["ellipsoid", "hex prism", "cone"]
    x = np.arange(len(rows))
    width = 0.17
    method_labels = {
        "trilinear": "trilinear SDF",
        "tricubic": "tricubic SDF",
        "uniform": "uniform atlas",
        "adaptive": "feature-adaptive",
        "projection": "online projection",
    }

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), constrained_layout=True)
    ax = axes.ravel()

    normal_series = {
        "trilinear": [r["grid_normal_mean_deg"] for r in rows],
        "tricubic": [r["tricubic_normal_mean_deg"] for r in rows],
        "uniform": [r["atlas_best_normal_mean_deg"] for r in rows],
        "adaptive": [r["adaptive_best_normal_mean_deg"] for r in rows],
    }
    grouped_bars(ax[0], x, normal_series, width, labels=method_labels)
    ax[0].set_title("Normal consistency")
    ax[0].set_ylabel("mean angular error (deg, lower)")

    gap_series = {
        "trilinear": [r["grid_gap_rmse"] for r in rows],
        "tricubic": [r["tricubic_gap_rmse"] for r in rows],
        "uniform": [r["atlas_gap_rmse"] for r in rows],
        "adaptive": [r["adaptive_gap_rmse"] for r in rows],
    }
    grouped_bars(ax[1], x, gap_series, width, labels=method_labels)
    ax[1].set_title("Gap accuracy")
    ax[1].set_ylabel("gap RMSE (lower)")

    sharp_rows = [r for r in rows if r["sharp_queries"] > 0]
    sharp_labels = ["hex prism", "cone"]
    sx = np.arange(len(sharp_rows))
    cone_width = 0.30
    ax[2].bar(sx - cone_width / 2,
              [r["atlas_sharp_cone_hit_rate"] for r in sharp_rows],
              cone_width, color=COLORS["uniform"], label=method_labels["uniform"])
    ax[2].bar(sx + cone_width / 2,
              [r["adaptive_sharp_cone_hit_rate"] for r in sharp_rows],
              cone_width, color=COLORS["adaptive"], label=method_labels["adaptive"])
    ax[2].set_ylim(0, 1.05)
    ax[2].set_title("Sharp-feature sectors")
    ax[2].set_ylabel("normal-cone hit rate (higher)")
    for xi, r in zip(sx, sharp_rows):
        gain = r["adaptive_sharp_cone_hit_rate"] - r["atlas_sharp_cone_hit_rate"]
        ax[2].text(xi, min(r["adaptive_sharp_cone_hit_rate"] + 0.06, 1.02),
                   f"+{gain:.2f}", ha="center", va="bottom", fontsize=6.2,
                   color=COLORS["adaptive"])

    query_series = {
        "projection": [r["projection_us_per_query"] for r in rows],
        "trilinear": [r["trilinear_us_per_query"] for r in rows],
        "tricubic": [r["tricubic_us_per_query"] for r in rows],
        "adaptive": [r["adaptive_us_per_query"] for r in rows],
    }
    grouped_bars(ax[3], x, query_series, width, labels=method_labels)
    ax[3].set_yscale("log")
    ax[3].set_title("Compact query cost")
    ax[3].set_ylabel(r"time per query ($\mu$s, lower)")

    for i, axis in enumerate(ax):
        if i == 2:
            axis.set_xticks(sx)
            axis.set_xticklabels(sharp_labels, rotation=0)
        else:
            axis.set_xticks(x)
            axis.set_xticklabels(labels, rotation=18, ha="right")
        style_axis(axis)
        panel_label(axis, "abcd"[i])

    legend_order = ["projection", "trilinear", "tricubic", "uniform", "adaptive"]
    handles = [Patch(facecolor=COLORS[name], label=method_labels[name]) for name in legend_order]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=6.4,
               frameon=False, bbox_to_anchor=(0.5, 1.04))
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig_results.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig_results.svg", bbox_inches="tight")


def line_pair(ax, x, y_uniform, y_adaptive, *, xlabel: str, ylabel: str,
              title: str, xticklabels=None, ylim=None) -> None:
    ax.plot(x, y_uniform, marker="o", linewidth=1.5, markersize=4,
            color=COLORS["uniform"], label="uniform atlas")
    ax.plot(x, y_adaptive, marker="s", linewidth=1.5, markersize=4,
            color=COLORS["adaptive"], label="feature-adaptive")
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

    for i, axis in enumerate(ax):
        title = axis.get_title()
        axis.set_title(title[4:] if title.startswith("(") else title)
        panel_label(axis, "abcdef"[i])

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.04))
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig_supplemental_validation.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig_supplemental_validation.svg", bbox_inches="tight")


def main() -> None:
    generate_main_figure()
    generate_supplemental_figure()


if __name__ == "__main__":
    main()
