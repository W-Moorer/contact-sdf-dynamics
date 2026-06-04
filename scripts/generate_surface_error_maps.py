from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contact_sdf.atlas import AdaptiveContactSDFAtlas, ContactSDFAtlas  # noqa: E402
from contact_sdf.grid_sdf import CubicGridSDF, GridSDF  # noqa: E402
from contact_sdf.mesh_format import CornerNormalMesh  # noqa: E402
from contact_sdf.metrics import best_candidate_angle_deg, percentile  # noqa: E402
from contact_sdf.projection import MeshProjector, angular_error_deg, normalize  # noqa: E402
from scripts.generate_surface_visuals import (  # noqa: E402
    cubic_bbox,
    load_mesh,
    load_or_build_adaptive_atlas,
    load_or_build_grid,
    load_or_build_uniform_atlas,
)


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
RESULTS = ROOT / "results" / "surface_error_maps"

MAIN_MODELS = ["ellipsoid", "hex_prism", "cone"]
SUPPLEMENTAL_MODELS = ["sphere", "cylinder", "torus"]
METHODS = ["trilinear", "tricubic", "uniform_atlas", "feature_adaptive"]
METHOD_LABELS = {
    "trilinear": "trilinear",
    "tricubic": "tricubic",
    "uniform_atlas": "uniform atlas",
    "feature_adaptive": "adaptive atlas",
}
MODEL_LABELS = {
    "ellipsoid": "ellipsoid",
    "hex_prism": "hex prism",
    "cone": "cone",
    "sphere": "sphere",
    "cylinder": "cylinder",
    "torus": "torus",
}

GAP_NORM = LogNorm(vmin=1e-4, vmax=8e-2)
NORMAL_NORM = Normalize(vmin=0.0, vmax=45.0)
GAP_CMAP = mpl.colormaps["magma"]
NORMAL_CMAP = mpl.colormaps["viridis"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 6.5,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render original-surface contact error maps for manuscript figures."
    )
    parser.add_argument("--grid-resolution", type=int, default=9)
    parser.add_argument("--force", action="store_true", help="Rebuild cached SDF/atlas data.")
    parser.add_argument(
        "--sets",
        nargs="+",
        default=["main", "supplemental"],
        choices=["main", "supplemental"],
        help="Figure sets to render.",
    )
    return parser.parse_args()


def sharp_corner_mask(mesh: CornerNormalMesh, tol: float = 1e-8, normal_tol_deg: float = 5.0) -> np.ndarray:
    points = mesh.triangles.reshape(-1, 3)
    normals = mesh.corner_normals.reshape(-1, 3)
    keys = np.round(points / tol).astype(np.int64)
    groups: dict[tuple[int, int, int], list[int]] = {}
    for i, key in enumerate(map(tuple, keys)):
        groups.setdefault(key, []).append(i)

    cos_tol = float(np.cos(np.radians(normal_tol_deg)))
    mask = np.zeros(points.shape[0], dtype=bool)
    for ids in groups.values():
        unique: list[np.ndarray] = []
        for idx in ids:
            n = normals[idx]
            if not any(float(np.dot(n, u)) > cos_tol for u in unique):
                unique.append(n)
        if len(unique) > 1:
            mask[np.asarray(ids, dtype=np.int64)] = True
    return mask.reshape(mesh.n_faces, 3)


def face_normals(triangles: np.ndarray) -> np.ndarray:
    e1 = triangles[:, 1] - triangles[:, 0]
    e2 = triangles[:, 2] - triangles[:, 0]
    return normalize(np.cross(e1, e2))


def shaded_values_to_rgba(values: np.ndarray, metric: str) -> np.ndarray:
    if metric == "gap":
        clipped = np.clip(values, GAP_NORM.vmin, GAP_NORM.vmax)
        rgba = GAP_CMAP(GAP_NORM(clipped))
    elif metric == "normal":
        clipped = np.clip(values, NORMAL_NORM.vmin, NORMAL_NORM.vmax)
        rgba = NORMAL_CMAP(NORMAL_NORM(clipped))
    else:
        raise ValueError(f"Unknown metric: {metric}")
    rgba[:, 3] = 1.0
    return rgba


def apply_lighting(triangles: np.ndarray, rgba: np.ndarray) -> np.ndarray:
    normals = face_normals(triangles)
    light = np.asarray([-0.35, -0.45, 0.82], dtype=float)
    light /= np.linalg.norm(light)
    shade = 0.74 + 0.26 * np.clip(normals @ light, 0.0, 1.0)
    out = rgba.copy()
    out[:, :3] = out[:, :3] * shade[:, None] + 0.04 * (1.0 - shade[:, None])
    return out


def set_equal_axis(ax, mesh: CornerNormalMesh) -> None:
    vertices = mesh.triangles.reshape(-1, 3)
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    center = 0.5 * (lo + hi)
    radius = 0.56 * float(np.max(hi - lo))
    if radius <= 0.0:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=20, azim=-52)
    ax.set_axis_off()


def evaluate_method(
    method: str,
    points: np.ndarray,
    ref_normals: np.ndarray,
    grid: GridSDF,
    cubic: CubicGridSDF,
    uniform: ContactSDFAtlas,
    adaptive: AdaptiveContactSDFAtlas,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "trilinear":
        phi, normal = grid.eval(points)
        normal_error = angular_error_deg(normal, ref_normals)
        return phi, normal_error
    if method == "tricubic":
        phi, normal = cubic.eval(points)
        normal_error = angular_error_deg(normal, ref_normals)
        return phi, normal_error
    if method == "uniform_atlas":
        ev = uniform.eval(points)
        normal_error = best_candidate_angle_deg(ev.candidate_normals, ref_normals)
        return ev.phi, normal_error
    if method == "feature_adaptive":
        ev = adaptive.eval(points)
        normal_error = best_candidate_angle_deg(ev.candidate_normals, ref_normals)
        return ev.phi, normal_error
    raise ValueError(f"Unknown method: {method}")


def model_error_data(
    model: str,
    resolution: int,
    force: bool,
) -> tuple[CornerNormalMesh, dict[str, dict[str, np.ndarray]], list[dict]]:
    mesh = load_mesh(model)
    bbox = cubic_bbox(mesh)
    projector = MeshProjector(mesh, k=min(96, mesh.n_faces))
    grid = load_or_build_grid(model, projector, bbox, resolution, force=force)
    cubic = CubicGridSDF.from_grid(grid)
    uniform = load_or_build_uniform_atlas(model, projector, bbox, resolution, force=force)
    adaptive = load_or_build_adaptive_atlas(model, projector, bbox, force=force)

    face_points = mesh.triangles.mean(axis=1)
    ref = projector.project(face_points, active_tol=0.01, sector_angle_deg=35.0)
    corner_points = mesh.triangles.reshape(-1, 3)
    corner_ref_normals = mesh.corner_normals.reshape(-1, 3)
    sharp = sharp_corner_mask(mesh)
    rows: list[dict] = []
    data: dict[str, dict[str, np.ndarray]] = {}
    for method in METHODS:
        phi, normal_error = evaluate_method(method, face_points, ref.normal, grid, cubic, uniform, adaptive)
        gap_error = np.abs(phi - ref.phi)
        _corner_phi, corner_normal_error = evaluate_method(
            method,
            corner_points,
            corner_ref_normals,
            grid,
            cubic,
            uniform,
            adaptive,
        )
        miss = sharp.reshape(-1) & (corner_normal_error > 7.5)
        face_miss = miss.reshape(mesh.n_faces, 3).any(axis=1)
        sharp_count = int(sharp.sum())
        hit_rate = float(np.mean(corner_normal_error.reshape(mesh.n_faces, 3)[sharp] <= 7.5)) if sharp_count else np.nan
        data[method] = {
            "gap_face": gap_error,
            "normal_face": normal_error,
            "miss_face": face_miss,
            "miss_points": corner_points[miss],
        }
        rows.append({
            "model": model,
            "method": method,
            "surface_samples": int(face_points.shape[0]),
            "sharp_corners": sharp_count,
            "gap_mean": float(np.mean(gap_error)),
            "gap_p95": percentile(gap_error, 95),
            "normal_mean_deg": float(np.mean(normal_error)),
            "normal_p95_deg": percentile(normal_error, 95),
            "sharp_sector_hit_rate": hit_rate,
        })
    return mesh, data, rows


def render_error_panel(
    ax,
    mesh: CornerNormalMesh,
    values: np.ndarray,
    metric: str,
    miss_points: np.ndarray | None = None,
) -> None:
    triangles = mesh.triangles
    rgba = apply_lighting(triangles, shaded_values_to_rgba(values, metric))
    collection = Poly3DCollection(
        triangles,
        facecolors=rgba,
        edgecolors=(0.03, 0.03, 0.03, 0.08),
        linewidths=0.035,
        antialiased=True,
    )
    collection.set_rasterized(True)
    ax.add_collection3d(collection)
    if miss_points is not None and len(miss_points):
        ax.scatter(
            miss_points[:, 0],
            miss_points[:, 1],
            miss_points[:, 2],
            c="#b30000",
            s=1.6,
            alpha=0.68,
            depthshade=False,
        )
    set_equal_axis(ax, mesh)


def add_colorbars(fig) -> None:
    gap_sm = mpl.cm.ScalarMappable(norm=GAP_NORM, cmap=GAP_CMAP)
    normal_sm = mpl.cm.ScalarMappable(norm=NORMAL_NORM, cmap=NORMAL_CMAP)
    cax_gap = fig.add_axes([0.915, 0.55, 0.015, 0.31])
    cb_gap = fig.colorbar(gap_sm, cax=cax_gap)
    cb_gap.set_label(r"$|g|$ error", labelpad=3)
    cb_gap.set_ticks([1e-4, 1e-3, 1e-2, 8e-2])
    cb_gap.ax.tick_params(length=2, pad=1.5, labelsize=5.6)
    cax_n = fig.add_axes([0.915, 0.16, 0.015, 0.31])
    cb_n = fig.colorbar(normal_sm, cax=cax_n)
    cb_n.set_label("normal error (deg)", labelpad=3)
    cb_n.set_ticks([0, 15, 30, 45])
    cb_n.ax.tick_params(length=2, pad=1.5, labelsize=5.6)


def render_figure(
    figure_name: str,
    models: list[str],
    all_data: dict[str, tuple[CornerNormalMesh, dict[str, dict[str, np.ndarray]]]],
) -> None:
    nrows = 2 * len(models)
    ncols = len(METHODS)
    fig = plt.figure(figsize=(7.25, 1.16 * nrows + 0.68), dpi=350)
    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols,
        left=0.095,
        right=0.895,
        bottom=0.035,
        top=0.925,
        wspace=-0.14,
        hspace=-0.22,
    )

    for col, method in enumerate(METHODS):
        ax = fig.add_subplot(gs[0, col], projection="3d")
        mesh, model_data = all_data[models[0]]
        render_error_panel(ax, mesh, model_data[method]["gap_face"], "gap")
        ax.set_title(METHOD_LABELS[method], pad=-2.0, fontsize=6.4)

    for row, model in enumerate(models):
        mesh, model_data = all_data[model]
        for metric_i, metric in enumerate(["gap", "normal"]):
            r = 2 * row + metric_i
            for col, method in enumerate(METHODS):
                if row == 0 and metric_i == 0:
                    ax = fig.axes[col]
                else:
                    ax = fig.add_subplot(gs[r, col], projection="3d")
                    values = model_data[method][f"{metric}_face"]
                    miss = model_data[method]["miss_points"] if metric == "normal" else None
                    render_error_panel(ax, mesh, values, metric, miss_points=miss)
        y_gap = 0.925 - (2 * row + 0.46) * (0.89 / nrows)
        y_normal = 0.925 - (2 * row + 1.42) * (0.89 / nrows)
        fig.text(0.041, y_gap, f"{MODEL_LABELS[model]}\n$|g|$", ha="center", va="center", fontsize=6.4)
        fig.text(0.041, y_normal, "normal", ha="center", va="center", fontsize=6.4)

    fig.text(
        0.095,
        0.965,
        "All panels are drawn on the same original triangle surface; only face color changes.",
        ha="left",
        va="center",
        fontsize=6.3,
    )
    fig.text(
        0.895,
        0.965,
        "red dots: sharp-sector miss",
        ha="right",
        va="center",
        fontsize=6.0,
        color="#a60000",
    )
    add_colorbars(fig)
    out = PAPER / figure_name
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=500, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_summary(rows: list[dict]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "surface_error_map_summary.csv"
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}", flush=True)


def main() -> None:
    args = parse_args()
    PAPER.mkdir(exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    requested_sets = set(args.sets)
    models: list[str] = []
    if "main" in requested_sets:
        models.extend(MAIN_MODELS)
    if "supplemental" in requested_sets:
        models.extend(SUPPLEMENTAL_MODELS)

    all_data: dict[str, tuple[CornerNormalMesh, dict[str, dict[str, np.ndarray]]]] = {}
    rows: list[dict] = []
    for model in models:
        t0 = time.perf_counter()
        mesh, data, model_rows = model_error_data(model, args.grid_resolution, args.force)
        all_data[model] = (mesh, data)
        rows.extend(model_rows)
        print(f"[{model}] error maps ready in {time.perf_counter() - t0:.2f}s", flush=True)

    if rows:
        write_summary(rows)
    if "main" in requested_sets:
        render_figure("fig_surface_error_main", MAIN_MODELS, all_data)
        print("wrote paper/fig_surface_error_main.[pdf|svg|png]", flush=True)
    if "supplemental" in requested_sets:
        render_figure("fig_surface_error_supplemental", SUPPLEMENTAL_MODELS, all_data)
        print("wrote paper/fig_surface_error_supplemental.[pdf|svg|png]", flush=True)


if __name__ == "__main__":
    main()
