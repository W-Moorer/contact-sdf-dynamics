from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contact_sdf.mesh_format import CornerNormalMesh  # noqa: E402
from contact_sdf.projection import MeshProjector  # noqa: E402
from scripts.generate_rmd_surface_visuals import (  # noqa: E402
    DEFAULT_OUT as RMD_VISUAL_OUT,
    NamedMesh,
    load_mesh_file,
    surface_for_method,
)
from scripts.generate_surface_visuals import COLORS, cubic_bbox, shaded_facecolors, triangles_from_mesh  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
FIGURES = PAPER / "figures" / "08_numerical_validation"
RESULTS = ROOT / "results" / "rmd_representatives"


@dataclass(frozen=True)
class Representative:
    key: str
    label: str
    mesh_path: Path
    elev: float
    azim: float
    zoom: float = 0.56
    edge_alpha: float = 0.08
    rotate_axis: str | None = None
    rotate_deg: float = 0.0


REPRESENTATIVES = [
    Representative(
        key="ellipsoid",
        label="smooth ellipsoid",
        mesh_path=ROOT / "results" / "rmd_extracted" / "groove_sphere_8_SOLID__Body2.Ellipsoid1.npz",
        elev=22,
        azim=-42,
        zoom=0.58,
        edge_alpha=0.07,
    ),
    Representative(
        key="hollow_cylinder",
        label="hollow cylinder",
        mesh_path=ROOT / "results" / "rmd_extracted" / "rev_clearance_joint_3_GSURFACE__Body1.Subtract1.npz",
        elev=18,
        azim=-58,
        zoom=0.56,
        edge_alpha=0.075,
    ),
    Representative(
        key="gear",
        label="gear",
        mesh_path=ROOT / "results" / "rmd_extracted" / "jiandanjiaolian_2_GSURFACE__GEAR21.GEAR21.npz",
        elev=16,
        azim=-58,
        zoom=0.36,
        edge_alpha=0.065,
    ),
    Representative(
        key="csg_solid",
        label="grooved solid",
        mesh_path=ROOT / "results" / "rmd_extracted" / "groove_cube_2_SOLID__Body1.Subtract2.npz",
        elev=24,
        azim=-42,
        zoom=0.57,
        edge_alpha=0.060,
        rotate_axis="x",
        rotate_deg=90.0,
    ),
    Representative(
        key="box",
        label="sharp box",
        mesh_path=ROOT / "results" / "rmd_extracted" / "groove_cube_8_SOLID__Body2.Box1.npz",
        elev=24,
        azim=-42,
        zoom=0.74,
        edge_alpha=0.055,
    ),
]

METHODS = ["original", "feature_adaptive"]
METHOD_LABELS = {
    "original": "original mesh",
    "feature_adaptive": "adaptive zero level",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the five selected RMD representative models for the manuscript.")
    parser.add_argument("--out-name", default="fig_rmd_representatives")
    parser.add_argument("--force", action="store_true", help="Rebuild adaptive atlas caches and rerender panels.")
    parser.add_argument("--write-panels", action="store_true", help="Also write one independent PNG/PDF per model/method.")
    parser.add_argument("--base-resolution", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--feature-max-depth", type=int, default=3)
    parser.add_argument("--feature-active-tol-factor", type=float, default=0.5)
    parser.add_argument("--anchor-tol-factor", type=float, default=0.25)
    parser.add_argument("--probe-resolution", type=int, default=17)
    parser.add_argument("--projector-k", type=int, default=96)
    return parser.parse_args()


def rmd_surface_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        out_dir=RMD_VISUAL_OUT,
        base_resolution=args.base_resolution,
        max_depth=args.max_depth,
        feature_max_depth=args.feature_max_depth,
        feature_active_tol_factor=args.feature_active_tol_factor,
        anchor_tol_factor=args.anchor_tol_factor,
        probe_resolution=args.probe_resolution,
        projector_k=args.projector_k,
        force=args.force,
    )


def set_panel_axis(ax, vertices: np.ndarray, rep: Representative) -> None:
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    center = 0.5 * (lo + hi)
    radius = rep.zoom * float(np.max(hi - lo))
    if radius <= 0:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=rep.elev, azim=rep.azim)
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass


def rotation_matrix(axis: str, deg: float) -> np.ndarray:
    theta = np.radians(float(deg))
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    axis = axis.lower()
    if axis == "x":
        return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == "y":
        return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    if axis == "z":
        return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    raise ValueError(f"Unsupported rotation axis: {axis}")


def display_vertices(vertices: np.ndarray, rep: Representative) -> np.ndarray:
    if rep.rotate_axis is None or abs(float(rep.rotate_deg)) <= 1e-12:
        return vertices
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    matrix = rotation_matrix(rep.rotate_axis, rep.rotate_deg)
    return (vertices - center) @ matrix.T + center


def render_panel(ax, vertices: np.ndarray, faces: np.ndarray, method: str, rep: Representative) -> None:
    vertices = display_vertices(vertices, rep)
    triangles = triangles_from_mesh(vertices, faces)
    facecolors = shaded_facecolors(triangles, COLORS[method])
    edge_alpha = rep.edge_alpha if method == "original" else min(rep.edge_alpha, 0.035)
    collection = Poly3DCollection(
        triangles,
        facecolors=facecolors,
        edgecolors=(0.07, 0.07, 0.07, edge_alpha),
        linewidths=0.055 if method == "original" else 0.025,
        antialiased=True,
    )
    collection.set_rasterized(True)
    ax.add_collection3d(collection)
    set_panel_axis(ax, vertices, rep)


def load_surfaces(args: argparse.Namespace) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    backend_args = rmd_surface_args(args)
    out: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    total = len(REPRESENTATIVES) * len(METHODS)
    done = 0
    for rep in REPRESENTATIVES:
        named: NamedMesh = load_mesh_file(rep.mesh_path)
        bbox = cubic_bbox(named.mesh)
        t0 = time.perf_counter()
        projector = MeshProjector(named.mesh, k=min(args.projector_k, named.mesh.n_faces))
        print(f"[{rep.key}] {named.mesh.n_faces} faces, projector {time.perf_counter() - t0:.2f}s", flush=True)
        for method in METHODS:
            done += 1
            t1 = time.perf_counter()
            vertices, faces = surface_for_method(method, backend_args, named, projector, bbox)
            out[(rep.key, method)] = (vertices, faces)
            print(
                f"[{done}/{total}] {rep.key}/{method}: "
                f"{len(vertices)} vertices, {len(faces)} faces, {time.perf_counter() - t1:.1f}s",
                flush=True,
            )
    return out


def write_independent_panels(
    surfaces: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
) -> None:
    panel_dir = RESULTS / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    for rep in REPRESENTATIVES:
        for method in METHODS:
            fig = plt.figure(figsize=(2.0, 2.0), dpi=400)
            ax = fig.add_subplot(111, projection="3d")
            vertices, faces = surfaces[(rep.key, method)]
            render_panel(ax, vertices, faces, method, rep)
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            out = panel_dir / f"rmd_representative_{rep.key}_{method}"
            fig.savefig(out.with_suffix(".pdf"), facecolor="white", pad_inches=0)
            fig.savefig(out.with_suffix(".png"), dpi=500, facecolor="white", pad_inches=0)
            plt.close(fig)


def render_composite(
    surfaces: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.25, 3.18), dpi=400)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=5,
        left=0.065,
        right=0.992,
        bottom=0.045,
        top=0.88,
        wspace=-0.08,
        hspace=0.02,
    )
    for col, rep in enumerate(REPRESENTATIVES):
        for row, method in enumerate(METHODS):
            ax = fig.add_subplot(gs[row, col], projection="3d")
            vertices, faces = surfaces[(rep.key, method)]
            render_panel(ax, vertices, faces, method, rep)
            if row == 0:
                ax.set_title(rep.label, fontsize=7.0, pad=-1.0)
    fig.text(0.025, 0.645, METHOD_LABELS["original"], rotation=90, ha="center", va="center", fontsize=7.0)
    fig.text(0.025, 0.245, METHOD_LABELS["feature_adaptive"], rotation=90, ha="center", va="center", fontsize=7.0)
    fig.text(
        0.065,
        0.965,
        "Five category-balanced RecurDyn RMD representatives; each column is scaled to its own bounding box.",
        ha="left",
        va="center",
        fontsize=6.5,
    )
    out = FIGURES / args.out_name
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".png"), dpi=550, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out.with_suffix('.pdf')} / .svg / .png", flush=True)


def main() -> None:
    args = parse_args()
    surfaces = load_surfaces(args)
    if args.write_panels:
        write_independent_panels(surfaces, args)
    render_composite(surfaces, args)


if __name__ == "__main__":
    main()
