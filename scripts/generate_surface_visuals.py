from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contact_sdf.atlas import (  # noqa: E402
    AdaptiveContactSDFAtlas,
    ContactSDFAtlas,
    build_adaptive_contact_sdf_atlas,
    build_contact_sdf_atlas,
)
from contact_sdf.grid_sdf import CubicGridSDF, GridSDF, build_grid_sdf  # noqa: E402
from contact_sdf.mesh_format import CornerNormalMesh, weld_positions  # noqa: E402
from contact_sdf.projection import MeshProjector  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "surface_visuals"
CACHE = RESULTS / "cache"
PAPER = ROOT / "paper"

MODEL_ORDER = [
    "ellipsoid",
    "hex_prism",
    "cone",
    "sphere",
    "wedge",
    "cylinder",
    "torus",
]

METHOD_ORDER = [
    "original",
    "trilinear",
    "tricubic",
    "uniform_atlas",
    "feature_adaptive",
]

METHOD_LABELS = {
    "original": "Original",
    "trilinear": "Trilinear SDF",
    "tricubic": "Tricubic SDF",
    "uniform_atlas": "Uniform atlas",
    "feature_adaptive": "Feature-adaptive atlas",
}

COLORS = {
    "original": "#D8D9DD",
    "trilinear": "#666A70",
    "tricubic": "#9AA6B8",
    "uniform_atlas": "#2B8C84",
    "feature_adaptive": "#D75A45",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate independent model-method zero-level surface images."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODEL_ORDER,
        choices=MODEL_ORDER,
        help="Models to render.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=METHOD_ORDER,
        choices=METHOD_ORDER,
        help="Methods to render.",
    )
    parser.add_argument("--grid-resolution", type=int, default=17)
    parser.add_argument("--visual-resolution", type=int, default=54)
    parser.add_argument("--force", action="store_true", help="Rebuild cached SDF/atlas data.")
    return parser.parse_args()


def load_mesh(model: str) -> CornerNormalMesh:
    path = DATA / f"{model}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing mesh file: {path}")
    return CornerNormalMesh.load_npz(path)


def cubic_bbox(mesh: CornerNormalMesh, pad: float = 0.24) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = mesh.bbox(pad=pad)
    center = 0.5 * (lo + hi)
    half = 0.5 * float(np.max(hi - lo))
    return center - half, center + half


def load_or_build_grid(
    model: str,
    projector: MeshProjector,
    bbox: tuple[np.ndarray, np.ndarray],
    resolution: int,
    force: bool,
) -> GridSDF:
    path = CACHE / f"{model}_grid_res{resolution}.npz"
    if path.exists() and not force:
        z = np.load(path)
        return GridSDF(origin=z["origin"], spacing=float(z["spacing"]), values=z["values"])
    grid = build_grid_sdf(projector, bbox, resolution=resolution, active_tol=0.01)
    np.savez_compressed(path, origin=grid.origin, spacing=grid.spacing, values=grid.values)
    return grid


def load_or_build_uniform_atlas(
    model: str,
    projector: MeshProjector,
    bbox: tuple[np.ndarray, np.ndarray],
    resolution: int,
    force: bool,
) -> ContactSDFAtlas:
    path = CACHE / f"{model}_uniform_atlas_res{resolution}.npz"
    if path.exists() and not force:
        return ContactSDFAtlas.load_npz(path)
    atlas = build_contact_sdf_atlas(
        projector,
        bbox,
        resolution=resolution,
        active_tol=0.03,
        sector_angle_deg=35.0,
        max_candidates=16,
    )
    atlas.save_npz(path)
    return atlas


def load_or_build_adaptive_atlas(
    model: str,
    projector: MeshProjector,
    bbox: tuple[np.ndarray, np.ndarray],
    force: bool,
) -> AdaptiveContactSDFAtlas:
    path = CACHE / f"{model}_feature_adaptive_atlas_b5_s2_f3.npz"
    if path.exists() and not force:
        return AdaptiveContactSDFAtlas.load_npz(path)
    extent = bbox[1] - bbox[0]
    refine_band = 0.09 * float(np.max(extent))
    atlas = build_adaptive_contact_sdf_atlas(
        projector,
        bbox,
        base_resolution=5,
        max_depth=2,
        feature_max_depth=3,
        active_tol=0.01,
        sector_angle_deg=35.0,
        max_candidates=16,
        gap_tol_factor=0.08,
        normal_tol_deg=6.0,
        feature_normal_tol_deg=5.0,
        feature_enrichment=True,
        hessian_for_smooth=True,
        multi_if_feature=False,
        refine_band=refine_band,
    )
    atlas.save_npz(path)
    return atlas


def mesh_vertices_faces(mesh: CornerNormalMesh) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = weld_positions(mesh, tol=1e-8)
    return vertices, faces.astype(np.int64)


def sample_zero_surface(
    eval_phi,
    bbox: tuple[np.ndarray, np.ndarray],
    resolution: int,
    reference_projector: MeshProjector | None = None,
    clip_band: float | None = None,
    chunk: int = 70000,
) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = bbox
    axes = [np.linspace(lo[d], hi[d], resolution) for d in range(3)]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    pts = np.c_[x.ravel(), y.ravel(), z.ravel()]
    values = np.empty(pts.shape[0], dtype=float)
    for start in range(0, pts.shape[0], chunk):
        stop = min(start + chunk, pts.shape[0])
        values[start:stop] = eval_phi(pts[start:stop])
    vol = values.reshape((resolution, resolution, resolution))
    if not (np.nanmin(vol) <= 0.0 <= np.nanmax(vol)):
        raise ValueError("The sampled field does not cross zero in the visualization box.")
    spacing = tuple((hi - lo) / (resolution - 1))
    verts, faces, _normals, _values = marching_cubes(vol, level=0.0, spacing=spacing)
    verts += lo[None, :]
    faces = faces.astype(np.int64)
    if reference_projector is not None and clip_band is not None:
        verts, faces = clip_to_reference_band(verts, faces, reference_projector, clip_band, chunk=chunk)
    return verts, faces


def reference_phi(
    projector: MeshProjector,
    points: np.ndarray,
    chunk: int = 70000,
) -> np.ndarray:
    phi = np.empty(points.shape[0], dtype=float)
    for start in range(0, points.shape[0], chunk):
        stop = min(start + chunk, points.shape[0])
        phi[start:stop] = projector.project(points[start:stop], active_tol=0.01).phi
    return phi


def compact_mesh(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(faces) == 0:
        return vertices[:0], faces
    used, inverse = np.unique(faces.ravel(), return_inverse=True)
    return vertices[used], inverse.reshape(faces.shape)


def largest_face_component(faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return faces
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for fid, face in enumerate(faces):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = (int(a), int(b)) if a < b else (int(b), int(a))
            edge_to_faces.setdefault(key, []).append(fid)
    neighbors = [[] for _ in range(len(faces))]
    for ids in edge_to_faces.values():
        if len(ids) < 2:
            continue
        for i in ids:
            neighbors[i].extend(j for j in ids if j != i)

    visited = np.zeros(len(faces), dtype=bool)
    best: list[int] = []
    for start in range(len(faces)):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        comp = []
        while stack:
            fid = stack.pop()
            comp.append(fid)
            for nb in neighbors[fid]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        if len(comp) > len(best):
            best = comp
    return faces[np.asarray(best, dtype=np.int64)]


def clip_to_reference_band(
    vertices: np.ndarray,
    faces: np.ndarray,
    projector: MeshProjector,
    clip_band: float,
    chunk: int = 70000,
) -> tuple[np.ndarray, np.ndarray]:
    ref = np.abs(reference_phi(projector, vertices, chunk=chunk))
    near_face = np.max(ref[faces], axis=1) <= clip_band
    filtered = faces[near_face]
    if len(filtered) == 0:
        filtered = faces[np.argsort(np.mean(ref[faces], axis=1))[: max(1, len(faces) // 4)]]
    filtered = largest_face_component(filtered)
    return compact_mesh(vertices, filtered)


def atlas_visual_phi(
    atlas: ContactSDFAtlas | AdaptiveContactSDFAtlas,
    points: np.ndarray,
) -> np.ndarray:
    """Scalarize atlas records for visualization only.

    Solver queries keep the full candidate set.  A zero-level rendering needs a
    scalar field, so multi-sector leaves are visualized by the candidate gap
    plane with the smallest absolute value at each point.
    """
    ev = atlas.eval(points)
    phi = ev.phi.copy()
    for i, cand in enumerate(ev.candidate_phi):
        if len(cand):
            phi[i] = cand[int(np.argmin(np.abs(cand)))]
    return phi


def zero_gap_surface_from_mesh(
    mesh: CornerNormalMesh,
    eval_phi,
    search_radius: float,
    n_scan: int = 9,
    n_bisect: int = 18,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover a zero-gap surface by root-finding along input corner normals.

    This is used for atlas records, whose feature leaves represent contact
    constraints rather than one globally smooth scalar field.  The procedure is
    still generic: every triangle corner is moved only along its own supplied
    corner normal until the queried gap is closest to zero.
    """
    base = mesh.triangles.reshape(-1, 3)
    normals = mesh.corner_normals.reshape(-1, 3)
    ts = np.linspace(-search_radius, search_radius, n_scan)
    vals = np.empty((base.shape[0], n_scan), dtype=float)
    for j, t in enumerate(ts):
        vals[:, j] = eval_phi(base + t * normals)

    lo = np.full(base.shape[0], -search_radius, dtype=float)
    hi = np.full(base.shape[0], search_radius, dtype=float)
    bracketed = np.zeros(base.shape[0], dtype=bool)
    best_width = np.full(base.shape[0], np.inf, dtype=float)
    for j in range(n_scan - 1):
        v0 = vals[:, j]
        v1 = vals[:, j + 1]
        cross = (v0 == 0.0) | (v1 == 0.0) | (np.signbit(v0) != np.signbit(v1))
        width = np.maximum(np.abs(v0), np.abs(v1))
        take = cross & (width < best_width)
        lo[take] = ts[j]
        hi[take] = ts[j + 1]
        best_width[take] = width[take]
        bracketed[take] = True

    roots = np.empty(base.shape[0], dtype=float)
    if np.any(bracketed):
        l = lo[bracketed].copy()
        h = hi[bracketed].copy()
        p = base[bracketed]
        n = normals[bracketed]
        fl = eval_phi(p + l[:, None] * n)
        for _ in range(n_bisect):
            m = 0.5 * (l + h)
            fm = eval_phi(p + m[:, None] * n)
            same = np.signbit(fl) == np.signbit(fm)
            l[same] = m[same]
            fl[same] = fm[same]
            h[~same] = m[~same]
        roots[bracketed] = 0.5 * (l + h)

    if np.any(~bracketed):
        nearest = np.argmin(np.abs(vals[~bracketed]), axis=1)
        roots[~bracketed] = ts[nearest]

    vertices = base + roots[:, None] * normals
    faces = np.arange(vertices.shape[0], dtype=np.int64).reshape(-1, 3)
    return vertices, faces


def triangles_from_mesh(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return vertices[faces]


def shaded_facecolors(triangles: np.ndarray, base_hex: str) -> np.ndarray:
    base = np.asarray(to_rgb(base_hex), dtype=float)
    e1 = triangles[:, 1] - triangles[:, 0]
    e2 = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(e1, e2)
    nlen = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(nlen, 1e-12)
    light = np.asarray([-0.35, -0.45, 0.82], dtype=float)
    light /= np.linalg.norm(light)
    shade = 0.42 + 0.58 * np.clip(normals @ light, 0.0, 1.0)
    rgb = 1.0 - (1.0 - base[None, :]) * shade[:, None]
    return np.c_[rgb, np.full(rgb.shape[0], 1.0)]


def set_equal_axis(ax, vertices: np.ndarray) -> None:
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    center = 0.5 * (lo + hi)
    radius = 0.52 * float(np.max(hi - lo))
    if radius <= 0:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=20, azim=-52)


def render_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    method: str,
    out_stem: Path,
) -> None:
    triangles = triangles_from_mesh(vertices, faces)
    fig = plt.figure(figsize=(1.75, 1.75), dpi=400)
    ax = fig.add_subplot(111, projection="3d")
    facecolors = shaded_facecolors(triangles, COLORS[method])
    edge_alpha = 0.18 if method == "original" else 0.05
    collection = Poly3DCollection(
        triangles,
        facecolors=facecolors,
        edgecolors=(0.08, 0.08, 0.08, edge_alpha),
        linewidths=0.08 if method == "original" else 0.03,
        antialiased=True,
    )
    collection.set_rasterized(True)
    ax.add_collection3d(collection)
    set_equal_axis(ax, vertices)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_stem}.png", dpi=450, transparent=False, facecolor="white", pad_inches=0)
    fig.savefig(f"{out_stem}.pdf", transparent=False, facecolor="white", pad_inches=0)
    plt.close(fig)


def method_surface(
    method: str,
    mesh: CornerNormalMesh,
    model: str,
    projector: MeshProjector,
    bbox: tuple[np.ndarray, np.ndarray],
    grid_resolution: int,
    visual_resolution: int,
    force: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "original":
        return mesh_vertices_faces(mesh)

    grid = None
    if method in {"trilinear", "tricubic"}:
        grid = load_or_build_grid(model, projector, bbox, grid_resolution, force=force)
    clip_band = 1.35 * float(np.max(bbox[1] - bbox[0])) / float(grid_resolution - 1)

    if method == "trilinear":
        return sample_zero_surface(
            lambda pts: grid.eval(pts)[0],
            bbox,
            visual_resolution,
            reference_projector=projector,
            clip_band=clip_band,
        )
    if method == "tricubic":
        cubic = CubicGridSDF.from_grid(grid)
        return sample_zero_surface(
            lambda pts: cubic.eval(pts)[0],
            bbox,
            visual_resolution,
            reference_projector=projector,
            clip_band=clip_band,
        )
    if method == "uniform_atlas":
        atlas = load_or_build_uniform_atlas(model, projector, bbox, grid_resolution, force=force)
        return zero_gap_surface_from_mesh(mesh, lambda pts: atlas.eval_compact(pts).phi, 0.55 * clip_band)
    if method == "feature_adaptive":
        atlas = load_or_build_adaptive_atlas(model, projector, bbox, force=force)
        return zero_gap_surface_from_mesh(mesh, lambda pts: atlas.eval_compact(pts).phi, 0.55 * clip_band)
    raise ValueError(f"Unknown method: {method}")


def main() -> None:
    args = parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(exist_ok=True)
    total = len(args.models) * len(args.methods)
    done = 0
    for model in args.models:
        mesh = load_mesh(model)
        bbox = cubic_bbox(mesh)
        t0 = time.perf_counter()
        projector = MeshProjector(mesh, k=min(96, mesh.n_faces))
        print(f"[{model}] projector ready in {time.perf_counter() - t0:.2f}s", flush=True)
        for method in args.methods:
            done += 1
            out = PAPER / f"fig_surface_{model}_{method}"
            if out.with_suffix(".png").exists() and out.with_suffix(".pdf").exists() and not args.force:
                print(f"[{done}/{total}] keep {out.name}", flush=True)
                continue
            t1 = time.perf_counter()
            vertices, faces = method_surface(
                method,
                mesh,
                model,
                projector,
                bbox,
                args.grid_resolution,
                args.visual_resolution,
                args.force,
            )
            render_surface(vertices, faces, method, out)
            print(
                f"[{done}/{total}] wrote {out.name}.png/.pdf "
                f"({len(vertices)} vertices, {len(faces)} faces, {time.perf_counter() - t1:.1f}s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
