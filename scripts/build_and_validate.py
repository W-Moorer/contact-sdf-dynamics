from __future__ import annotations
from pathlib import Path
import sys, csv, json
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contact_sdf.mesh_format import CornerNormalMesh, weld_positions
from contact_sdf.shapes import ellipsoid_mesh, prism_mesh, cone_mesh, sample_near_surface
from contact_sdf.projection import MeshProjector, angular_error_deg, normalize
from contact_sdf.grid_sdf import build_grid_sdf
from contact_sdf.atlas import build_contact_sdf_atlas, build_adaptive_contact_sdf_atlas, MODE_MULTI
from contact_sdf.metrics import rmse, percentile, best_candidate_angle_deg, cone_hit_rate, time_call

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


def edge_key(a, b):
    return (a, b) if a < b else (b, a)


def detect_sharp_edges(mesh: CornerNormalMesh, normal_jump_deg: float = 20.0):
    """Return geometric sharp edges with adjacent sector normals."""
    verts, idx = weld_positions(mesh, tol=1e-8)
    edge_map = {}
    # local edge definitions: (local a, local b)
    for f in range(mesh.n_faces):
        for la, lb in [(0, 1), (1, 2), (2, 0)]:
            ga, gb = int(idx[f, la]), int(idx[f, lb])
            k = edge_key(ga, gb)
            n_edge = normalize((mesh.corner_normals[f, la] + mesh.corner_normals[f, lb])[None, :])[0]
            edge_map.setdefault(k, []).append((f, n_edge))
    sharp = []
    cos_thr = np.cos(np.radians(normal_jump_deg))
    for (ga, gb), recs in edge_map.items():
        if len(recs) < 2:
            continue
        ns = [r[1] for r in recs]
        is_sharp = False
        for i in range(len(ns)):
            for j in range(i + 1, len(ns)):
                if float(np.dot(ns[i], ns[j])) < cos_thr:
                    is_sharp = True
        if is_sharp:
            # Unique sector normals.
            unique = []
            for n in ns:
                if not any(np.dot(n, u) > 0.985 for u in unique):
                    unique.append(n)
            sharp.append((verts[ga], verts[gb], np.asarray(unique)))
    return sharp


def sample_near_sharp_edges(mesh: CornerNormalMesh, n: int, band: float, seed: int = 1):
    rng = np.random.default_rng(seed)
    edges = detect_sharp_edges(mesh)
    if not edges:
        return np.empty((0, 3))
    lengths = np.array([np.linalg.norm(b - a) for a, b, _ in edges])
    prob = lengths / lengths.sum()
    ids = rng.choice(len(edges), size=n, p=prob)
    pts = []
    for eid in ids:
        a, b, ns = edges[eid]
        t = rng.random()
        p = (1 - t) * a + t * b
        # Pick a convex combination of adjacent sector normals to probe the normal cone zone.
        if ns.shape[0] == 1:
            n = ns[0]
        else:
            w = rng.random(ns.shape[0]); w /= w.sum()
            n = normalize((ns * w[:, None]).sum(axis=0, keepdims=True))[0]
        d = rng.uniform(-band, band)
        # small tangential jitter to avoid sampling exactly on the edge line
        tangent = normalize((b - a)[None, :])[0]
        jitter = rng.uniform(-0.25 * band, 0.25 * band) * tangent
        pts.append(p + d * n + jitter)
    return np.asarray(pts)


def build_shape_meshes():
    return [
        ellipsoid_mesh(n_lon=12, n_lat=6),
        prism_mesh(n_sides=6),
        cone_mesh(n_seg=16),
    ]


def normal_sector_sharp_mask(mesh: CornerNormalMesh, active_normals: list[np.ndarray]) -> np.ndarray:
    """Return queries with competing physical normal sectors.

    The sharp-feature metric is not a point-to-triangle region metric.  Smooth
    analytic benchmarks can project to a triangle edge or vertex without having
    a physical normal discontinuity.
    """
    mask = np.array([(len(a) > 1) for a in active_normals])
    if mesh.tags.get("type") == "smooth":
        mask[:] = False
    return mask


def validate_one(mesh: CornerNormalMesh, resolution: int = 9, n_surface: int = 180, n_edge: int = 80):
    print(f"\n=== {mesh.name}: faces={mesh.n_faces}, res={resolution} ===", flush=True)
    npz_path = DATA / f"{mesh.name}.npz"
    cnmesh_path = DATA / f"{mesh.name}.cnmesh"
    if not npz_path.exists():
        mesh.save_npz(npz_path)
    if not cnmesh_path.exists():
        mesh.save_rmd_like(cnmesh_path)
    bbox = mesh.bbox(pad=0.20)
    projector = MeshProjector(mesh, k=min(72, mesh.n_faces))

    band = 0.06 * max(*(bbox[1] - bbox[0]))
    q_surface = sample_near_surface(mesh, n_surface, band=band, seed=42)
    q_edge = sample_near_sharp_edges(mesh, n_edge, band=band, seed=43)
    queries = np.vstack([q_surface, q_edge]) if len(q_edge) else q_surface

    # Build offline structures.  The adaptive atlas starts from a coarser base
    # grid, then refines cells whose local jet/normal-cone record is not
    # accurate enough at sampled corners and face centers.
    grid = build_grid_sdf(projector, bbox, resolution=resolution, active_tol=0.01)
    atlas = build_contact_sdf_atlas(projector, bbox, resolution=resolution, active_tol=0.03)
    atlas.save_npz(RESULTS / f"{mesh.name}_uniform_atlas.npz")
    # Feature-specific refinement: smooth shapes use the ordinary adaptive depth;
    # sharp/mixed shapes spend extra depth only around normal-sector transitions.
    if mesh.name == "ellipsoid":
        adapt_kwargs = dict(base_resolution=5, max_depth=2, feature_max_depth=2,
                            active_tol=0.025, sector_angle_deg=35.0,
                            gap_tol_factor=0.08, normal_tol_deg=7.5,
                            feature_normal_tol_deg=5.0, feature_enrichment=False,
                            hessian_for_smooth=True)
    else:
        adapt_kwargs = dict(base_resolution=4, max_depth=1, feature_max_depth=3,
                            active_tol=0.005, sector_angle_deg=35.0,
                            gap_tol_factor=0.10, normal_tol_deg=10.0,
                            feature_normal_tol_deg=5.0, feature_enrichment=False,
                            hessian_for_smooth=False)
    adaptive = build_adaptive_contact_sdf_atlas(
        projector, bbox, multi_if_feature=False, refine_band=1.5 * band,
        **adapt_kwargs
    )
    adaptive.save_npz(RESULTS / f"{mesh.name}_feature_adaptive_atlas.npz")

    # Accuracy relative to online corner-normal projection baseline.
    proj = projector.project(queries, active_tol=0.01)
    grid_phi, grid_n = grid.eval(queries)
    atlas_eval = atlas.eval(queries)
    adaptive_eval = adaptive.eval(queries)

    grid_ang = angular_error_deg(grid_n, proj.normal)
    atlas_ang = best_candidate_angle_deg(atlas_eval.candidate_normals, proj.normal)
    adaptive_ang = best_candidate_angle_deg(adaptive_eval.candidate_normals, proj.normal)

    sharp_mask = normal_sector_sharp_mask(mesh, proj.active_normals)
    if sharp_mask.any():
        sharp_hit = cone_hit_rate([atlas_eval.candidate_normals[i] for i in np.where(sharp_mask)[0]],
                                  proj.normal[sharp_mask], tol_deg=7.5)
        adaptive_sharp_hit = cone_hit_rate([adaptive_eval.candidate_normals[i] for i in np.where(sharp_mask)[0]],
                                           proj.normal[sharp_mask], tol_deg=7.5)
    else:
        sharp_hit = float('nan')
        adaptive_sharp_hit = float('nan')

    # Timings on a subset.  Keep projection as online projection baseline.
    bench_q = queries[:min(1600, len(queries))]
    _, t_proj = time_call(projector.project, bench_q, repeat=2, active_tol=0.01)
    _, t_grid = time_call(grid.eval, bench_q, repeat=5)
    _, t_atlas = time_call(atlas.eval, bench_q, repeat=5)
    _, t_adaptive = time_call(adaptive.eval, bench_q, repeat=5)

    row = {
        "shape": mesh.name,
        "faces": mesh.n_faces,
        "queries": len(queries),
        "sharp_queries": int(sharp_mask.sum()),
        "resolution": resolution,
        "grid_gap_rmse": rmse(grid_phi, proj.phi),
        "atlas_gap_rmse": rmse(atlas_eval.phi, proj.phi),
        "adaptive_gap_rmse": rmse(adaptive_eval.phi, proj.phi),
        "grid_normal_mean_deg": float(np.mean(grid_ang)),
        "atlas_best_normal_mean_deg": float(np.mean(atlas_ang)),
        "adaptive_best_normal_mean_deg": float(np.mean(adaptive_ang)),
        "grid_normal_p95_deg": percentile(grid_ang, 95),
        "atlas_best_normal_p95_deg": percentile(atlas_ang, 95),
        "adaptive_best_normal_p95_deg": percentile(adaptive_ang, 95),
        "grid_grad_norm_mean_abs_err": float(np.mean(np.abs(np.linalg.norm(grid_n, axis=1) - 1.0))),
        "atlas_multi_mode_rate": float(np.mean(atlas_eval.mode == MODE_MULTI)),
        "adaptive_multi_mode_rate": float(np.mean(adaptive_eval.mode == MODE_MULTI)),
        "atlas_sharp_cone_hit_rate": sharp_hit,
        "adaptive_sharp_cone_hit_rate": adaptive_sharp_hit,
        "adaptive_leaf_count": int(adaptive.n_leaves),
        "adaptive_feature_leaf_count": int(adaptive.stats.get("feature_leaf_count", 0)),
        "adaptive_feature_split_count": int(adaptive.stats.get("feature_split_count", 0)),
        "adaptive_smooth_split_count": int(adaptive.stats.get("smooth_split_count", 0)),
        "adaptive_max_depth_leaf_count": int(adaptive.stats.get("max_depth_leaf_count", 0)),
        "adaptive_smooth_max_depth": int(adaptive.stats.get("smooth_max_depth", 0)),
        "adaptive_feature_max_depth": int(adaptive.stats.get("feature_max_depth", 0)),
        "adaptive_compression_vs_finest_uniform": float(adaptive.stats.get("compression_vs_finest_uniform", 0.0)),
        "projection_us_per_query": 1e6 * t_proj / len(bench_q),
        "grid_us_per_query": 1e6 * t_grid / len(bench_q),
        "atlas_us_per_query": 1e6 * t_atlas / len(bench_q),
        "adaptive_us_per_query": 1e6 * t_adaptive / len(bench_q),
        "speedup_projection_vs_atlas": t_proj / max(t_atlas, 1e-12),
        "speedup_projection_vs_adaptive": t_proj / max(t_adaptive, 1e-12),
        "speedup_projection_vs_grid": t_proj / max(t_grid, 1e-12),
    }
    print(json.dumps(row, indent=2), flush=True)
    return row


def main():
    rows = []
    for mesh in build_shape_meshes():
        rows.append(validate_one(mesh))
    out_csv = RESULTS / "validation_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    with (RESULTS / "validation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {out_csv}")

if __name__ == "__main__":
    main()
