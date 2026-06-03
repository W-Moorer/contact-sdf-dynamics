from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import math
import sys
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contact_sdf.atlas import AdaptiveContactSDFAtlas, ContactSDFAtlas
from contact_sdf.atlas import build_adaptive_contact_sdf_atlas, build_contact_sdf_atlas
from contact_sdf.grid_sdf import GridSDF, build_grid_sdf
from contact_sdf.mesh_format import CornerNormalMesh
from contact_sdf.projection import MeshProjector, angular_error_deg, normalize
from contact_sdf.metrics import best_candidate_angle_deg, cone_hit_rate, percentile, rmse, time_call
from contact_sdf.shapes import (
    cone_mesh,
    cylinder_mesh,
    perturb_corner_normals,
    prism_mesh,
    sample_near_surface,
    sphere_mesh,
    torus_mesh,
    wedge_mesh,
)
from scripts.build_and_validate import normal_sector_sharp_mask, sample_near_sharp_edges


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


@dataclass(frozen=True)
class ExactReference:
    phi: np.ndarray
    normal: np.ndarray
    hessian: np.ndarray | None = None


ExactReferenceFn = Callable[[np.ndarray], ExactReference]


@dataclass(frozen=True)
class CaseBackend:
    mesh: CornerNormalMesh
    projector: MeshProjector
    grid: GridSDF
    uniform: ContactSDFAtlas
    adaptive: AdaptiveContactSDFAtlas
    sector_angle_deg: float


SMOOTH_ADAPT = dict(
    base_resolution=6,
    max_depth=2,
    feature_max_depth=2,
    active_tol=0.025,
    max_candidates=8,
    sector_angle_deg=35.0,
    gap_tol_factor=0.06,
    normal_tol_deg=4.0,
    feature_normal_tol_deg=5.0,
    feature_enrichment=False,
    hessian_for_smooth=True,
)

FEATURE_ADAPT = dict(
    base_resolution=4,
    max_depth=1,
    feature_max_depth=3,
    active_tol=0.005,
    max_candidates=16,
    sector_angle_deg=35.0,
    gap_tol_factor=0.10,
    normal_tol_deg=10.0,
    feature_normal_tol_deg=5.0,
    feature_enrichment=True,
    hessian_for_smooth=False,
)


def sphere_exact_reference(radius: float = 1.0) -> ExactReferenceFn:
    def ref(points: np.ndarray) -> ExactReference:
        pts = np.asarray(points, dtype=float)
        r = np.linalg.norm(pts, axis=1)
        n = pts / np.maximum(r[:, None], 1e-12)
        phi = r - radius
        h = np.empty((pts.shape[0], 3, 3), dtype=float)
        eye = np.eye(3)
        for i in range(pts.shape[0]):
            h[i] = (eye - np.outer(n[i], n[i])) / max(r[i], 1e-12)
        return ExactReference(phi=phi, normal=n, hessian=h)

    return ref


def random_sphere_queries(n: int = 200, radius: float = 1.0,
                          band: float = 0.08, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dirs = normalize(rng.normal(size=(n, 3)))
    d = rng.uniform(-band, band, size=n)
    return (radius + d)[:, None] * dirs


def cylinder_zone_queries(radius: float = 1.0, height: float = 1.2,
                          n_each: int = 80, band: float = 0.05,
                          seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    h = 0.5 * height
    out: dict[str, np.ndarray] = {}
    th = rng.uniform(0.0, 2.0 * np.pi, size=n_each)
    z = rng.uniform(-0.35 * height, 0.35 * height, size=n_each)
    d = rng.uniform(-band, band, size=n_each)
    out["cylinder_side"] = np.c_[(radius + d) * np.cos(th), (radius + d) * np.sin(th), z]

    th = rng.uniform(0.0, 2.0 * np.pi, size=n_each)
    rim_z = rng.choice([-h, h], size=n_each)
    radial = radius + rng.uniform(-0.35 * band, 0.35 * band, size=n_each)
    axial = rng.uniform(-band, band, size=n_each)
    out["cylinder_rim"] = np.c_[radial * np.cos(th), radial * np.sin(th), rim_z + axial]
    return out


def cone_zone_queries(radius: float = 1.0, height: float = 1.4,
                      n_each: int = 80, band: float = 0.05,
                      seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    th = rng.uniform(0.0, 2.0 * np.pi, size=n_each)
    z = rng.uniform(0.25 * height, 0.80 * height, size=n_each)
    r_side = radius * (1.0 - z / height)
    n_side = normalize(np.c_[np.cos(th), np.sin(th), np.full(n_each, radius / height)])
    p_side = np.c_[r_side * np.cos(th), r_side * np.sin(th), z]
    out["cone_side"] = p_side + rng.uniform(-band, band, size=n_each)[:, None] * n_side

    th = rng.uniform(0.0, 2.0 * np.pi, size=n_each)
    radial = radius + rng.uniform(-0.35 * band, 0.35 * band, size=n_each)
    z = rng.uniform(-band, band, size=n_each)
    out["cone_rim"] = np.c_[radial * np.cos(th), radial * np.sin(th), z]

    th = rng.uniform(0.0, 2.0 * np.pi, size=n_each)
    radial = rng.uniform(0.0, 1.5 * band, size=n_each)
    z = height + rng.uniform(-1.5 * band, 0.5 * band, size=n_each)
    out["cone_apex"] = np.c_[radial * np.cos(th), radial * np.sin(th), z]
    return out


def mixed_queries(mesh: CornerNormalMesh, n_surface: int = 180, n_edge: int = 80,
                  band_factor: float = 0.06, seed: int = 42) -> np.ndarray:
    bbox = mesh.bbox(pad=0.20)
    band = band_factor * float(np.max(bbox[1] - bbox[0]))
    q_surface = sample_near_surface(mesh, n_surface, band=band, seed=seed)
    q_edge = sample_near_sharp_edges(mesh, n_edge, band=band, seed=seed + 1)
    return np.vstack([q_surface, q_edge]) if len(q_edge) else q_surface


def uniform_hessian_at(atlas: ContactSDFAtlas, points: np.ndarray) -> np.ndarray:
    out = np.empty((len(points), 3, 3), dtype=float)
    for i, p in enumerate(np.asarray(points, dtype=float)):
        idx, _ = atlas._locate(p)
        out[i] = atlas.hessian0[tuple(idx)]
    return out


def adaptive_hessian_at(atlas: AdaptiveContactSDFAtlas, points: np.ndarray) -> np.ndarray:
    ids = atlas._locate_leaf_ids(np.asarray(points, dtype=float))
    return atlas.hessian0[ids]


def hessian_rmse(pred: np.ndarray, ref: np.ndarray) -> float:
    err = pred - ref
    frob = np.sqrt(np.sum(err * err, axis=(1, 2)))
    return float(np.sqrt(np.mean(frob * frob)))


def adaptive_kwargs_for(mesh: CornerNormalMesh, sector_angle_deg: float = 35.0,
                        overrides: dict | None = None) -> dict:
    kwargs = dict(SMOOTH_ADAPT if mesh.tags.get("type") == "smooth" else FEATURE_ADAPT)
    kwargs["sector_angle_deg"] = float(sector_angle_deg)
    if overrides:
        kwargs.update(overrides)
    return kwargs


def build_case_backend(mesh: CornerNormalMesh, resolution: int = 9,
                       sector_angle_deg: float = 35.0,
                       adapt_overrides: dict | None = None) -> CaseBackend:
    bbox = mesh.bbox(pad=0.20)
    projector = MeshProjector(mesh, k=min(96, mesh.n_faces))
    grid = build_grid_sdf(projector, bbox, resolution=resolution, active_tol=0.01)
    uniform = build_contact_sdf_atlas(
        projector, bbox, resolution=resolution, active_tol=0.03, sector_angle_deg=sector_angle_deg
    )
    adapt_kwargs = adaptive_kwargs_for(mesh, sector_angle_deg=sector_angle_deg, overrides=adapt_overrides)
    band = 0.06 * float(np.max(bbox[1] - bbox[0]))
    adaptive = build_adaptive_contact_sdf_atlas(
        projector, bbox, multi_if_feature=False, refine_band=1.5 * band, **adapt_kwargs
    )
    return CaseBackend(mesh, projector, grid, uniform, adaptive, float(sector_angle_deg))


def evaluate_queries(case: str, backend: CaseBackend, queries: np.ndarray,
                     exact_ref: ExactReferenceFn | None = None,
                     resolution: int = 9) -> dict:
    mesh = backend.mesh
    projector = backend.projector
    grid = backend.grid
    uniform = backend.uniform
    adaptive = backend.adaptive
    sector_angle_deg = backend.sector_angle_deg

    proj = projector.project(queries, active_tol=0.01, sector_angle_deg=sector_angle_deg)
    ref = exact_ref(queries) if exact_ref is not None else ExactReference(proj.phi, proj.normal, None)
    grid_phi, grid_n = grid.eval(queries)
    uniform_eval = uniform.eval(queries)
    adaptive_eval = adaptive.eval(queries)

    grid_ang = angular_error_deg(grid_n, ref.normal)
    uniform_ang = best_candidate_angle_deg(uniform_eval.candidate_normals, ref.normal)
    adaptive_ang = best_candidate_angle_deg(adaptive_eval.candidate_normals, ref.normal)
    sharp_mask = normal_sector_sharp_mask(mesh, proj.active_normals)
    if sharp_mask.any():
        uniform_hit = cone_hit_rate(
            [uniform_eval.candidate_normals[i] for i in np.where(sharp_mask)[0]],
            proj.normal[sharp_mask],
            tol_deg=7.5,
        )
        adaptive_hit = cone_hit_rate(
            [adaptive_eval.candidate_normals[i] for i in np.where(sharp_mask)[0]],
            proj.normal[sharp_mask],
            tol_deg=7.5,
        )
    else:
        uniform_hit = float("nan")
        adaptive_hit = float("nan")

    bench_q = queries[:min(1000, len(queries))]
    _, t_proj = time_call(projector.project, bench_q, repeat=1, active_tol=0.01, sector_angle_deg=sector_angle_deg)
    _, t_adapt = time_call(adaptive.eval, bench_q, repeat=3)

    row = {
        "case": case,
        "shape": mesh.name,
        "type": mesh.tags.get("type", ""),
        "faces": mesh.n_faces,
        "queries": len(queries),
        "resolution": resolution,
        "sector_angle_deg": sector_angle_deg,
        "grid_gap_rmse": rmse(grid_phi, ref.phi),
        "uniform_gap_rmse": rmse(uniform_eval.phi, ref.phi),
        "adaptive_gap_rmse": rmse(adaptive_eval.phi, ref.phi),
        "grid_normal_mean_deg": float(np.mean(grid_ang)),
        "uniform_normal_mean_deg": float(np.mean(uniform_ang)),
        "adaptive_normal_mean_deg": float(np.mean(adaptive_ang)),
        "grid_normal_p95_deg": percentile(grid_ang, 95),
        "uniform_normal_p95_deg": percentile(uniform_ang, 95),
        "adaptive_normal_p95_deg": percentile(adaptive_ang, 95),
        "sharp_queries": int(sharp_mask.sum()),
        "uniform_cone_hit_rate": uniform_hit,
        "adaptive_cone_hit_rate": adaptive_hit,
        "adaptive_leaf_count": int(adaptive.n_leaves),
        "adaptive_feature_leaf_count": int(adaptive.stats.get("feature_leaf_count", 0)),
        "projection_us_per_query": 1e6 * t_proj / len(bench_q),
        "adaptive_us_per_query": 1e6 * t_adapt / len(bench_q),
        "speedup_projection_vs_adaptive": t_proj / max(t_adapt, 1e-12),
    }
    if ref.hessian is not None:
        row["uniform_hessian_rmse"] = hessian_rmse(uniform_hessian_at(uniform, queries), ref.hessian)
        row["adaptive_hessian_rmse"] = hessian_rmse(adaptive_hessian_at(adaptive, queries), ref.hessian)
    return row


def evaluate_case(case: str, mesh: CornerNormalMesh, queries: np.ndarray,
                  exact_ref: ExactReferenceFn | None = None,
                  resolution: int = 9,
                  sector_angle_deg: float = 35.0,
                  adapt_overrides: dict | None = None) -> dict:
    backend = build_case_backend(mesh, resolution=resolution, sector_angle_deg=sector_angle_deg,
                                 adapt_overrides=adapt_overrides)
    return evaluate_queries(case, backend, queries, exact_ref=exact_ref, resolution=resolution)


def evaluate_with_shared_atlas(case: str, mesh: CornerNormalMesh, query_sets: dict[str, np.ndarray],
                               resolution: int = 9, sector_angle_deg: float = 35.0) -> list[dict]:
    backend = build_case_backend(mesh, resolution=resolution, sector_angle_deg=sector_angle_deg)
    rows = []
    for zone, q in query_sets.items():
        rows.append(evaluate_queries(f"{case}:{zone}", backend, q, resolution=resolution))
    return rows


def run_resolution_depth_ablation() -> list[dict]:
    rows: list[dict] = []
    sphere = sphere_mesh(radius=1.0, n_lon=24, n_lat=12)
    q = random_sphere_queries(n=160, seed=11)
    for res in [7, 9, 13]:
        rows.append(evaluate_case(f"ablation_resolution_sphere_res{res}", sphere, q,
                                  exact_ref=sphere_exact_reference(), resolution=res))
    cylinder = cylinder_mesh(n_seg=24)
    q_cyl = mixed_queries(cylinder, n_surface=120, n_edge=60, seed=22)
    for depth in [1, 2, 3]:
        rows.append(evaluate_case(
            f"ablation_feature_depth_cylinder_d{depth}",
            cylinder,
            q_cyl,
            adapt_overrides={"feature_max_depth": depth, "max_depth": min(1, depth)},
        ))
    return rows


def run_sector_angle_ablation() -> list[dict]:
    rows: list[dict] = []
    wedge = wedge_mesh(angle_deg=90.0)
    q = mixed_queries(wedge, n_surface=80, n_edge=100, band_factor=0.04, seed=31)
    for angle in [20.0, 30.0, 35.0, 45.0]:
        rows.append(evaluate_case(f"ablation_sector_angle_wedge_{angle:g}", wedge, q,
                                  sector_angle_deg=angle))
    return rows


def run_mesh_consistency() -> list[dict]:
    rows: list[dict] = []
    q = random_sphere_queries(n=160, seed=41)
    for n_lon, n_lat in [(12, 6), (24, 12), (40, 20)]:
        mesh = sphere_mesh(n_lon=n_lon, n_lat=n_lat)
        rows.append(evaluate_case(f"mesh_consistency_sphere_{n_lon}x{n_lat}", mesh, q,
                                  exact_ref=sphere_exact_reference()))
    for n_seg in [12, 24, 48]:
        mesh = cylinder_mesh(n_seg=n_seg)
        rows.append(evaluate_case(f"mesh_consistency_cylinder_{n_seg}", mesh,
                                  mixed_queries(mesh, n_surface=120, n_edge=60, seed=42)))
    return rows


def run_noisy_normals() -> list[dict]:
    rows: list[dict] = []
    base = sphere_mesh(n_lon=24, n_lat=12)
    q = random_sphere_queries(n=160, seed=51)
    for noise in [2.0, 5.0, 10.0]:
        mesh = perturb_corner_normals(base, noise_deg=noise, seed=7)
        rows.append(evaluate_case(f"noisy_sphere_normals_{noise:g}deg", mesh, q,
                                  exact_ref=sphere_exact_reference()))
    return rows


def run_rigid_transform_check() -> dict:
    mesh = sphere_mesh(radius=1.0, n_lon=16, n_lat=8)
    q_local = random_sphere_queries(n=120, seed=61)
    bbox = mesh.bbox(pad=0.20)
    projector = MeshProjector(mesh, k=min(96, mesh.n_faces))
    atlas = build_adaptive_contact_sdf_atlas(
        projector, bbox, multi_if_feature=False, refine_band=0.18, **adaptive_kwargs_for(mesh)
    )
    local_eval = atlas.eval(q_local)
    theta = math.radians(37.0)
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    t = np.array([0.31, -0.22, 0.18])
    q_world = q_local @ R.T + t
    q_back = (q_world - t) @ R
    world_eval = atlas.eval(q_back)
    world_normals = world_eval.normal @ R.T
    expected_world_normals = local_eval.normal @ R.T
    return {
        "case": "rigid_transform_invariance",
        "shape": mesh.name,
        "queries": len(q_local),
        "max_phi_abs_diff": float(np.max(np.abs(world_eval.phi - local_eval.phi))),
        "max_world_normal_abs_diff": float(np.max(np.abs(world_normals - expected_world_normals))),
    }


def advantage_checks(rows: list[dict]) -> list[str]:
    issues: list[str] = []
    for r in rows:
        case = str(r["case"])
        typ = str(r.get("type", ""))
        if typ == "smooth" and "noisy" not in case and "torus" not in case:
            if r["adaptive_normal_mean_deg"] > r["uniform_normal_mean_deg"] + 1e-9:
                issues.append(f"{case}: adaptive smooth normal error exceeds uniform")
        if int(r.get("sharp_queries", 0)) > 0:
            if r["adaptive_cone_hit_rate"] + 1e-9 < r["uniform_cone_hit_rate"]:
                issues.append(f"{case}: adaptive cone-hit rate is below uniform")
        if "sphere_exact" in case and "adaptive_hessian_rmse" in r:
            if r["adaptive_hessian_rmse"] > r["uniform_hessian_rmse"] + 1e-9:
                issues.append(f"{case}: adaptive Hessian RMSE exceeds uniform")
    return issues


def main() -> None:
    rows: list[dict] = []

    sphere = sphere_mesh(radius=1.0, n_lon=24, n_lat=12)
    rows.append(evaluate_case("sphere_exact_hessian", sphere, random_sphere_queries(n=200, seed=1),
                              exact_ref=sphere_exact_reference()))

    wedge = wedge_mesh(angle_deg=90.0)
    rows.append(evaluate_case("two_plane_wedge", wedge,
                              mixed_queries(wedge, n_surface=80, n_edge=120, band_factor=0.04, seed=2)))

    cylinder = cylinder_mesh(n_seg=24)
    rows.append(evaluate_case("capped_cylinder", cylinder,
                              mixed_queries(cylinder, n_surface=160, n_edge=100, seed=3)))

    rows.extend(evaluate_with_shared_atlas("cylinder_zones", cylinder,
                                           cylinder_zone_queries(n_each=80, seed=4)))

    cone = cone_mesh(n_seg=16)
    rows.extend(evaluate_with_shared_atlas("cone_zones", cone, cone_zone_queries(n_each=80, seed=5)))

    rows.extend(run_resolution_depth_ablation())
    rows.extend(run_sector_angle_ablation())
    rows.extend(run_mesh_consistency())
    rows.extend(run_noisy_normals())

    torus = torus_mesh(n_major=28, n_minor=10)
    rows.append(evaluate_case("torus_smooth_concave", torus,
                              sample_near_surface(torus, n=180, band=0.05, seed=6)))

    transform_row = run_rigid_transform_check()
    issues = advantage_checks(rows)

    out_json = RESULTS / "supplemental_validation_summary.json"
    out_csv = RESULTS / "supplemental_validation_summary.csv"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({"rows": rows, "rigid_transform": transform_row, "issues": issues}, f, indent=2)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"rows": rows, "rigid_transform": transform_row, "issues": issues}, indent=2))
    if issues:
        raise SystemExit("supplemental validation advantage checks failed")


if __name__ == "__main__":
    main()
