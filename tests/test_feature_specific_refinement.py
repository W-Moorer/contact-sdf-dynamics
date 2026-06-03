import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from contact_sdf.shapes import prism_mesh, sample_near_surface
from contact_sdf.projection import MeshProjector
from contact_sdf.atlas import build_adaptive_contact_sdf_atlas, MODE_MULTI
from contact_sdf.metrics import best_candidate_angle_deg, cone_hit_rate
from contact_sdf.mesh_format import weld_positions
from scripts.build_and_validate import normal_sector_sharp_mask


def _sample_near_geometric_edges(mesh, n=60, band=0.04, seed=123):
    # Lightweight local sampler for this test: choose welded mesh edges whose
    # incident corner normals are discontinuous, then sample around them.
    rng = np.random.default_rng(seed)
    verts, idx = weld_positions(mesh, tol=1e-8)
    edge_map = {}
    for f in range(mesh.n_faces):
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            key = tuple(sorted((int(idx[f, a]), int(idx[f, b]))))
            nn = mesh.corner_normals[f, a] + mesh.corner_normals[f, b]
            nn = nn / max(np.linalg.norm(nn), 1e-12)
            edge_map.setdefault(key, []).append(nn)
    sharp = []
    for key, ns in edge_map.items():
        if len(ns) < 2:
            continue
        if min(float(np.dot(ns[i], ns[j])) for i in range(len(ns)) for j in range(i+1, len(ns))) < np.cos(np.radians(30)):
            sharp.append(key)
    assert sharp
    pts = []
    for _ in range(n):
        a_id, b_id = sharp[rng.integers(0, len(sharp))]
        a, b = verts[a_id], verts[b_id]
        t = rng.random()
        p = (1-t)*a + t*b
        tangent = b - a
        tangent = tangent / max(np.linalg.norm(tangent), 1e-12)
        # random direction not parallel to tangent
        v = rng.normal(size=3)
        v -= np.dot(v, tangent) * tangent
        v = v / max(np.linalg.norm(v), 1e-12)
        pts.append(p + rng.uniform(-band, band) * v)
    return np.asarray(pts)


def test_feature_specific_refinement_reaches_deeper_feature_leaves():
    mesh = prism_mesh(n_sides=6)
    projector = MeshProjector(mesh, k=24)
    atlas = build_adaptive_contact_sdf_atlas(
        projector, mesh.bbox(pad=0.2),
        base_resolution=4,
        max_depth=1,
        feature_max_depth=3,
        active_tol=0.005,
        multi_if_feature=False,
        refine_band=0.18,
        gap_tol_factor=0.10,
        normal_tol_deg=10.0,
        feature_normal_tol_deg=5.0,
        hessian_for_smooth=False,
        feature_enrichment=False,
        sector_angle_deg=35.0,
    )
    assert atlas.stats["smooth_max_depth"] == 1
    assert atlas.stats["feature_max_depth"] == 3
    assert atlas.stats["feature_split_count"] > 0
    assert atlas.stats["feature_leaf_count"] > 0
    assert np.any(atlas.leaf_level > atlas.stats["smooth_max_depth"])

    q = _sample_near_geometric_edges(mesh, n=50, band=0.035)
    ref = projector.project(q, active_tol=0.005, sector_angle_deg=35.0)
    ev = atlas.eval(q)
    sharp_mask = np.array([(len(a) > 1) for a in ref.active_normals])
    if sharp_mask.any():
        hit = cone_hit_rate([ev.candidate_normals[i] for i in np.where(sharp_mask)[0]],
                            ref.normal[sharp_mask], tol_deg=10.0)
        assert hit > 0.70
    assert np.all(np.isfinite(ev.phi))


def test_sharp_metric_excludes_smooth_benchmark_sector_artifacts():
    from contact_sdf.shapes import ellipsoid_mesh
    mesh = ellipsoid_mesh(n_lon=8, n_lat=4)
    active = [
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.asarray([[0.0, 0.0, 1.0]]),
    ]
    mask = normal_sector_sharp_mask(mesh, active)
    assert not np.any(mask)


def test_sharp_metric_flags_competing_sectors_on_sharp_mesh():
    mesh = prism_mesh(n_sides=6)
    active = [
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.asarray([[0.0, 0.0, 1.0]]),
    ]
    mask = normal_sector_sharp_mask(mesh, active)
    assert mask.tolist() == [True, False]
