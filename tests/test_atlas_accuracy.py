import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from contact_sdf.shapes import ellipsoid_mesh, sample_near_surface
from contact_sdf.projection import MeshProjector, angular_error_deg
from contact_sdf.atlas import build_contact_sdf_atlas
from contact_sdf.grid_sdf import build_grid_sdf, build_cubic_grid_sdf
from contact_sdf.metrics import best_candidate_angle_deg


def test_atlas_has_reasonable_gap_and_normal_accuracy_on_smooth_shape():
    mesh = ellipsoid_mesh(n_lon=20, n_lat=10)
    bbox = mesh.bbox(pad=0.2)
    projector = MeshProjector(mesh, k=40)
    atlas = build_contact_sdf_atlas(projector, bbox, resolution=17, active_tol=0.02)
    q = sample_near_surface(mesh, n=200, band=0.05, seed=7)
    ref = projector.project(q, active_tol=0.01)
    ae = atlas.eval(q)
    gap_rmse = np.sqrt(np.mean((ae.phi - ref.phi) ** 2))
    ang = best_candidate_angle_deg(ae.candidate_normals, ref.normal)
    assert gap_rmse < 0.08
    assert np.percentile(ang, 95) < 25.0


def test_scalar_grid_gradient_is_available_baseline():
    mesh = ellipsoid_mesh(n_lon=16, n_lat=8)
    projector = MeshProjector(mesh, k=32)
    grid = build_grid_sdf(projector, mesh.bbox(pad=0.2), resolution=15)
    q = sample_near_surface(mesh, n=50, band=0.04, seed=8)
    phi, n = grid.eval(q)
    assert phi.shape == (50,)
    assert n.shape == (50, 3)
    assert np.all(np.isfinite(n))


def test_tricubic_grid_gradient_is_available_baseline():
    mesh = ellipsoid_mesh(n_lon=16, n_lat=8)
    projector = MeshProjector(mesh, k=32)
    grid = build_cubic_grid_sdf(projector, mesh.bbox(pad=0.2), resolution=15)
    q = sample_near_surface(mesh, n=50, band=0.04, seed=18)
    phi, n = grid.eval(q)
    assert phi.shape == (50,)
    assert n.shape == (50, 3)
    assert np.all(np.isfinite(phi))
    assert np.all(np.isfinite(n))


def test_uniform_compact_eval_matches_diagnostic_primary_fields():
    mesh = ellipsoid_mesh(n_lon=16, n_lat=8)
    bbox = mesh.bbox(pad=0.2)
    projector = MeshProjector(mesh, k=32)
    atlas = build_contact_sdf_atlas(projector, bbox, resolution=13, active_tol=0.02)
    q = sample_near_surface(mesh, n=80, band=0.04, seed=19)
    rich = atlas.eval(q)
    compact = atlas.eval_compact(q)
    assert np.allclose(compact.phi, rich.phi)
    assert np.allclose(compact.normal, rich.normal)
    assert np.array_equal(compact.mode, rich.mode)
    assert compact.leaf_id.shape == (80,)
    assert compact.candidate_count.shape == (80,)


def test_adaptive_atlas_refines_and_evaluates_without_online_projection():
    from contact_sdf.shapes import cone_mesh
    from contact_sdf.atlas import build_adaptive_contact_sdf_atlas, MODE_MULTI
    mesh = cone_mesh(n_seg=12)
    projector = MeshProjector(mesh, k=24)
    adaptive = build_adaptive_contact_sdf_atlas(
        projector, mesh.bbox(pad=0.2), base_resolution=5, max_depth=2,
        active_tol=0.03, refine_band=0.18, normal_tol_deg=15.0,
        gap_tol_factor=0.18, hessian_for_smooth=False,
    )
    q = sample_near_surface(mesh, n=80, band=0.035, seed=11)
    ref = projector.project(q, active_tol=0.01)
    ae = adaptive.eval(q)
    ang = best_candidate_angle_deg(ae.candidate_normals, ref.normal)
    assert adaptive.n_leaves > (8 - 1) ** 3
    assert np.percentile(ang, 95) < 70.0
    assert np.any(ae.mode == MODE_MULTI)
    assert np.all(np.isfinite(ae.phi))


def test_adaptive_compact_eval_matches_diagnostic_primary_fields():
    from contact_sdf.shapes import cone_mesh
    from contact_sdf.atlas import build_adaptive_contact_sdf_atlas

    mesh = cone_mesh(n_seg=12)
    projector = MeshProjector(mesh, k=24)
    adaptive = build_adaptive_contact_sdf_atlas(
        projector, mesh.bbox(pad=0.2), base_resolution=5, max_depth=2,
        active_tol=0.03, refine_band=0.18, normal_tol_deg=15.0,
        gap_tol_factor=0.18, hessian_for_smooth=False,
    )
    q = sample_near_surface(mesh, n=80, band=0.035, seed=21)
    rich = adaptive.eval(q)
    compact = adaptive.eval_compact(q)
    assert np.allclose(compact.phi, rich.phi)
    assert np.allclose(compact.normal, rich.normal)
    assert np.array_equal(compact.mode, rich.mode)
    assert compact.leaf_id.shape == (80,)
    assert compact.candidate_count.shape == (80,)


def test_smooth_adaptive_hessian_improves_ellipsoid_normals():
    from contact_sdf.atlas import build_adaptive_contact_sdf_atlas

    mesh = ellipsoid_mesh(n_lon=10, n_lat=5)
    bbox = mesh.bbox(pad=0.2)
    projector = MeshProjector(mesh, k=min(40, mesh.n_faces))
    q = sample_near_surface(mesh, n=60, band=0.05, seed=17)
    ref = projector.project(q, active_tol=0.01)
    common = dict(
        base_resolution=4,
        max_depth=1,
        feature_max_depth=1,
        active_tol=0.025,
        sector_angle_deg=35.0,
        gap_tol_factor=0.10,
        normal_tol_deg=10.0,
        feature_normal_tol_deg=5.0,
        feature_enrichment=False,
        multi_if_feature=False,
        refine_band=0.2,
    )

    no_hessian = build_adaptive_contact_sdf_atlas(projector, bbox, hessian_for_smooth=False, **common)
    with_hessian = build_adaptive_contact_sdf_atlas(projector, bbox, hessian_for_smooth=True, **common)
    no_hessian_eval = no_hessian.eval(q)
    with_hessian_eval = with_hessian.eval(q)

    no_hessian_ang = best_candidate_angle_deg(no_hessian_eval.candidate_normals, ref.normal)
    with_hessian_ang = best_candidate_angle_deg(with_hessian_eval.candidate_normals, ref.normal)
    assert float(np.mean(with_hessian_ang)) < 0.75 * float(np.mean(no_hessian_ang))
