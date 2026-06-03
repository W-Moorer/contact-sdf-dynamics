import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from contact_sdf.shapes import prism_mesh, cone_mesh, ellipsoid_mesh
from contact_sdf.mesh_format import weld_positions
from contact_sdf.projection import MeshProjector


def test_same_position_can_have_multiple_corner_normals_on_prism():
    mesh = prism_mesh(n_sides=6)
    verts, idx = weld_positions(mesh, tol=1e-9)
    # find a geometric vertex with incident normals that are not all equal
    found = False
    for v in range(len(verts)):
        loc = np.argwhere(idx == v)
        normals = []
        for f, li in loc:
            normals.append(mesh.corner_normals[f, li])
        if len(normals) >= 2:
            dots = np.array([[np.dot(a, b) for b in normals] for a in normals])
            if dots.min() < 0.95:
                found = True
                break
    assert found, "prism should preserve multiple corner normals at a shared geometric node"


def test_ellipsoid_corner_normals_are_unit():
    mesh = ellipsoid_mesh(n_lon=16, n_lat=8)
    assert np.allclose(np.linalg.norm(mesh.corner_normals, axis=-1), 1.0, atol=1e-10)


def test_cone_has_side_and_base_normal_sectors():
    mesh = cone_mesh(n_seg=16)
    verts, idx = weld_positions(mesh, tol=1e-9)
    found_rim = False
    for v in range(len(verts)):
        if abs(verts[v, 2]) < 1e-9 and np.linalg.norm(verts[v, :2]) > 0.5:
            loc = np.argwhere(idx == v)
            normals = [mesh.corner_normals[f, li] for f, li in loc]
            if len(normals) >= 2 and min(np.dot(a, b) for a in normals for b in normals) < 0.5:
                found_rim = True
                break
    assert found_rim


def test_projector_vertex_projection_exposes_welded_normal_sectors():
    mesh = cone_mesh(n_seg=16)
    projector = MeshProjector(mesh, k=min(32, mesh.n_faces))
    apex_z = float(mesh.tags["height"])
    res = projector.project(np.asarray([[0.0, 0.0, apex_z + 0.02]]),
                            active_tol=1e-4, sector_angle_deg=20.0)
    assert int(res.feature[0]) == 2
    assert len(res.active_normals[0]) >= 4
