"""Analytic benchmark shapes and corner-normal mesh generation."""
from __future__ import annotations
import numpy as np
from .mesh_format import CornerNormalMesh


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def ellipsoid_mesh(a: float = 1.4, b: float = 0.9, c: float = 0.65,
                   n_lon: int = 48, n_lat: int = 24) -> CornerNormalMesh:
    """Smooth ellipsoid with analytic corner normals.

    Axes are a,b,c.  Normals are exact normals of x^2/a^2+y^2/b^2+z^2/c^2=1.
    """
    verts = []
    normals = []
    # Include poles. phi in [0,pi]
    for i in range(n_lat + 1):
        phi = np.pi * i / n_lat
        sp, cp = np.sin(phi), np.cos(phi)
        for j in range(n_lon):
            th = 2 * np.pi * j / n_lon
            x = np.array([a * sp * np.cos(th), b * sp * np.sin(th), c * cp], dtype=float)
            n = _normalize(np.array([x[0] / (a * a), x[1] / (b * b), x[2] / (c * c)]))
            verts.append(x)
            normals.append(n)
    verts = np.asarray(verts)
    normals = np.asarray(normals)

    def vid(i, j):
        return i * n_lon + (j % n_lon)

    tris, ns = [], []
    for i in range(n_lat):
        for j in range(n_lon):
            if i == 0:
                ids = [vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)]
                tris.append(verts[ids]); ns.append(normals[ids])
            elif i == n_lat - 1:
                ids = [vid(i, j), vid(i + 1, j), vid(i, j + 1)]
                tris.append(verts[ids]); ns.append(normals[ids])
            else:
                ids1 = [vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)]
                ids2 = [vid(i, j), vid(i + 1, j + 1), vid(i, j + 1)]
                tris.append(verts[ids1]); ns.append(normals[ids1])
                tris.append(verts[ids2]); ns.append(normals[ids2])
    return CornerNormalMesh(np.asarray(tris), np.asarray(ns), name="ellipsoid",
                            tags={"type": "smooth", "axes": [a, b, c]})


def prism_mesh(n_sides: int = 6, radius: float = 1.0, height: float = 1.2) -> CornerNormalMesh:
    """Regular prism with intentionally discontinuous corner normals at edges."""
    h = height / 2.0
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    base = np.c_[radius * np.cos(angles), radius * np.sin(angles)]
    tris, ns = [], []

    # Side faces: split each rectangular side into two triangles.  Each side has a flat normal.
    for i in range(n_sides):
        j = (i + 1) % n_sides
        p0 = np.array([base[i, 0], base[i, 1], -h])
        p1 = np.array([base[j, 0], base[j, 1], -h])
        p2 = np.array([base[j, 0], base[j, 1], h])
        p3 = np.array([base[i, 0], base[i, 1], h])
        center_angle = (angles[i] + angles[j]) / 2.0
        # handle wrap around
        if j == 0:
            center_angle = (angles[i] + 2 * np.pi) / 2.0
        n = _normalize(np.array([np.cos(center_angle), np.sin(center_angle), 0.0]))
        for tri in ([p0, p1, p2], [p0, p2, p3]):
            tris.append(np.asarray(tri)); ns.append(np.tile(n, (3, 1)))

    # Top and bottom caps, fan triangulation.  Normals differ from side normals at shared positions.
    top_center = np.array([0.0, 0.0, h])
    bot_center = np.array([0.0, 0.0, -h])
    nt = np.array([0.0, 0.0, 1.0])
    nb = np.array([0.0, 0.0, -1.0])
    for i in range(n_sides):
        j = (i + 1) % n_sides
        pi = np.array([base[i, 0], base[i, 1], h])
        pj = np.array([base[j, 0], base[j, 1], h])
        tris.append(np.asarray([top_center, pi, pj])); ns.append(np.tile(nt, (3, 1)))
        pi_b = np.array([base[i, 0], base[i, 1], -h])
        pj_b = np.array([base[j, 0], base[j, 1], -h])
        # reversed winding for bottom
        tris.append(np.asarray([bot_center, pj_b, pi_b])); ns.append(np.tile(nb, (3, 1)))
    return CornerNormalMesh(np.asarray(tris), np.asarray(ns), name="hex_prism",
                            tags={"type": "sharp", "n_sides": n_sides, "radius": radius, "height": height})


def cone_mesh(radius: float = 1.0, height: float = 1.4, n_seg: int = 64) -> CornerNormalMesh:
    """Right circular cone with smooth side normals and sharp base rim/apex sectors."""
    # Base at z=0, apex at z=height, outward side normal grad of rho - R(1-z/h).
    angles = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
    base = np.c_[radius * np.cos(angles), radius * np.sin(angles), np.zeros(n_seg)]
    apex = np.array([0.0, 0.0, height])
    base_center = np.array([0.0, 0.0, 0.0])
    tris, ns = [], []
    for i in range(n_seg):
        j = (i + 1) % n_seg
        th_i, th_j = angles[i], angles[j]
        p_i, p_j = base[i], base[j]
        # Side triangle; corner normals at base/apex are sector-specific.
        n_i = _normalize(np.array([np.cos(th_i), np.sin(th_i), radius / height]))
        n_j = _normalize(np.array([np.cos(th_j), np.sin(th_j), radius / height]))
        th_m = 0.5 * (th_i + (th_j if j != 0 else 2 * np.pi))
        n_a = _normalize(np.array([np.cos(th_m), np.sin(th_m), radius / height]))
        tris.append(np.asarray([p_i, p_j, apex])); ns.append(np.asarray([n_i, n_j, n_a]))
        # Base cap; same geometric rim positions but different normals.
        nb = np.array([0.0, 0.0, -1.0])
        tris.append(np.asarray([base_center, p_j, p_i])); ns.append(np.tile(nb, (3, 1)))
    return CornerNormalMesh(np.asarray(tris), np.asarray(ns), name="cone",
                            tags={"type": "mixed", "radius": radius, "height": height})


# Analytic auxiliary functions for generating near-surface query points.
def sample_surface_points(mesh: CornerNormalMesh, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    tris = mesh.triangles
    ns = mesh.corner_normals
    area = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1) * 0.5
    prob = area / area.sum()
    fids = rng.choice(mesh.n_faces, size=n, p=prob)
    r1 = rng.random(n)
    r2 = rng.random(n)
    sqrt_r1 = np.sqrt(r1)
    w0 = 1 - sqrt_r1
    w1 = sqrt_r1 * (1 - r2)
    w2 = sqrt_r1 * r2
    w = np.c_[w0, w1, w2]
    p = (tris[fids] * w[:, :, None]).sum(axis=1)
    nrm = _normalize((ns[fids] * w[:, :, None]).sum(axis=1))
    return p, nrm


def sample_near_surface(mesh: CornerNormalMesh, n: int = 5000, band: float = 0.06,
                        seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p, nrm = sample_surface_points(mesh, n, rng)
    # Mixed inside/outside offsets: useful for signed gap tests.
    d = rng.uniform(-band, band, size=n)
    return p + d[:, None] * nrm
