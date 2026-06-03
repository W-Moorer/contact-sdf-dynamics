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


def sphere_mesh(radius: float = 1.0, n_lon: int = 32, n_lat: int = 16) -> CornerNormalMesh:
    mesh = ellipsoid_mesh(a=radius, b=radius, c=radius, n_lon=n_lon, n_lat=n_lat)
    mesh.name = "sphere"
    mesh.tags.update({"type": "smooth", "primitive": "sphere", "radius": radius})
    return mesh


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


def cylinder_mesh(radius: float = 1.0, height: float = 1.2, n_seg: int = 32) -> CornerNormalMesh:
    """Capped cylinder with smooth side sectors and sharp cap rims."""
    h = 0.5 * height
    angles = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
    ring = np.c_[radius * np.cos(angles), radius * np.sin(angles)]
    tris, ns = [], []

    for i in range(n_seg):
        j = (i + 1) % n_seg
        p0 = np.array([ring[i, 0], ring[i, 1], -h])
        p1 = np.array([ring[j, 0], ring[j, 1], -h])
        p2 = np.array([ring[j, 0], ring[j, 1], h])
        p3 = np.array([ring[i, 0], ring[i, 1], h])
        n0 = _normalize(np.array([np.cos(angles[i]), np.sin(angles[i]), 0.0]))
        n1 = _normalize(np.array([np.cos(angles[j]), np.sin(angles[j]), 0.0]))
        tris.append(np.asarray([p0, p1, p2])); ns.append(np.asarray([n0, n1, n1]))
        tris.append(np.asarray([p0, p2, p3])); ns.append(np.asarray([n0, n1, n0]))

    top_center = np.array([0.0, 0.0, h])
    bot_center = np.array([0.0, 0.0, -h])
    nt = np.array([0.0, 0.0, 1.0])
    nb = np.array([0.0, 0.0, -1.0])
    for i in range(n_seg):
        j = (i + 1) % n_seg
        pi = np.array([ring[i, 0], ring[i, 1], h])
        pj = np.array([ring[j, 0], ring[j, 1], h])
        tris.append(np.asarray([top_center, pi, pj])); ns.append(np.tile(nt, (3, 1)))
        pi_b = np.array([ring[i, 0], ring[i, 1], -h])
        pj_b = np.array([ring[j, 0], ring[j, 1], -h])
        tris.append(np.asarray([bot_center, pj_b, pi_b])); ns.append(np.tile(nb, (3, 1)))
    return CornerNormalMesh(np.asarray(tris), np.asarray(ns), name="cylinder",
                            tags={"type": "mixed", "primitive": "cylinder",
                                  "radius": radius, "height": height, "n_seg": n_seg})


def wedge_mesh(angle_deg: float = 90.0, length: float = 1.4, height: float = 1.2) -> CornerNormalMesh:
    """Two finite planes meeting along a sharp dihedral edge."""
    alpha = np.radians(angle_deg)
    dirs = [np.array([1.0, 0.0, 0.0]), np.array([np.cos(alpha), np.sin(alpha), 0.0])]
    normals = [
        np.array([0.0, -1.0, 0.0]),
        np.array([np.sin(alpha), -np.cos(alpha), 0.0]),
    ]
    tris, ns = [], []
    z0, z1 = -0.5 * height, 0.5 * height
    edge0 = np.array([0.0, 0.0, z0])
    edge1 = np.array([0.0, 0.0, z1])
    for d, n in zip(dirs, normals):
        q0 = edge0 + length * d
        q1 = edge1 + length * d
        tris.append(np.asarray([edge0, q0, q1])); ns.append(np.tile(_normalize(n), (3, 1)))
        tris.append(np.asarray([edge0, q1, edge1])); ns.append(np.tile(_normalize(n), (3, 1)))
    return CornerNormalMesh(np.asarray(tris), np.asarray(ns), name="wedge",
                            tags={"type": "sharp", "primitive": "wedge",
                                  "angle_deg": angle_deg, "length": length, "height": height})


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


def torus_mesh(major_radius: float = 1.1, minor_radius: float = 0.32,
               n_major: int = 32, n_minor: int = 12) -> CornerNormalMesh:
    """Smooth torus with analytic corner normals."""
    verts = np.empty((n_major, n_minor, 3), dtype=float)
    normals = np.empty_like(verts)
    for i in range(n_major):
        u = 2 * np.pi * i / n_major
        cu, su = np.cos(u), np.sin(u)
        radial = np.array([cu, su, 0.0])
        center = major_radius * radial
        for j in range(n_minor):
            v = 2 * np.pi * j / n_minor
            n = _normalize(np.array([np.cos(v) * cu, np.cos(v) * su, np.sin(v)]))
            verts[i, j] = center + minor_radius * n
            normals[i, j] = n
    tris, ns = [], []
    for i in range(n_major):
        ip = (i + 1) % n_major
        for j in range(n_minor):
            jp = (j + 1) % n_minor
            ids1 = [(i, j), (ip, j), (ip, jp)]
            ids2 = [(i, j), (ip, jp), (i, jp)]
            tris.append(np.asarray([verts[a, b] for a, b in ids1]))
            ns.append(np.asarray([normals[a, b] for a, b in ids1]))
            tris.append(np.asarray([verts[a, b] for a, b in ids2]))
            ns.append(np.asarray([normals[a, b] for a, b in ids2]))
    return CornerNormalMesh(np.asarray(tris), np.asarray(ns), name="torus",
                            tags={"type": "smooth", "primitive": "torus",
                                  "major_radius": major_radius, "minor_radius": minor_radius})


def perturb_corner_normals(mesh: CornerNormalMesh, noise_deg: float,
                           seed: int = 0, name_suffix: str | None = None) -> CornerNormalMesh:
    """Return a copy with deterministic tangent-plane normal perturbations."""
    rng = np.random.default_rng(seed)
    n = np.asarray(mesh.corner_normals, dtype=float)
    v = rng.normal(size=n.shape)
    v -= np.sum(v * n, axis=-1, keepdims=True) * n
    v = _normalize(v)
    angle = np.radians(noise_deg)
    noisy = _normalize(np.cos(angle) * n + np.sin(angle) * v)
    tags = dict(mesh.tags)
    tags.update({"normal_noise_deg": float(noise_deg), "source_name": mesh.name})
    suffix = name_suffix if name_suffix is not None else f"noise_{noise_deg:g}deg"
    return CornerNormalMesh(mesh.triangles.copy(), noisy, name=f"{mesh.name}_{suffix}", tags=tags)


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
