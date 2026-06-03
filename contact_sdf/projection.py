"""Online closest-triangle projection baseline for corner-normal meshes.

This is intentionally kept as the expensive reference.  The atlas moves this
work offline and replaces it by cell lookup + polynomial evaluation.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from .mesh_format import CornerNormalMesh


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def closest_point_on_triangle_batch(p: np.ndarray, tri: np.ndarray):
    """Closest points from one point p to many triangles.

    Returns closest points, squared distances, barycentric coordinates and a
    feature code: 0 face interior, 1 edge, 2 vertex.
    Implementation follows the region tests from Real-Time Collision Detection.
    """
    p = np.asarray(p, dtype=float)
    a = tri[:, 0]
    b = tri[:, 1]
    c = tri[:, 2]
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = np.einsum('ij,ij->i', ab, ap)
    d2 = np.einsum('ij,ij->i', ac, ap)
    ntri = tri.shape[0]
    cp = np.empty((ntri, 3), dtype=float)
    bary = np.empty((ntri, 3), dtype=float)
    feature = np.zeros(ntri, dtype=np.int8)

    # Start with face-region projection for all, overwrite special regions.
    bp = p - b
    d3 = np.einsum('ij,ij->i', ab, bp)
    d4 = np.einsum('ij,ij->i', ac, bp)
    cp_p = p - c
    d5 = np.einsum('ij,ij->i', ab, cp_p)
    d6 = np.einsum('ij,ij->i', ac, cp_p)

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    denom = va + vb + vc
    v = vb / np.where(np.abs(denom) < 1e-18, 1.0, denom)
    w = vc / np.where(np.abs(denom) < 1e-18, 1.0, denom)
    u = 1.0 - v - w
    cp[:] = u[:, None] * a + v[:, None] * b + w[:, None] * c
    bary[:] = np.c_[u, v, w]
    feature[:] = 0

    # Vertex A
    mask = (d1 <= 0) & (d2 <= 0)
    cp[mask] = a[mask]; bary[mask] = np.array([1.0, 0.0, 0.0]); feature[mask] = 2
    # Vertex B
    mask = (d3 >= 0) & (d4 <= d3)
    cp[mask] = b[mask]; bary[mask] = np.array([0.0, 1.0, 0.0]); feature[mask] = 2
    # Edge AB
    mask = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    vv = d1 / np.maximum(d1 - d3, 1e-18)
    cp[mask] = a[mask] + vv[mask, None] * ab[mask]
    bary[mask] = np.c_[1 - vv[mask], vv[mask], np.zeros(mask.sum())]
    feature[mask] = 1
    # Vertex C
    mask = (d6 >= 0) & (d5 <= d6)
    cp[mask] = c[mask]; bary[mask] = np.array([0.0, 0.0, 1.0]); feature[mask] = 2
    # Edge AC
    mask = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    ww = d2 / np.maximum(d2 - d6, 1e-18)
    cp[mask] = a[mask] + ww[mask, None] * ac[mask]
    bary[mask] = np.c_[1 - ww[mask], np.zeros(mask.sum()), ww[mask]]
    feature[mask] = 1
    # Edge BC
    mask = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    ww = (d4 - d3) / np.maximum((d4 - d3) + (d5 - d6), 1e-18)
    cp[mask] = b[mask] + ww[mask, None] * (c[mask] - b[mask])
    bary[mask] = np.c_[np.zeros(mask.sum()), 1 - ww[mask], ww[mask]]
    feature[mask] = 1

    sq = np.einsum('ij,ij->i', p - cp, p - cp)
    return cp, sq, bary, feature


@dataclass
class ProjectionResult:
    phi: np.ndarray
    closest: np.ndarray
    normal: np.ndarray
    face_id: np.ndarray
    bary: np.ndarray
    feature: np.ndarray
    active_normals: list[np.ndarray]


class MeshProjector:
    """KD-tree accelerated triangle projection baseline."""

    def __init__(self, mesh: CornerNormalMesh, k: int = 48):
        self.mesh = mesh
        self.k = int(k)
        self.centroids = mesh.triangles.mean(axis=1)
        self.tree = cKDTree(self.centroids)
        self.face_normals = mesh.face_normals()

    def _project_with_ids(self, x: np.ndarray, ids: np.ndarray, active_tol: float, sector_angle_deg: float) -> tuple:
        ids = np.atleast_1d(ids)
        tri = self.mesh.triangles[ids]
        cp, sq, bary, feature = closest_point_on_triangle_batch(x, tri)
        j = int(np.argmin(sq))
        fid = int(ids[j])
        p = cp[j]
        bc = bary[j]
        # Interpolate corner normals; for sharp features this is one sector normal only.
        n = normalize((self.mesh.corner_normals[fid] * bc[:, None]).sum(axis=0, keepdims=True))[0]
        # Signed local gap: sign from selected sector normal. Good for narrow-band contact.
        phi = float(np.dot(x - p, n))
        # Build small active normal set among candidates with nearly equal distance or edge/vertex hit.
        dmin = float(np.sqrt(max(sq[j], 0.0)))
        active = []
        sector_cos = float(np.cos(np.radians(sector_angle_deg)))
        for loc, q2 in enumerate(sq):
            if np.sqrt(max(q2, 0.0)) <= dmin + active_tol:
                f2 = int(ids[loc])
                nn = normalize((self.mesh.corner_normals[f2] * bary[loc, :, None]).sum(axis=0, keepdims=True))[0]
                # Keep only genuinely different normal sectors. Smooth-chart
                # variations are not a normal cone.
                if not any(np.dot(nn, aa) > sector_cos for aa in active):
                    active.append(nn)
        if not active:
            active = [n]
        return phi, p, n, fid, bc, int(feature[j]), np.asarray(active)

    def project_one(self, x: np.ndarray, active_tol: float = 2e-3, sector_angle_deg: float = 20.0) -> tuple:
        k = min(self.k, self.mesh.n_faces)
        _, ids = self.tree.query(x, k=k)
        return self._project_with_ids(x, ids, active_tol, sector_angle_deg)

    def project(self, points: np.ndarray, active_tol: float = 2e-3, sector_angle_deg: float = 20.0) -> ProjectionResult:
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts[None, :]
        phi = np.empty(pts.shape[0])
        closest = np.empty_like(pts)
        normal = np.empty_like(pts)
        face_id = np.empty(pts.shape[0], dtype=np.int64)
        bary = np.empty((pts.shape[0], 3))
        feature = np.empty(pts.shape[0], dtype=np.int8)
        active_normals = []
        k = min(self.k, self.mesh.n_faces)
        _, all_ids = self.tree.query(pts, k=k)
        if k == 1:
            all_ids = all_ids[:, None]
        for i, x in enumerate(pts):
            phi[i], closest[i], normal[i], face_id[i], bary[i], feature[i], an = self._project_with_ids(
                x, all_ids[i], active_tol=active_tol, sector_angle_deg=sector_angle_deg
            )
            active_normals.append(an)
        return ProjectionResult(phi, closest, normal, face_id, bary, feature, active_normals)


def angular_error_deg(n_pred: np.ndarray, n_ref: np.ndarray) -> np.ndarray:
    n_pred = normalize(n_pred)
    n_ref = normalize(n_ref)
    c = np.clip(np.einsum('ij,ij->i', n_pred, n_ref), -1.0, 1.0)
    return np.degrees(np.arccos(c))
