"""Conventional scalar grid SDF with trilinear interpolation.

This is the baseline that illustrates the problem: the normal is derived from
interpolating scalar samples rather than from contact features/corner normals.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .projection import MeshProjector, normalize


@dataclass
class GridSDF:
    origin: np.ndarray
    spacing: float
    values: np.ndarray  # (nx,ny,nz) scalar node values

    def clamp_cell_indices(self, p: np.ndarray):
        u = (p - self.origin) / self.spacing
        idx = np.floor(u).astype(int)
        max_idx = np.array(self.values.shape) - 2
        idx = np.minimum(np.maximum(idx, 0), max_idx)
        f = u - idx
        f = np.minimum(np.maximum(f, 0.0), 1.0)
        return idx, f

    def eval(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(points, dtype=float)
        phi = np.empty(pts.shape[0])
        grad = np.empty_like(pts)
        h = self.spacing
        V = self.values
        for q, p in enumerate(pts):
            (i, j, k), (tx, ty, tz) = self.clamp_cell_indices(p)
            c000 = V[i, j, k]
            c100 = V[i+1, j, k]
            c010 = V[i, j+1, k]
            c110 = V[i+1, j+1, k]
            c001 = V[i, j, k+1]
            c101 = V[i+1, j, k+1]
            c011 = V[i, j+1, k+1]
            c111 = V[i+1, j+1, k+1]

            c00 = c000*(1-tx)+c100*tx
            c10 = c010*(1-tx)+c110*tx
            c01 = c001*(1-tx)+c101*tx
            c11 = c011*(1-tx)+c111*tx
            c0 = c00*(1-ty)+c10*ty
            c1 = c01*(1-ty)+c11*ty
            phi[q] = c0*(1-tz)+c1*tz

            # Analytic derivative of trilinear interpolant.
            dphidx = ((c100-c000)*(1-ty)*(1-tz) + (c110-c010)*ty*(1-tz) +
                      (c101-c001)*(1-ty)*tz + (c111-c011)*ty*tz) / h
            dphidy = ((c010-c000)*(1-tx)*(1-tz) + (c110-c100)*tx*(1-tz) +
                      (c011-c001)*(1-tx)*tz + (c111-c101)*tx*tz) / h
            dphidz = ((c001-c000)*(1-tx)*(1-ty) + (c101-c100)*tx*(1-ty) +
                      (c011-c010)*(1-tx)*ty + (c111-c110)*tx*ty) / h
            grad[q] = [dphidx, dphidy, dphidz]
        return phi, normalize(grad)


def build_grid_sdf(projector: MeshProjector, bbox: tuple[np.ndarray, np.ndarray],
                   resolution: int = 32, active_tol: float = 2e-3) -> GridSDF:
    lo, hi = bbox
    extent = hi - lo
    spacing = float(np.max(extent) / (resolution - 1))
    # Center the cubic grid around bbox.
    center = 0.5 * (lo + hi)
    half = 0.5 * spacing * (resolution - 1)
    origin = center - half
    axes = [origin[d] + spacing * np.arange(resolution) for d in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing='ij')
    pts = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    res = projector.project(pts, active_tol=active_tol)
    values = res.phi.reshape((resolution, resolution, resolution))
    return GridSDF(origin=origin, spacing=spacing, values=values)
