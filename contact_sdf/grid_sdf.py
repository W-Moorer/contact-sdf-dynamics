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
        h = self.spacing
        V = self.values
        u = (pts - self.origin[None, :]) / h
        idx = np.floor(u).astype(int)
        max_idx = np.array(V.shape) - 2
        idx = np.minimum(np.maximum(idx, 0), max_idx)
        f = np.minimum(np.maximum(u - idx, 0.0), 1.0)
        i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
        tx, ty, tz = f[:, 0], f[:, 1], f[:, 2]

        c000 = V[i, j, k]
        c100 = V[i + 1, j, k]
        c010 = V[i, j + 1, k]
        c110 = V[i + 1, j + 1, k]
        c001 = V[i, j, k + 1]
        c101 = V[i + 1, j, k + 1]
        c011 = V[i, j + 1, k + 1]
        c111 = V[i + 1, j + 1, k + 1]

        c00 = c000 * (1 - tx) + c100 * tx
        c10 = c010 * (1 - tx) + c110 * tx
        c01 = c001 * (1 - tx) + c101 * tx
        c11 = c011 * (1 - tx) + c111 * tx
        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty
        phi = c0 * (1 - tz) + c1 * tz

        grad = np.empty_like(pts)
        grad[:, 0] = ((c100 - c000) * (1 - ty) * (1 - tz) + (c110 - c010) * ty * (1 - tz) +
                      (c101 - c001) * (1 - ty) * tz + (c111 - c011) * ty * tz) / h
        grad[:, 1] = ((c010 - c000) * (1 - tx) * (1 - tz) + (c110 - c100) * tx * (1 - tz) +
                      (c011 - c001) * (1 - tx) * tz + (c111 - c101) * tx * tz) / h
        grad[:, 2] = ((c001 - c000) * (1 - tx) * (1 - ty) + (c101 - c100) * tx * (1 - ty) +
                      (c011 - c010) * (1 - tx) * ty + (c111 - c110) * tx * ty) / h
        return phi, normalize(grad)


def _cubic_weights(t: float) -> tuple[np.ndarray, np.ndarray]:
    t2 = t * t
    t3 = t2 * t
    w = np.array([
        -0.5 * t + t2 - 0.5 * t3,
        1.0 - 2.5 * t2 + 1.5 * t3,
        0.5 * t + 2.0 * t2 - 1.5 * t3,
        -0.5 * t2 + 0.5 * t3,
    ], dtype=float)
    dw = np.array([
        -0.5 + 2.0 * t - 1.5 * t2,
        -5.0 * t + 4.5 * t2,
        0.5 + 4.0 * t - 4.5 * t2,
        -t + 1.5 * t2,
    ], dtype=float)
    return w, dw


def _cubic_weights_array(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float)
    t2 = t * t
    t3 = t2 * t
    w = np.stack([
        -0.5 * t + t2 - 0.5 * t3,
        1.0 - 2.5 * t2 + 1.5 * t3,
        0.5 * t + 2.0 * t2 - 1.5 * t3,
        -0.5 * t2 + 0.5 * t3,
    ], axis=1)
    dw = np.stack([
        -0.5 + 2.0 * t - 1.5 * t2,
        -5.0 * t + 4.5 * t2,
        0.5 + 4.0 * t - 4.5 * t2,
        -t + 1.5 * t2,
    ], axis=1)
    return w, dw


@dataclass
class CubicGridSDF:
    """Conventional scalar grid SDF with tricubic interpolation.

    This baseline still stores only scalar signed distances on grid nodes.  Its
    normal is the normalized gradient of the scalar interpolant, not a stored
    contact normal or normal-sector record.
    """
    origin: np.ndarray
    spacing: float
    values: np.ndarray

    @classmethod
    def from_grid(cls, grid: GridSDF) -> "CubicGridSDF":
        return cls(origin=grid.origin.copy(), spacing=float(grid.spacing), values=grid.values.copy())

    def clamp_cell_indices(self, p: np.ndarray):
        u = (p - self.origin) / self.spacing
        base = np.floor(u).astype(int)
        max_base = np.array(self.values.shape) - 2
        base = np.minimum(np.maximum(base, 0), max_base)
        f = u - base
        f = np.minimum(np.maximum(f, 0.0), 1.0)
        start = base - 1
        return start, f

    def eval(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(points, dtype=float)
        h = self.spacing
        V = self.values
        shape = np.array(V.shape)
        u = (pts - self.origin[None, :]) / h
        base = np.floor(u).astype(int)
        max_base = shape - 2
        base = np.minimum(np.maximum(base, 0), max_base)
        f = np.minimum(np.maximum(u - base, 0.0), 1.0)
        start = base - 1

        wx, dwx = _cubic_weights_array(f[:, 0])
        wy, dwy = _cubic_weights_array(f[:, 1])
        wz, dwz = _cubic_weights_array(f[:, 2])
        offs = np.arange(4)
        ix = np.clip(start[:, 0, None] + offs[None, :], 0, shape[0] - 1)
        iy = np.clip(start[:, 1, None] + offs[None, :], 0, shape[1] - 1)
        iz = np.clip(start[:, 2, None] + offs[None, :], 0, shape[2] - 1)
        block = V[ix[:, :, None, None], iy[:, None, :, None], iz[:, None, None, :]]

        phi = np.einsum("ni,nj,nk,nijk->n", wx, wy, wz, block)
        grad = np.empty_like(pts)
        grad[:, 0] = np.einsum("ni,nj,nk,nijk->n", dwx, wy, wz, block) / h
        grad[:, 1] = np.einsum("ni,nj,nk,nijk->n", wx, dwy, wz, block) / h
        grad[:, 2] = np.einsum("ni,nj,nk,nijk->n", wx, wy, dwz, block) / h
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


def build_cubic_grid_sdf(projector: MeshProjector, bbox: tuple[np.ndarray, np.ndarray],
                         resolution: int = 32, active_tol: float = 2e-3) -> CubicGridSDF:
    grid = build_grid_sdf(projector, bbox, resolution=resolution, active_tol=active_tol)
    return CubicGridSDF.from_grid(grid)
