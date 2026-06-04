"""Feature-aware Contact-SDF Atlas.

Uniform atlas:
  Offline: expensive projection is done once at cell centers.
  Runtime: contact queries are O(1) cell lookup plus a local polynomial or
  multi-normal-cone evaluation.

Adaptive atlas:
  Stores the same contact records on an octree-like dyadic grid.  Cells near
  high curvature, sharp feature transitions, or large local jet error are
  recursively split.  Runtime lookup is still O(1): every finest-grid slot stores
  the id of the leaf cell that covers it.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from itertools import product
import json
import numpy as np
from .projection import MeshProjector, normalize, angular_error_deg

MODE_SMOOTH = 0
MODE_MULTI = 1


@dataclass
class AtlasEval:
    phi: np.ndarray
    normal: np.ndarray
    mode: np.ndarray
    candidate_normals: list[np.ndarray]
    candidate_phi: list[np.ndarray]


@dataclass
class CompactAtlasEval:
    """Projection-free atlas result for solver hot paths.

    ``eval()`` keeps per-query candidate lists for diagnostics and normal-cone
    metrics.  ``eval_compact()`` avoids those Python allocations and returns the
    fixed-size fields a contact solver typically needs in its inner loop.
    """
    phi: np.ndarray
    normal: np.ndarray
    mode: np.ndarray
    leaf_id: np.ndarray
    candidate_count: np.ndarray


@dataclass
class _CellRecord:
    center: np.ndarray
    half_width: float
    level: int
    phi0: float
    normal0: np.ndarray
    hessian0: np.ndarray
    mode: int
    cand_phi0: np.ndarray
    cand_normal0: np.ndarray
    cand_count: int
    error_bound: float = 0.0


def _eval_record(record: _CellRecord, points: np.ndarray) -> AtlasEval:
    pts = np.asarray(points, dtype=float)
    dx = pts - record.center[None, :]
    phi = np.empty(pts.shape[0])
    normal = np.empty_like(pts)
    mode = np.full(pts.shape[0], record.mode, dtype=np.int8)
    cand_normals: list[np.ndarray] = []
    cand_phi: list[np.ndarray] = []
    H = record.hessian0
    n0 = record.normal0
    if record.mode == MODE_SMOOTH:
        phi[:] = record.phi0 + dx @ n0 + 0.5 * np.einsum('ni,ij,nj->n', dx, H, dx)
        normal[:] = normalize(n0[None, :] + dx @ H.T)
        for i in range(pts.shape[0]):
            cand_normals.append(normal[i][None, :])
            cand_phi.append(np.array([phi[i]]))
    else:
        c = int(record.cand_count)
        if c <= 0:
            phi[:] = record.phi0 + dx @ n0
            normal[:] = n0
            for i in range(pts.shape[0]):
                cand_normals.append(n0[None, :])
                cand_phi.append(np.array([phi[i]]))
        else:
            ns = record.cand_normal0[:c]
            # For scalar signed-distance reporting use the primary closest-sector jet;
            # all candidate gap planes are returned for normal-cone / multi-constraint contact.
            phi[:] = record.phi0 + dx @ n0
            normal[:] = n0
            gs = record.cand_phi0[:c][None, :] + dx @ ns.T
            for i in range(pts.shape[0]):
                cand_normals.append(ns.copy())
                cand_phi.append(gs[i].copy())
    return AtlasEval(phi=phi, normal=normalize(normal), mode=mode,
                     candidate_normals=cand_normals, candidate_phi=cand_phi)


@dataclass
class ContactSDFAtlas:
    origin: np.ndarray
    spacing: float
    phi0: np.ndarray          # (nx,ny,nz)
    normal0: np.ndarray       # (nx,ny,nz,3)
    hessian0: np.ndarray      # (nx,ny,nz,3,3)
    mode: np.ndarray          # (nx,ny,nz)
    cand_phi0: np.ndarray     # (nx,ny,nz,K)
    cand_normal0: np.ndarray  # (nx,ny,nz,K,3)
    cand_count: np.ndarray    # (nx,ny,nz)
    name: str = "atlas"

    def save_npz(self, path: str | Path) -> None:
        np.savez_compressed(path, origin=self.origin, spacing=self.spacing, phi0=self.phi0,
                            normal0=self.normal0, hessian0=self.hessian0, mode=self.mode,
                            cand_phi0=self.cand_phi0, cand_normal0=self.cand_normal0,
                            cand_count=self.cand_count, name=np.array(self.name))

    @staticmethod
    def load_npz(path: str | Path) -> "ContactSDFAtlas":
        z = np.load(path, allow_pickle=True)
        name = str(z["name"]) if "name" in z else Path(path).stem
        return ContactSDFAtlas(z["origin"], float(z["spacing"]), z["phi0"], z["normal0"],
                               z["hessian0"], z["mode"], z["cand_phi0"], z["cand_normal0"],
                               z["cand_count"], name=name)

    @property
    def shape(self):
        return self.phi0.shape

    def _locate(self, p: np.ndarray):
        u = (p - self.origin) / self.spacing
        idx = np.floor(u).astype(int)
        max_idx = np.array(self.shape) - 1
        idx = np.minimum(np.maximum(idx, 0), max_idx)
        center = self.origin + (idx + 0.5) * self.spacing
        dx = p - center
        return idx, dx

    def eval(self, points: np.ndarray) -> AtlasEval:
        pts = np.asarray(points, dtype=float)
        phi = np.empty(pts.shape[0])
        normal = np.empty_like(pts)
        mode = np.empty(pts.shape[0], dtype=np.int8)
        cand_normals: list[np.ndarray] = []
        cand_phi: list[np.ndarray] = []
        for q, p in enumerate(pts):
            (i, j, k), dx = self._locate(p)
            m = int(self.mode[i, j, k])
            mode[q] = m
            H = self.hessian0[i, j, k]
            n0 = self.normal0[i, j, k]
            if m == MODE_SMOOTH:
                phi[q] = self.phi0[i, j, k] + float(np.dot(n0, dx)) + 0.5 * float(dx @ H @ dx)
                normal[q] = normalize((n0 + H @ dx)[None, :])[0]
                cand_normals.append(normal[q][None, :])
                cand_phi.append(np.array([phi[q]]))
            else:
                c = int(self.cand_count[i, j, k])
                ns = self.cand_normal0[i, j, k, :c]
                gs = self.cand_phi0[i, j, k, :c] + ns @ dx
                if c == 0:
                    phi[q] = self.phi0[i, j, k] + float(np.dot(n0, dx))
                    normal[q] = n0
                    cand_normals.append(n0[None, :])
                    cand_phi.append(np.array([phi[q]]))
                else:
                    phi[q] = self.phi0[i, j, k] + float(np.dot(n0, dx))
                    normal[q] = n0
                    cand_normals.append(ns.copy())
                    cand_phi.append(gs.copy())
        return AtlasEval(phi=phi, normal=normalize(normal), mode=mode,
                         candidate_normals=cand_normals, candidate_phi=cand_phi)

    def eval_compact(self, points: np.ndarray) -> CompactAtlasEval:
        pts = np.asarray(points, dtype=float)
        u = (pts - self.origin[None, :]) / self.spacing
        idx = np.floor(u).astype(int)
        max_idx = np.array(self.shape) - 1
        idx = np.minimum(np.maximum(idx, 0), max_idx)
        center = self.origin[None, :] + (idx.astype(float) + 0.5) * self.spacing
        dx = pts - center

        i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
        phi0 = self.phi0[i, j, k]
        n0 = self.normal0[i, j, k]
        H = self.hessian0[i, j, k]
        mode = self.mode[i, j, k].astype(np.int8, copy=True)

        phi = phi0 + np.einsum("ni,ni->n", n0, dx)
        normal = n0.copy()
        smooth = mode == MODE_SMOOTH
        if np.any(smooth):
            dx_s = dx[smooth]
            H_s = H[smooth]
            phi[smooth] += 0.5 * np.einsum("ni,nij,nj->n", dx_s, H_s, dx_s)
            normal[smooth] = n0[smooth] + np.einsum("nij,nj->ni", H_s, dx_s)

        flat_id = np.ravel_multi_index((i, j, k), self.shape).astype(np.int64)
        candidate_count = self.cand_count[i, j, k].astype(np.int16, copy=True)
        return CompactAtlasEval(
            phi=phi,
            normal=normalize(normal),
            mode=mode,
            leaf_id=flat_id,
            candidate_count=candidate_count,
        )


def _estimate_hessians(normals: np.ndarray, spacing: float) -> np.ndarray:
    nx, ny, nz, _ = normals.shape
    H = np.zeros((nx, ny, nz, 3, 3), dtype=float)
    # H[row, col] = d n_row / d x_col.
    for axis in range(3):
        sp_mid = [slice(None)] * 3
        sp_p = [slice(None)] * 3
        sp_m = [slice(None)] * 3
        sp_mid[axis] = slice(1, -1)
        sp_p[axis] = slice(2, None)
        sp_m[axis] = slice(None, -2)
        deriv = (normals[tuple(sp_p) + (slice(None),)] - normals[tuple(sp_m) + (slice(None),)]) / (2 * spacing)
        H[tuple(sp_mid) + (slice(None), axis)] = deriv
    # Symmetrize Hessian as the Jacobian of a true SDF gradient should be symmetric.
    H = 0.5 * (H + np.swapaxes(H, -1, -2))
    return H


def build_contact_sdf_atlas(projector: MeshProjector, bbox: tuple[np.ndarray, np.ndarray],
                            resolution: int = 31, max_candidates: int = 6,
                            active_tol: float = 8e-3,
                            multi_if_feature: bool = True,
                            sector_angle_deg: float = 20.0) -> ContactSDFAtlas:
    """Build a uniform atlas on cell centers.

    ``resolution`` is the number of scalar-grid nodes one would use; atlas stores
    ``(resolution-1)^3`` cell records.
    """
    lo, hi = bbox
    extent = hi - lo
    spacing = float(np.max(extent) / (resolution - 1))
    center = 0.5 * (lo + hi)
    half = 0.5 * spacing * (resolution - 1)
    node_origin = center - half
    cell_origin = node_origin
    ncell = resolution - 1
    axes = [node_origin[d] + (np.arange(ncell) + 0.5) * spacing for d in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing='ij')
    pts = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    res = projector.project(pts, active_tol=active_tol, sector_angle_deg=sector_angle_deg)

    phi0 = res.phi.reshape((ncell, ncell, ncell))
    normal0 = res.normal.reshape((ncell, ncell, ncell, 3))
    mode = np.zeros((ncell, ncell, ncell), dtype=np.int8)
    cand_phi0 = np.zeros((ncell, ncell, ncell, max_candidates), dtype=float)
    cand_normal0 = np.zeros((ncell, ncell, ncell, max_candidates, 3), dtype=float)
    cand_count = np.zeros((ncell, ncell, ncell), dtype=np.int16)

    for flat, active in enumerate(res.active_normals):
        idx = np.unravel_index(flat, (ncell, ncell, ncell))
        if (len(active) > 1) or (multi_if_feature and res.feature[flat] > 0):
            mode[idx] = MODE_MULTI
            c = min(len(active), max_candidates)
            cand_count[idx] = c
            for a in range(c):
                n = active[a]
                cand_normal0[idx + (a, slice(None))] = n
                cand_phi0[idx + (a,)] = float(np.dot(pts[flat] - res.closest[flat], n))
        else:
            cand_count[idx] = 1
            cand_phi0[idx + (0,)] = res.phi[flat]
            cand_normal0[idx + (0, slice(None))] = res.normal[flat]

    H = _estimate_hessians(normal0, spacing)
    H[mode == MODE_MULTI] = 0.0
    return ContactSDFAtlas(origin=cell_origin, spacing=spacing, phi0=phi0, normal0=normal0,
                           hessian0=H, mode=mode, cand_phi0=cand_phi0,
                           cand_normal0=cand_normal0, cand_count=cand_count,
                           name=f"uniform_atlas_res{resolution}")


@dataclass
class AdaptiveContactSDFAtlas:
    """Adaptive dyadic Contact-SDF Atlas.

    The tree is stored as leaf records plus a dense finest-grid indirection table:
    ``leaf_grid[i,j,k]`` gives the leaf id that covers a query voxel.  This keeps
    runtime queries as cheap as a conventional SDF lookup even though the offline
    atlas is refined near difficult contact features.
    """
    origin: np.ndarray
    base_ncell: int
    max_depth: int
    base_spacing: float
    finest_spacing: float
    leaf_grid: np.ndarray
    leaf_center: np.ndarray
    leaf_half_width: np.ndarray
    leaf_level: np.ndarray
    phi0: np.ndarray
    normal0: np.ndarray
    hessian0: np.ndarray
    mode: np.ndarray
    cand_phi0: np.ndarray
    cand_normal0: np.ndarray
    cand_count: np.ndarray
    error_bound: np.ndarray
    stats: dict = field(default_factory=dict)
    name: str = "adaptive_atlas"

    def save_npz(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            origin=self.origin,
            base_ncell=np.array(self.base_ncell),
            max_depth=np.array(self.max_depth),
            base_spacing=np.array(self.base_spacing),
            finest_spacing=np.array(self.finest_spacing),
            leaf_grid=self.leaf_grid,
            leaf_center=self.leaf_center,
            leaf_half_width=self.leaf_half_width,
            leaf_level=self.leaf_level,
            phi0=self.phi0,
            normal0=self.normal0,
            hessian0=self.hessian0,
            mode=self.mode,
            cand_phi0=self.cand_phi0,
            cand_normal0=self.cand_normal0,
            cand_count=self.cand_count,
            error_bound=self.error_bound,
            stats=np.array(json.dumps(self.stats)),
            name=np.array(self.name),
        )

    @staticmethod
    def load_npz(path: str | Path) -> "AdaptiveContactSDFAtlas":
        z = np.load(path, allow_pickle=True)
        stats = {}
        if "stats" in z:
            try:
                stats = json.loads(str(z["stats"]))
            except Exception:
                stats = {}
        name = str(z["name"]) if "name" in z else Path(path).stem
        return AdaptiveContactSDFAtlas(
            origin=z["origin"],
            base_ncell=int(z["base_ncell"]),
            max_depth=int(z["max_depth"]),
            base_spacing=float(z["base_spacing"]),
            finest_spacing=float(z["finest_spacing"]),
            leaf_grid=z["leaf_grid"],
            leaf_center=z["leaf_center"],
            leaf_half_width=z["leaf_half_width"],
            leaf_level=z["leaf_level"],
            phi0=z["phi0"],
            normal0=z["normal0"],
            hessian0=z["hessian0"],
            mode=z["mode"],
            cand_phi0=z["cand_phi0"],
            cand_normal0=z["cand_normal0"],
            cand_count=z["cand_count"],
            error_bound=z["error_bound"],
            stats=stats,
            name=name,
        )

    @property
    def n_leaves(self) -> int:
        return int(self.leaf_center.shape[0])

    def _locate_leaf_ids(self, pts: np.ndarray) -> np.ndarray:
        u = (pts - self.origin[None, :]) / self.finest_spacing
        idx = np.floor(u).astype(int)
        max_idx = np.array(self.leaf_grid.shape) - 1
        idx = np.minimum(np.maximum(idx, 0), max_idx)
        return self.leaf_grid[idx[:, 0], idx[:, 1], idx[:, 2]]

    def eval(self, points: np.ndarray) -> AtlasEval:
        pts = np.asarray(points, dtype=float)
        leaf_ids = self._locate_leaf_ids(pts)
        phi = np.empty(pts.shape[0])
        normal = np.empty_like(pts)
        mode_out = np.empty(pts.shape[0], dtype=np.int8)
        cand_normals: list[np.ndarray] = []
        cand_phi: list[np.ndarray] = []
        for lid in np.unique(leaf_ids):
            lid = int(lid)
            mask = leaf_ids == lid
            rec = _CellRecord(
                center=self.leaf_center[lid],
                half_width=float(self.leaf_half_width[lid]),
                level=int(self.leaf_level[lid]),
                phi0=float(self.phi0[lid]),
                normal0=self.normal0[lid],
                hessian0=self.hessian0[lid],
                mode=int(self.mode[lid]),
                cand_phi0=self.cand_phi0[lid],
                cand_normal0=self.cand_normal0[lid],
                cand_count=int(self.cand_count[lid]),
                error_bound=float(self.error_bound[lid]),
            )
            ev = _eval_record(rec, pts[mask])
            phi[mask] = ev.phi
            normal[mask] = ev.normal
            mode_out[mask] = ev.mode
            # Fill list positions in original order below.
        # Lists are order-sensitive, so compute per point.  This is still only
        # polynomial evaluation; no projection occurs online.
        for i, lid in enumerate(leaf_ids):
            lid = int(lid)
            dx = pts[i] - self.leaf_center[lid]
            c = int(self.cand_count[lid])
            if int(self.mode[lid]) == MODE_MULTI and c > 0:
                ns = self.cand_normal0[lid, :c]
                gs = self.cand_phi0[lid, :c] + ns @ dx
                cand_normals.append(ns.copy())
                cand_phi.append(gs.copy())
            else:
                cand_normals.append(normal[i][None, :])
                cand_phi.append(np.array([phi[i]]))
        return AtlasEval(phi=phi, normal=normalize(normal), mode=mode_out,
                         candidate_normals=cand_normals, candidate_phi=cand_phi)

    def eval_compact(self, points: np.ndarray) -> CompactAtlasEval:
        pts = np.asarray(points, dtype=float)
        leaf_ids = self._locate_leaf_ids(pts).astype(np.int64, copy=False)
        center = self.leaf_center[leaf_ids]
        dx = pts - center
        n0 = self.normal0[leaf_ids]
        H = self.hessian0[leaf_ids]
        mode = self.mode[leaf_ids].astype(np.int8, copy=True)

        phi = self.phi0[leaf_ids] + np.einsum("ni,ni->n", n0, dx)
        normal = n0.copy()
        smooth = mode == MODE_SMOOTH
        if np.any(smooth):
            dx_s = dx[smooth]
            H_s = H[smooth]
            phi[smooth] += 0.5 * np.einsum("ni,nij,nj->n", dx_s, H_s, dx_s)
            normal[smooth] = n0[smooth] + np.einsum("nij,nj->ni", H_s, dx_s)

        return CompactAtlasEval(
            phi=phi,
            normal=normalize(normal),
            mode=mode,
            leaf_id=leaf_ids,
            candidate_count=self.cand_count[leaf_ids].astype(np.int16, copy=True),
        )


def _make_record(projector: MeshProjector, center: np.ndarray, h: float,
                 max_candidates: int, active_tol: float,
                 multi_if_feature: bool, hessian_for_smooth: bool,
                 sector_angle_deg: float = 30.0) -> _CellRecord:
    res = projector.project(center[None, :], active_tol=active_tol, sector_angle_deg=sector_angle_deg)
    active = res.active_normals[0]
    mode = MODE_MULTI if ((len(active) > 1) or (multi_if_feature and int(res.feature[0]) > 0)) else MODE_SMOOTH
    cand_phi0 = np.zeros((max_candidates,), dtype=float)
    cand_normal0 = np.zeros((max_candidates, 3), dtype=float)
    c = min(len(active), max_candidates) if mode == MODE_MULTI else 1
    if mode == MODE_MULTI:
        for a in range(c):
            n = active[a]
            cand_normal0[a] = n
            cand_phi0[a] = float(np.dot(center - res.closest[0], n))
        H = np.zeros((3, 3), dtype=float)
    else:
        cand_phi0[0] = float(res.phi[0])
        cand_normal0[0] = res.normal[0]
        H = np.zeros((3, 3), dtype=float)
        if hessian_for_smooth:
            delta = max(0.20 * h, 1e-6)
            pts = []
            for ax in range(3):
                e = np.zeros(3); e[ax] = delta
                pts.append(center + e); pts.append(center - e)
            rr = projector.project(np.asarray(pts), active_tol=active_tol, sector_angle_deg=sector_angle_deg)
            for ax in range(3):
                np_ = rr.normal[2 * ax]
                nm_ = rr.normal[2 * ax + 1]
                H[:, ax] = (np_ - nm_) / (2 * delta)
            H = 0.5 * (H + H.T)
    return _CellRecord(center=np.asarray(center, dtype=float), half_width=0.5 * h, level=0,
                       phi0=float(res.phi[0]), normal0=res.normal[0], hessian0=H,
                       mode=mode, cand_phi0=cand_phi0, cand_normal0=cand_normal0,
                       cand_count=int(c), error_bound=0.0)


def _sample_cell(center: np.ndarray, h: float) -> np.ndarray:
    # 8 near-corners plus 6 face centers.  Use 0.49h to avoid classifying a
    # point exactly on a neighboring leaf boundary during error estimation.
    pts = []
    a = 0.49 * h
    for sx, sy, sz in product([-1.0, 1.0], repeat=3):
        pts.append(center + np.array([sx, sy, sz]) * a)
    for ax in range(3):
        for s in [-1.0, 1.0]:
            d = np.zeros(3); d[ax] = s * a
            pts.append(center + d)
    return np.asarray(pts)



def _merge_candidate_planes(center: np.ndarray, records: list[tuple[np.ndarray, float]],
                            max_candidates: int, sector_angle_deg: float = 12.0) -> tuple[np.ndarray, np.ndarray, int]:
    """Merge candidate normal/gap planes by angular sector.

    records contain (normal, phi_at_center).  The first occurrence of a sector is
    kept; later records in the same sector update the constant by averaging.  A
    feature cell does not need a dense list of all triangles; it needs one stable
    plane per normal sector for multi-constraint contact.
    """
    if not records:
        return np.zeros((max_candidates,), dtype=float), np.zeros((max_candidates, 3), dtype=float), 0
    cos_thr = float(np.cos(np.radians(sector_angle_deg)))
    normals: list[np.ndarray] = []
    phis: list[list[float]] = []
    for n, phi_c in records:
        n = normalize(np.asarray(n, dtype=float)[None, :])[0]
        placed = False
        for j, u in enumerate(normals):
            if float(np.dot(n, u)) > cos_thr:
                phis[j].append(float(phi_c))
                placed = True
                break
        if not placed:
            normals.append(n)
            phis.append([float(phi_c)])
        if len(normals) >= max_candidates:
            # Keep only the most distinct first sectors; this mirrors the fixed
            # storage budget in a real atlas leaf.
            pass
    # Sort sectors by the absolute plane gap at the center: most contact-relevant first.
    order = sorted(range(len(normals)), key=lambda i: abs(float(np.mean(phis[i]))))[:max_candidates]
    cand_phi0 = np.zeros((max_candidates,), dtype=float)
    cand_normal0 = np.zeros((max_candidates, 3), dtype=float)
    for out_i, j in enumerate(order):
        cand_normal0[out_i] = normals[j]
        cand_phi0[out_i] = float(np.mean(phis[j]))
    return cand_phi0, cand_normal0, len(order)


def _feature_enrich_record(projector: MeshProjector, record: _CellRecord, h: float,
                           max_candidates: int, active_tol: float,
                           sector_angle_deg: float = 30.0) -> _CellRecord:
    """Turn a leaf into a feature-specific multi-record when its cell samples
    indicate edge/vertex/multi-sector behavior.

    This is the key difference from ordinary adaptive refinement: at the final
    feature depth, the atlas does not store a single smoothed normal.  It stores
    a compact set of normal-sector gap planes collected from the cell center and
    local samples.  Runtime evaluation remains a lookup + dot products.
    """
    sample = np.vstack([record.center[None, :], _sample_cell(record.center, h)])
    ref = projector.project(sample, active_tol=active_tol, sector_angle_deg=sector_angle_deg)
    records: list[tuple[np.ndarray, float]] = []
    has_feature = False
    for x_s, p_s, phi_s, feat_s, active in zip(sample, ref.closest, ref.phi, ref.feature, ref.active_normals):
        # A triangle edge/vertex hit is not necessarily a physical sharp feature
        # for a corner-normal mesh; smooth charts can be triangulated.  We treat
        # only genuinely different normal sectors as feature evidence.
        if len(active) > 1:
            has_feature = True
        for n in active:
            # Plane gap value at the leaf center for this local sector.
            phi_center = float(phi_s - np.dot(n, x_s - record.center))
            records.append((n, phi_center))
    if not has_feature and record.mode != MODE_MULTI:
        return record
    cand_phi0, cand_normal0, c = _merge_candidate_planes(
        record.center, records, max_candidates=max_candidates, sector_angle_deg=sector_angle_deg
    )
    if c > 0:
        record.mode = MODE_MULTI
        record.cand_phi0[:] = 0.0
        record.cand_normal0[:] = 0.0
        record.cand_phi0[:c] = cand_phi0[:c]
        record.cand_normal0[:c] = cand_normal0[:c]
        record.cand_count = int(c)
        # Keep the scalar phi/normal fields as primary reporting values from the
        # center projection, but force Hessian off in non-smooth leaves.
        record.hessian0[:] = 0.0
    return record


def _needs_refine(projector: MeshProjector, record: _CellRecord, h: float, level: int,
                  smooth_max_depth: int, feature_max_depth: int, refine_band: float,
                  active_tol: float, gap_tol_factor: float, normal_tol_deg: float,
                  feature_normal_tol_deg: float, force_feature_to_depth: int,
                  sector_angle_deg: float = 30.0) -> tuple[bool, float, dict]:
    """Decide whether a leaf should split.

    Smooth leaves are allowed to refine only to ``smooth_max_depth``.  Feature
    leaves (edge/vertex/multi-normal-sector leaves) can refine deeper to
    ``feature_max_depth``.  This is feature-specific refinement: the extra depth
    is spent only where contact is non-smooth or where the local active feature
    set changes within the cell.
    """
    if level >= feature_max_depth:
        return False, 0.0, {"reason": "feature_max_depth"}

    cell_radius = float(np.sqrt(3.0) * 0.5 * h)
    if abs(record.phi0) > refine_band + cell_radius:
        return False, 0.0, {"reason": "far_center_lipschitz"}

    sample = _sample_cell(record.center, h)
    ref = projector.project(sample, active_tol=active_tol, sector_angle_deg=sector_angle_deg)
    near = bool(min(np.min(np.abs(ref.phi)), abs(record.phi0)) <= refine_band)
    if not near:
        return False, 0.0, {"reason": "far_samples"}

    ev = _eval_record(record, sample)
    gap_err = np.abs(ev.phi - ref.phi)
    ang = np.empty(sample.shape[0])
    for i, ns in enumerate(ev.candidate_normals):
        dots = normalize(ns) @ normalize(ref.normal[i][None, :])[0]
        ang[i] = np.degrees(np.arccos(np.clip(np.max(dots), -1.0, 1.0)))
    # Feature-specific refinement must be driven by normal-sector competition,
    # not by the PL triangle region code.  A smooth curved surface can project to
    # a triangle edge even when there is no physical sharp feature.
    ref_multi = np.array([(len(a) > 1) for a in ref.active_normals])
    is_feature_cell = bool(record.mode == MODE_MULTI or np.any(ref_multi))

    max_gap = float(np.max(gap_err))
    max_ang = float(np.max(ang))
    err_bound = max(max_gap, h * max_ang / 180.0)
    reason = {
        "max_gap": max_gap,
        "max_ang": max_ang,
        "ref_multi": bool(np.any(ref_multi)),
        "mode_multi": bool(record.mode == MODE_MULTI),
        "is_feature_cell": is_feature_cell,
    }

    if is_feature_cell:
        if level < force_feature_to_depth:
            return True, err_bound, reason | {"reason": "feature_min_depth"}
        if max_ang > feature_normal_tol_deg and level < feature_max_depth:
            return True, err_bound, reason | {"reason": "feature_normal"}
        if max_gap > gap_tol_factor * h and level < feature_max_depth:
            return True, err_bound, reason | {"reason": "feature_gap"}
        if np.any(ref_multi) and level < feature_max_depth and max_ang > 0.5 * feature_normal_tol_deg:
            return True, err_bound, reason | {"reason": "feature_transition"}
        return False, err_bound, reason | {"reason": "feature_ok"}

    if level >= smooth_max_depth:
        return False, err_bound, reason | {"reason": "smooth_depth_stop"}
    if max_gap > gap_tol_factor * h:
        return True, err_bound, reason | {"reason": "smooth_gap"}
    if max_ang > normal_tol_deg:
        return True, err_bound, reason | {"reason": "smooth_normal"}
    return False, err_bound, reason | {"reason": "smooth_ok"}

def build_adaptive_contact_sdf_atlas(
    projector: MeshProjector,
    bbox: tuple[np.ndarray, np.ndarray],
    base_resolution: int = 13,
    max_depth: int = 2,
    max_candidates: int = 8,
    active_tol: float = 1.5e-2,
    multi_if_feature: bool = False,
    refine_band: float | None = None,
    gap_tol_factor: float = 0.10,
    normal_tol_deg: float = 8.0,
    always_refine_multi_to: int | None = None,
    hessian_for_smooth: bool = True,
    feature_max_depth: int | None = None,
    feature_normal_tol_deg: float = 4.0,
    feature_enrichment: bool = True,
    sector_angle_deg: float = 35.0,
) -> AdaptiveContactSDFAtlas:
    """Build an adaptive Feature-aware Contact-SDF Atlas.

    The earlier adaptive atlas used a single maximum depth for all cells.  This
    builder adds *feature-specific refinement*: smooth cells stop at
    ``max_depth`` while edge/vertex/multi-sector cells may keep splitting until
    ``feature_max_depth``.  At the final feature leaves, candidate normal-sector
    gap planes are baked into the record, so runtime evaluation still performs
    no closest-point projection.

    Parameters
    ----------
    base_resolution:
        Number of scalar-grid nodes in each direction before refinement.  The
        initial atlas has ``base_resolution-1`` cells per direction.
    max_depth:
        Maximum depth for ordinary smooth cells.
    feature_max_depth:
        Maximum depth for non-smooth feature cells.  If omitted, it equals
        ``max_depth`` and the behavior reduces to the previous adaptive atlas.
    refine_band:
        Only cells whose sampled reference distances come within this band are
        refined.  By default it is two base spacings.
    feature_enrichment:
        If true, final feature leaves store a union of local normal-sector
        records sampled inside the leaf rather than only the center projection.
    """
    lo, hi = bbox
    extent = hi - lo
    base_spacing = float(np.max(extent) / (base_resolution - 1))
    center = 0.5 * (lo + hi)
    half = 0.5 * base_spacing * (base_resolution - 1)
    origin = center - half
    base_ncell = base_resolution - 1
    if refine_band is None:
        refine_band = 2.0 * base_spacing
    if feature_max_depth is None:
        feature_max_depth = max_depth
    feature_max_depth = int(max(feature_max_depth, max_depth))
    smooth_max_depth = int(max_depth)
    if always_refine_multi_to is None:
        # Force feature cells at least one level beyond the smooth atlas when possible.
        always_refine_multi_to = min(feature_max_depth, smooth_max_depth + 1)
    tree_max_depth = int(feature_max_depth)
    finest_n = base_ncell * (2 ** tree_max_depth)
    finest_spacing = base_spacing / (2 ** tree_max_depth)
    leaf_grid = -np.ones((finest_n, finest_n, finest_n), dtype=np.int32)
    leaves: list[_CellRecord] = []
    stats = {
        "base_resolution": base_resolution,
        "base_ncell": base_ncell,
        "smooth_max_depth": smooth_max_depth,
        "feature_max_depth": feature_max_depth,
        "max_depth": feature_max_depth,
        "base_spacing": base_spacing,
        "finest_spacing": finest_spacing,
        "refine_band": float(refine_band),
        "gap_tol_factor": gap_tol_factor,
        "normal_tol_deg": normal_tol_deg,
        "feature_normal_tol_deg": feature_normal_tol_deg,
        "feature_enrichment": bool(feature_enrichment),
        "sector_angle_deg": float(sector_angle_deg),
        "split_count": 0,
        "feature_split_count": 0,
        "smooth_split_count": 0,
        "max_depth_leaf_count": 0,
        "feature_leaf_count": 0,
        "level_counts": {},
        "mode_counts": {},
        "split_reasons": {},
    }

    def add_leaf(record: _CellRecord, fine_min: np.ndarray, fine_size: int) -> None:
        lid = len(leaves)
        leaves.append(record)
        i0, j0, k0 = map(int, fine_min)
        leaf_grid[i0:i0 + fine_size, j0:j0 + fine_size, k0:k0 + fine_size] = lid
        stats["level_counts"][str(record.level)] = stats["level_counts"].get(str(record.level), 0) + 1
        stats["mode_counts"][str(record.mode)] = stats["mode_counts"].get(str(record.mode), 0) + 1
        if record.level == tree_max_depth:
            stats["max_depth_leaf_count"] += 1
        if int(record.mode) == MODE_MULTI:
            stats["feature_leaf_count"] += 1

    def recurse(fine_min: np.ndarray, fine_size: int, level: int) -> None:
        h = finest_spacing * fine_size
        center_cell = origin + (fine_min.astype(float) + 0.5 * fine_size) * finest_spacing
        rec = _make_record(projector, center_cell, h, max_candidates, active_tol,
                           multi_if_feature, hessian_for_smooth and level <= smooth_max_depth,
                           sector_angle_deg=sector_angle_deg)
        rec.level = level
        rec.half_width = 0.5 * h
        refine, err_bound, info = _needs_refine(
            projector, rec, h, level,
            smooth_max_depth=smooth_max_depth,
            feature_max_depth=feature_max_depth,
            refine_band=float(refine_band),
            active_tol=active_tol,
            gap_tol_factor=gap_tol_factor,
            normal_tol_deg=normal_tol_deg,
            feature_normal_tol_deg=feature_normal_tol_deg,
            force_feature_to_depth=int(always_refine_multi_to),
            sector_angle_deg=sector_angle_deg,
        )
        rec.error_bound = float(err_bound)
        reason = str(info.get("reason", "unknown"))
        if refine:
            stats["split_count"] += 1
            stats["split_reasons"][reason] = stats["split_reasons"].get(reason, 0) + 1
            if bool(info.get("is_feature_cell", False)) or reason.startswith("feature"):
                stats["feature_split_count"] += 1
            else:
                stats["smooth_split_count"] += 1
            child_size = fine_size // 2
            if child_size < 1:
                # Should not happen, but keep the atlas robust.
                if feature_enrichment:
                    rec = _feature_enrich_record(projector, rec, h, max_candidates, active_tol,
                                             sector_angle_deg=sector_angle_deg)
                add_leaf(rec, fine_min, fine_size)
                return
            for ox, oy, oz in product([0, child_size], repeat=3):
                recurse(fine_min + np.array([ox, oy, oz], dtype=int), child_size, level + 1)
        else:
            # At the final feature-specific depth, bake multiple normal sectors
            # if the leaf samples reveal an edge/vertex transition.  This is
            # still offline-only; eval() remains projection-free.
            if feature_enrichment and (int(rec.mode) == MODE_MULTI or bool(info.get("is_feature_cell", False))):
                rec = _feature_enrich_record(projector, rec, h, max_candidates, active_tol,
                                             sector_angle_deg=sector_angle_deg)
            add_leaf(rec, fine_min, fine_size)

    root_size = 2 ** tree_max_depth
    for i, j, k in product(range(base_ncell), repeat=3):
        recurse(np.array([i * root_size, j * root_size, k * root_size], dtype=int), root_size, 0)

    # Defensive fill: if numerical recursion ever misses a slot, assign nearest leaf.
    missing = leaf_grid < 0
    if np.any(missing):
        centers = np.asarray([r.center for r in leaves])
        miss_idx = np.argwhere(missing)
        miss_pts = origin[None, :] + (miss_idx.astype(float) + 0.5) * finest_spacing
        for start in range(0, len(miss_idx), 1024):
            pts = miss_pts[start:start + 1024]
            d2 = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            ids = np.argmin(d2, axis=1).astype(np.int32)
            idxs = miss_idx[start:start + 1024]
            leaf_grid[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = ids

    leaf_center = np.asarray([r.center for r in leaves], dtype=float)
    leaf_half_width = np.asarray([r.half_width for r in leaves], dtype=float)
    leaf_level = np.asarray([r.level for r in leaves], dtype=np.int16)
    phi0 = np.asarray([r.phi0 for r in leaves], dtype=float)
    normal0 = np.asarray([r.normal0 for r in leaves], dtype=float)
    hessian0 = np.asarray([r.hessian0 for r in leaves], dtype=float)
    mode = np.asarray([r.mode for r in leaves], dtype=np.int8)
    cand_phi0 = np.asarray([r.cand_phi0 for r in leaves], dtype=float)
    cand_normal0 = np.asarray([r.cand_normal0 for r in leaves], dtype=float)
    cand_count = np.asarray([r.cand_count for r in leaves], dtype=np.int16)
    error_bound = np.asarray([r.error_bound for r in leaves], dtype=float)
    stats["leaf_count"] = int(len(leaves))
    stats["compression_vs_finest_uniform"] = float((finest_n ** 3) / max(len(leaves), 1))

    return AdaptiveContactSDFAtlas(
        origin=origin,
        base_ncell=base_ncell,
        max_depth=tree_max_depth,
        base_spacing=base_spacing,
        finest_spacing=finest_spacing,
        leaf_grid=leaf_grid,
        leaf_center=leaf_center,
        leaf_half_width=leaf_half_width,
        leaf_level=leaf_level,
        phi0=phi0,
        normal0=normal0,
        hessian0=hessian0,
        mode=mode,
        cand_phi0=cand_phi0,
        cand_normal0=cand_normal0,
        cand_count=cand_count,
        error_bound=error_bound,
        stats=stats,
        name=f"feature_adaptive_atlas_base{base_resolution}_s{smooth_max_depth}_f{feature_max_depth}",
    )
