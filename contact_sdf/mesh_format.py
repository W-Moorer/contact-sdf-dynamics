"""Corner-normal mesh data model and simple RMD-like export.

A mesh is represented by triangle-local vertex positions and triangle-local
corner normals.  This intentionally allows the same geometric vertex to carry
multiple normals in different adjacent triangles.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


@dataclass
class CornerNormalMesh:
    """Triangle-local mesh: triangles[f, i, :] and normals[f, i, :]."""

    triangles: np.ndarray  # (F, 3, 3)
    corner_normals: np.ndarray  # (F, 3, 3)
    name: str = "mesh"
    tags: dict | None = None

    def __post_init__(self) -> None:
        self.triangles = np.asarray(self.triangles, dtype=float)
        self.corner_normals = _normalize(np.asarray(self.corner_normals, dtype=float))
        if self.triangles.ndim != 3 or self.triangles.shape[1:] != (3, 3):
            raise ValueError("triangles must have shape (F,3,3)")
        if self.corner_normals.shape != self.triangles.shape:
            raise ValueError("corner_normals must have the same shape as triangles")
        self.tags = dict(self.tags or {})

    @property
    def n_faces(self) -> int:
        return int(self.triangles.shape[0])

    def face_normals(self) -> np.ndarray:
        e1 = self.triangles[:, 1] - self.triangles[:, 0]
        e2 = self.triangles[:, 2] - self.triangles[:, 0]
        return _normalize(np.cross(e1, e2))

    def bbox(self, pad: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        lo = self.triangles.reshape(-1, 3).min(axis=0)
        hi = self.triangles.reshape(-1, 3).max(axis=0)
        ext = hi - lo
        return lo - pad * np.maximum(ext, 1e-9), hi + pad * np.maximum(ext, 1e-9)

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        np.savez_compressed(path, triangles=self.triangles, corner_normals=self.corner_normals,
                            name=np.array(self.name), tags=np.array(json.dumps(self.tags)))

    @staticmethod
    def load_npz(path: str | Path) -> "CornerNormalMesh":
        z = np.load(path, allow_pickle=True)
        tags = {}
        if "tags" in z:
            try:
                tags = json.loads(str(z["tags"]))
            except Exception:
                tags = {}
        name = str(z["name"]) if "name" in z else Path(path).stem
        return CornerNormalMesh(z["triangles"], z["corner_normals"], name=name, tags=tags)

    def save_rmd_like(self, path: str | Path) -> None:
        """Export an ASCII, RMD-inspired corner-normal triangle block.

        This is *not* the proprietary RecurDyn RMD syntax.  It is a compact
        interchange file with one TRI record per face:

        TRI x0 y0 z0 nx0 ny0 nz0 x1 y1 z1 nx1 ny1 nz1 x2 y2 z2 nx2 ny2 nz2
        """
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            f.write("# CNMESH_RMD_LIKE_V1\n")
            f.write(f"# name {self.name}\n")
            f.write(f"# n_faces {self.n_faces}\n")
            for tri, ns in zip(self.triangles, self.corner_normals):
                vals = []
                for i in range(3):
                    vals.extend(tri[i].tolist())
                    vals.extend(ns[i].tolist())
                f.write("TRI " + " ".join(f"{v:.17g}" for v in vals) + "\n")

    @staticmethod
    def load_rmd_like(path: str | Path, name: str | None = None) -> "CornerNormalMesh":
        tris, ns = [], []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if not line.startswith("TRI "):
                    continue
                vals = [float(x) for x in line.split()[1:]]
                if len(vals) != 18:
                    raise ValueError(f"TRI record must have 18 numbers, got {len(vals)}")
                tri = []
                nn = []
                for i in range(3):
                    base = 6 * i
                    tri.append(vals[base:base + 3])
                    nn.append(vals[base + 3:base + 6])
                tris.append(tri)
                ns.append(nn)
        return CornerNormalMesh(np.asarray(tris), np.asarray(ns), name=name or Path(path).stem)


def weld_positions(mesh: CornerNormalMesh, tol: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    """Return unique geometric vertices and a (F,3) index array.

    Normals are deliberately not welded.  This supports the case where a single
    geometric node has multiple corner normals in different incident triangles.
    """
    pts = mesh.triangles.reshape(-1, 3)
    key = np.round(pts / tol).astype(np.int64)
    table: dict[tuple[int, int, int], int] = {}
    unique = []
    idx = np.empty((pts.shape[0],), dtype=np.int64)
    for i, k in enumerate(map(tuple, key)):
        if k not in table:
            table[k] = len(unique)
            unique.append(pts[i])
        idx[i] = table[k]
    return np.asarray(unique), idx.reshape(-1, 3)
