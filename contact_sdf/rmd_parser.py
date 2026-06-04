"""Parser for RecurDyn ASCII RMD surface mesh blocks.

The parser targets the surface records observed in exported RMD model files:
``CSURFACE`` blocks with ``PATCHTYPE = EACHNODENORMAL`` and ``GGEOM`` blocks
with ``SURF_TYPE = TRIANGLE``.  Both formats store triangle node indices and
face-corner normals in each patch row, followed by an implicit 1-based node
coordinate list.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re

import numpy as np

from .mesh_format import CornerNormalMesh


_TOP_RECORD_RE = re.compile(r"^([A-Z][A-Z0-9_]*)\s*/\s*([0-9]+)")
_SURFACE_RE = re.compile(r"^(CSURFACE|GGEOM)\s*/\s*([0-9]+)\b")


@dataclass(frozen=True)
class RmdPart:
    part_id: int
    name: str = ""
    qg: np.ndarray | None = None
    reuler: np.ndarray | None = None


@dataclass(frozen=True)
class RmdMarker:
    marker_id: int
    name: str = ""
    part_id: int | None = None
    qp: np.ndarray | None = None
    reuler: np.ndarray | None = None


@dataclass
class RmdSurface:
    surface_id: int
    kind: str
    name: str
    rm_marker_id: int | None
    mesh: CornerNormalMesh
    raw_patch_count: int | None
    raw_node_count: int | None
    patch_format: str
    source_path: Path
    start_line: int


@dataclass
class RmdModel:
    path: Path
    parts: dict[int, RmdPart]
    markers: dict[int, RmdMarker]
    surfaces: list[RmdSurface]


def _parse_numeric_token(token: str) -> float:
    token = token.strip()
    if token.endswith(("D", "d")):
        return math.radians(float(token[:-1]))
    return float(token)


def _parse_number_list(line: str) -> list[float] | None:
    s = line.strip()
    if not s or s.startswith("!"):
        return None
    if s.startswith(","):
        s = s[1:].strip()
    if not s or not re.match(r"^[+\-0-9.]", s):
        return None
    parts = [p.strip() for p in s.split(",")]
    if any(p == "" for p in parts):
        return None
    try:
        return [_parse_numeric_token(p) for p in parts]
    except ValueError:
        return None


def _value_after_equals(line: str) -> str | None:
    if "=" not in line:
        return None
    return line.split("=", 1)[1].strip()


def _parse_name(line: str) -> str | None:
    value = _value_after_equals(line)
    if value is None:
        return None
    match = re.search(r"'([^']*)'", value)
    return match.group(1) if match else value.strip().strip(",")


def _parse_int_after_key(line: str, key: str) -> int | None:
    match = re.search(rf"\b{re.escape(key)}\s*=\s*([0-9]+)", line, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _parse_vector_assignment(line: str) -> np.ndarray | None:
    value = _value_after_equals(line)
    if value is None:
        return None
    nums = _parse_number_list(value)
    if nums is None or len(nums) < 3:
        return None
    return np.asarray(nums[:3], dtype=float)


def _rotation_xyz(angles: np.ndarray | None) -> np.ndarray:
    if angles is None:
        return np.eye(3)
    ax, ay, az = [float(v) for v in angles[:3]]
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def _top_record_name(line: str) -> str | None:
    match = _TOP_RECORD_RE.match(line.strip())
    return match.group(1) if match else None


def _read_parts_and_markers(lines: list[str]) -> tuple[dict[int, RmdPart], dict[int, RmdMarker]]:
    parts: dict[int, dict] = {}
    markers: dict[int, dict] = {}
    current_kind: str | None = None
    current_id: int | None = None

    for raw in lines:
        line = raw.strip()
        match = _TOP_RECORD_RE.match(line)
        if match:
            rec, rec_id = match.group(1), int(match.group(2))
            if rec == "PART":
                current_kind, current_id = "part", rec_id
                parts.setdefault(rec_id, {"part_id": rec_id})
            elif rec == "MARKER":
                current_kind, current_id = "marker", rec_id
                markers.setdefault(rec_id, {"marker_id": rec_id})
            else:
                current_kind, current_id = None, None
            continue

        if current_kind == "part" and current_id is not None:
            dst = parts[current_id]
            if ", NAME" in line:
                dst["name"] = _parse_name(line) or ""
            elif ", QG" in line:
                dst["qg"] = _parse_vector_assignment(line)
            elif ", REULER" in line:
                dst["reuler"] = _parse_vector_assignment(line)
        elif current_kind == "marker" and current_id is not None:
            dst = markers[current_id]
            if ", NAME" in line:
                dst["name"] = _parse_name(line) or ""
            elif ", PART" in line:
                value = _value_after_equals(line)
                if value is not None:
                    dst["part_id"] = int(float(value.split(",", 1)[0].strip()))
            elif ", QP" in line:
                dst["qp"] = _parse_vector_assignment(line)
            elif ", REULER" in line:
                dst["reuler"] = _parse_vector_assignment(line)

    return (
        {pid: RmdPart(**data) for pid, data in parts.items()},
        {mid: RmdMarker(**data) for mid, data in markers.items()},
    )


def _surface_block_end(lines: list[str], start: int) -> int:
    for i in range(start + 1, len(lines)):
        if _TOP_RECORD_RE.match(lines[i].strip()):
            return i
    return len(lines)


def _parse_patch_row(nums: list[float], kind: str) -> tuple[np.ndarray, np.ndarray, str]:
    if kind == "GGEOM" and len(nums) >= 16 and int(round(nums[0])) == 3:
        node_ids = np.asarray(nums[1:4], dtype=np.int64)
        normal_values = nums[4:]
        patch_format = "GGEOM_TRIANGLE_COUNT_FACE_AND_CORNER_NORMALS"
    else:
        node_ids = np.asarray(nums[:3], dtype=np.int64)
        normal_values = nums[3:]
        patch_format = "CSURFACE_TRIANGLE_FACE_AND_CORNER_NORMALS"

    if len(normal_values) >= 12:
        corner = np.asarray(normal_values[-9:], dtype=float).reshape(3, 3)
    elif len(normal_values) == 9:
        corner = np.asarray(normal_values, dtype=float).reshape(3, 3)
        patch_format += "_NO_FACE_NORMAL"
    elif len(normal_values) == 3:
        corner = np.tile(np.asarray(normal_values, dtype=float), (3, 1))
        patch_format += "_FACE_NORMAL_ONLY"
    else:
        raise ValueError(f"Unsupported patch row with {len(nums)} numeric fields")
    return node_ids, corner, patch_format


def _parse_surface_block(
    lines: list[str],
    start: int,
    end: int,
    path: Path,
    parts: dict[int, RmdPart],
    markers: dict[int, RmdMarker],
    frame: str,
) -> RmdSurface:
    header = lines[start:end]
    match = _SURFACE_RE.match(lines[start].strip())
    if not match:
        raise ValueError(f"Not a surface record at line {start + 1}")
    kind = match.group(1)
    surface_id = int(match.group(2))
    name = ""
    rm_marker_id: int | None = None
    raw_patch_count: int | None = None
    raw_node_count: int | None = None
    patch_line: int | None = None
    node_line: int | None = None

    for i, raw in enumerate(header, start=start):
        line = raw.strip()
        if ", NAME" in line and not name:
            name = _parse_name(line) or ""
        elif ", RM" in line:
            value = _value_after_equals(line)
            if value is not None:
                rm_marker_id = int(float(value.split(",", 1)[0].strip()))
        raw_patch_count = _parse_int_after_key(line, "NO_PATCH") or raw_patch_count
        raw_node_count = _parse_int_after_key(line, "NO_NODE") or raw_node_count
        upper = line.upper()
        if upper.startswith(", PATCH =") or upper.startswith(", PATCHES ="):
            patch_line = i
        elif upper.startswith(", NODE =") or upper.startswith(", NODES ="):
            node_line = i

    if patch_line is None or node_line is None or node_line <= patch_line:
        raise ValueError(f"Surface {surface_id} in {path} lacks ordered PATCH/NODE sections")

    patch_ids: list[np.ndarray] = []
    patch_normals: list[np.ndarray] = []
    patch_format = ""
    for raw in lines[patch_line + 1:node_line]:
        nums = _parse_number_list(raw)
        if nums is None:
            continue
        ids, corner, fmt = _parse_patch_row(nums, kind)
        patch_ids.append(ids)
        patch_normals.append(corner)
        patch_format = fmt

    nodes: list[list[float]] = []
    for raw in lines[node_line + 1:end]:
        if _top_record_name(raw) is not None:
            break
        nums = _parse_number_list(raw)
        if nums is None:
            continue
        if len(nums) >= 3:
            nodes.append(nums[:3])

    node_array = np.asarray(nodes, dtype=float)
    ids_array = np.asarray(patch_ids, dtype=np.int64)
    normals = _normalize(np.asarray(patch_normals, dtype=float))
    if node_array.ndim != 2 or node_array.shape[1] != 3:
        raise ValueError(f"Surface {surface_id} in {path} has no node coordinate list")
    if len(ids_array) == 0:
        raise ValueError(f"Surface {surface_id} in {path} has no patch list")
    if raw_patch_count is not None and len(ids_array) != raw_patch_count:
        raise ValueError(
            f"Surface {surface_id} in {path} declares {raw_patch_count} patches "
            f"but {len(ids_array)} were parsed"
        )
    if raw_node_count is not None and len(node_array) != raw_node_count:
        raise ValueError(
            f"Surface {surface_id} in {path} declares {raw_node_count} nodes "
            f"but {len(node_array)} were parsed"
        )
    if np.min(ids_array) < 1 or np.max(ids_array) > len(node_array):
        raise ValueError(f"Surface {surface_id} in {path} references node ids outside 1..{len(node_array)}")

    triangles = node_array[ids_array - 1]
    triangles, normals = _transform_surface(triangles, normals, rm_marker_id, parts, markers, frame)
    mesh_name = _safe_mesh_name(name or f"{kind}_{surface_id}")
    mesh = CornerNormalMesh(
        triangles=triangles,
        corner_normals=normals,
        name=mesh_name,
        tags={
            "source": "recurdyn_rmd",
            "source_file": str(path),
            "surface_id": surface_id,
            "surface_kind": kind,
            "surface_name": name,
            "rm_marker_id": rm_marker_id,
            "frame": frame,
            "patch_format": patch_format,
        },
    )
    return RmdSurface(
        surface_id=surface_id,
        kind=kind,
        name=name,
        rm_marker_id=rm_marker_id,
        mesh=mesh,
        raw_patch_count=raw_patch_count,
        raw_node_count=raw_node_count,
        patch_format=patch_format,
        source_path=path,
        start_line=start + 1,
    )


def _safe_mesh_name(name: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip("#' "))
    return out.strip("_") or "rmd_surface"


def _transform_surface(
    triangles: np.ndarray,
    normals: np.ndarray,
    rm_marker_id: int | None,
    parts: dict[int, RmdPart],
    markers: dict[int, RmdMarker],
    frame: str,
) -> tuple[np.ndarray, np.ndarray]:
    frame = frame.lower()
    if frame not in {"marker", "part", "world"}:
        raise ValueError("frame must be one of: marker, part, world")
    if frame == "marker" or rm_marker_id is None or rm_marker_id not in markers:
        return triangles, normals

    marker = markers[rm_marker_id]
    r_marker = _rotation_xyz(marker.reuler)
    t_marker = marker.qp if marker.qp is not None else np.zeros(3)
    tri = triangles @ r_marker.T + t_marker[None, None, :]
    nrm = normals @ r_marker.T

    if frame == "world" and marker.part_id is not None and marker.part_id in parts:
        part = parts[marker.part_id]
        r_part = _rotation_xyz(part.reuler)
        t_part = part.qg if part.qg is not None else np.zeros(3)
        tri = tri @ r_part.T + t_part[None, None, :]
        nrm = nrm @ r_part.T
    return tri, _normalize(nrm)


def load_rmd_model(path: str | Path, frame: str = "marker") -> RmdModel:
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    parts, markers = _read_parts_and_markers(lines)
    surfaces: list[RmdSurface] = []
    for i, raw in enumerate(lines):
        if _SURFACE_RE.match(raw.strip()):
            end = _surface_block_end(lines, i)
            surfaces.append(_parse_surface_block(lines, i, end, path, parts, markers, frame=frame))
    return RmdModel(path=path, parts=parts, markers=markers, surfaces=surfaces)


def load_rmd_surfaces(path: str | Path, frame: str = "marker") -> list[CornerNormalMesh]:
    return [surface.mesh for surface in load_rmd_model(path, frame=frame).surfaces]
