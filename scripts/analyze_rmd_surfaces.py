from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contact_sdf.mesh_format import CornerNormalMesh, weld_positions  # noqa: E402
from contact_sdf.rmd_parser import RmdSurface, load_rmd_model  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "rmd"
DEFAULT_OUT = ROOT / "results" / "rmd_surface_inventory.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and optionally export RecurDyn RMD surface meshes.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="RMD file or directory.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="CSV inventory path.")
    parser.add_argument("--frame", choices=["marker", "part", "world"], default="marker")
    parser.add_argument("--export-dir", type=Path, default=None, help="Optional directory for extracted .npz/.cnmesh.")
    parser.add_argument("--sector-angle-deg", type=float, default=5.0)
    return parser.parse_args()


def iter_rmd_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.rmd"))


def safe_stem(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "surface"


def normal_sector_stats(mesh: CornerNormalMesh, angle_deg: float) -> dict[str, float | int]:
    verts, corner_vertex_ids = weld_positions(mesh, tol=1e-8)
    normals = mesh.corner_normals.reshape(-1, 3)
    flat_ids = corner_vertex_ids.reshape(-1)
    cos_thr = float(np.cos(np.radians(angle_deg)))
    sector_counts = []
    for vid in range(len(verts)):
        ids = np.where(flat_ids == vid)[0]
        unique: list[np.ndarray] = []
        for idx in ids:
            n = normals[idx]
            if not any(float(np.dot(n, u)) > cos_thr for u in unique):
                unique.append(n)
        sector_counts.append(len(unique))
    sector_counts_arr = np.asarray(sector_counts, dtype=int) if sector_counts else np.zeros(0, dtype=int)
    return {
        "welded_vertices": int(len(verts)),
        "multi_sector_vertices": int(np.sum(sector_counts_arr > 1)),
        "max_sectors_per_vertex": int(np.max(sector_counts_arr)) if len(sector_counts_arr) else 0,
        "multi_sector_vertex_rate": float(np.mean(sector_counts_arr > 1)) if len(sector_counts_arr) else 0.0,
    }


def surface_row(path: Path, surface: RmdSurface, angle_deg: float) -> dict[str, object]:
    mesh = surface.mesh
    bbox_lo, bbox_hi = mesh.bbox(pad=0.0)
    norm_len = np.linalg.norm(mesh.corner_normals.reshape(-1, 3), axis=1)
    stats = normal_sector_stats(mesh, angle_deg)
    return {
        "file": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "surface_id": surface.surface_id,
        "kind": surface.kind,
        "name": surface.name,
        "rm_marker_id": surface.rm_marker_id,
        "start_line": surface.start_line,
        "patch_format": surface.patch_format,
        "patches_declared": surface.raw_patch_count,
        "patches_parsed": mesh.n_faces,
        "nodes_declared": surface.raw_node_count,
        **stats,
        "normal_length_min": float(np.min(norm_len)),
        "normal_length_max": float(np.max(norm_len)),
        "bbox_min_x": float(bbox_lo[0]),
        "bbox_min_y": float(bbox_lo[1]),
        "bbox_min_z": float(bbox_lo[2]),
        "bbox_max_x": float(bbox_hi[0]),
        "bbox_max_y": float(bbox_hi[1]),
        "bbox_max_z": float(bbox_hi[2]),
    }


def export_surface(export_dir: Path, source: Path, surface: RmdSurface) -> None:
    mesh = surface.mesh
    stem = safe_stem(f"{source.stem}_{surface.surface_id}_{mesh.name}")
    export_dir.mkdir(parents=True, exist_ok=True)
    mesh.save_npz(export_dir / f"{stem}.npz")
    mesh.save_rmd_like(export_dir / f"{stem}.cnmesh")


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    files = iter_rmd_files(args.input)
    if not files:
        raise SystemExit(f"No .rmd files found under {args.input}")

    for path in files:
        model = load_rmd_model(path, frame=args.frame)
        print(f"{path}: {len(model.surfaces)} surface block(s)")
        for surface in model.surfaces:
            row = surface_row(path, surface, args.sector_angle_deg)
            rows.append(row)
            print(
                f"  {surface.kind}/{surface.surface_id}: "
                f"{surface.name or surface.mesh.name}, patches={surface.mesh.n_faces}, "
                f"nodes={row['welded_vertices']}, multi-sector vertices={row['multi_sector_vertices']}"
            )
            if args.export_dir is not None:
                export_surface(args.export_dir, path, surface)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")
    if args.export_dir is not None:
        print(f"exported meshes to {args.export_dir}")


if __name__ == "__main__":
    main()
