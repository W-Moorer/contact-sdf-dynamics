from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contact_sdf.atlas import AdaptiveContactSDFAtlas, build_adaptive_contact_sdf_atlas  # noqa: E402
from contact_sdf.mesh_format import CornerNormalMesh  # noqa: E402
from contact_sdf.projection import MeshProjector  # noqa: E402
from contact_sdf.rmd_parser import load_rmd_model  # noqa: E402
from scripts.generate_surface_visuals import (  # noqa: E402
    cubic_bbox,
    mesh_vertices_faces,
    render_surface,
    zero_gap_surface_from_mesh,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MESH_DIR = ROOT / "results" / "rmd_extracted"
DEFAULT_RMD_DIR = ROOT / "data" / "rmd"
DEFAULT_OUT = ROOT / "results" / "rmd_surface_visuals"

METHODS = ("original", "feature_adaptive")


@dataclass(frozen=True)
class NamedMesh:
    stem: str
    mesh: CornerNormalMesh
    source: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render RecurDyn RMD surfaces as independent original and "
            "feature-adaptive zero-level surface images."
        )
    )
    parser.add_argument(
        "--mesh",
        nargs="*",
        type=Path,
        default=None,
        help="Specific extracted .npz CornerNormalMesh files to render.",
    )
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=DEFAULT_MESH_DIR,
        help="Directory containing extracted .npz meshes.",
    )
    parser.add_argument(
        "--rmd",
        nargs="*",
        type=Path,
        default=None,
        help="Specific .rmd files to parse and render directly.",
    )
    parser.add_argument(
        "--rmd-dir",
        type=Path,
        default=None,
        help="Directory of .rmd files to parse when extracted meshes are unavailable.",
    )
    parser.add_argument("--frame", choices=["marker", "part", "world"], default="marker")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS),
        choices=METHODS,
        help="Only original and feature_adaptive are supported for RMD visualization.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--base-resolution", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--feature-max-depth", type=int, default=3)
    parser.add_argument(
        "--feature-active-tol-factor",
        type=float,
        default=0.5,
        help="Cell-size factor used to discover competing feature sectors offline.",
    )
    parser.add_argument("--probe-resolution", type=int, default=17)
    parser.add_argument(
        "--anchor-tol-factor",
        type=float,
        default=0.25,
        help="Keep original corners whose atlas gap is within this fraction of the normal search radius.",
    )
    parser.add_argument("--projector-k", type=int, default=96)
    parser.add_argument("--max-surfaces", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Regenerate images and atlas caches.")
    return parser.parse_args()


def safe_stem(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "surface"


def iter_rmd_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.rmd"))


def load_mesh_file(path: Path) -> NamedMesh:
    mesh = CornerNormalMesh.load_npz(path)
    return NamedMesh(stem=safe_stem(path.stem), mesh=mesh, source=path)


def load_rmd_file(path: Path, frame: str) -> list[NamedMesh]:
    model = load_rmd_model(path, frame=frame)
    out: list[NamedMesh] = []
    for surface in model.surfaces:
        stem = safe_stem(f"{path.stem}_{surface.surface_id}_{surface.mesh.name}")
        out.append(NamedMesh(stem=stem, mesh=surface.mesh, source=path))
    return out


def collect_meshes(args: argparse.Namespace) -> list[NamedMesh]:
    if args.mesh:
        meshes = [load_mesh_file(path) for path in args.mesh]
    elif args.rmd or args.rmd_dir is not None:
        meshes = []
        for path in args.rmd or []:
            meshes.extend(load_rmd_file(path, args.frame))
        if args.rmd_dir is not None:
            for path in iter_rmd_files(args.rmd_dir):
                meshes.extend(load_rmd_file(path, args.frame))
    elif args.mesh_dir.exists():
        meshes = [load_mesh_file(path) for path in sorted(args.mesh_dir.glob("*.npz"))]
    else:
        rmd_input = args.rmd_dir or DEFAULT_RMD_DIR
        meshes = []
        for path in iter_rmd_files(rmd_input):
            meshes.extend(load_rmd_file(path, args.frame))

    if args.max_surfaces is not None:
        meshes = meshes[: args.max_surfaces]
    if not meshes:
        raise SystemExit("No RMD surfaces found. Provide --mesh, --mesh-dir, --rmd, or --rmd-dir.")
    return meshes


def atlas_cache_path(args: argparse.Namespace, stem: str) -> Path:
    return (
        args.out_dir
        / "cache"
        / (
            f"{stem}_feature_adaptive"
            f"_b{args.base_resolution}_d{args.max_depth}_f{args.feature_max_depth}"
            f"_a{args.feature_active_tol_factor:g}.npz"
        )
    )


def load_or_build_adaptive_atlas(
    args: argparse.Namespace,
    named: NamedMesh,
    projector: MeshProjector,
    bbox: tuple[np.ndarray, np.ndarray],
) -> AdaptiveContactSDFAtlas:
    path = atlas_cache_path(args, named.stem)
    if path.exists() and not args.force:
        return AdaptiveContactSDFAtlas.load_npz(path)

    extent = bbox[1] - bbox[0]
    refine_band = 0.09 * float(np.max(extent))
    atlas = build_adaptive_contact_sdf_atlas(
        projector,
        bbox,
        base_resolution=args.base_resolution,
        max_depth=args.max_depth,
        feature_max_depth=args.feature_max_depth,
        active_tol=0.01,
        sector_angle_deg=35.0,
        max_candidates=16,
        gap_tol_factor=0.08,
        normal_tol_deg=6.0,
        feature_normal_tol_deg=5.0,
        feature_enrichment=True,
        feature_active_tol_factor=args.feature_active_tol_factor,
        hessian_for_smooth=True,
        multi_if_feature=False,
        refine_band=refine_band,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    atlas.save_npz(path)
    return atlas


def adaptive_visual_phi(atlas: AdaptiveContactSDFAtlas, points: np.ndarray) -> np.ndarray:
    """Scalarize adaptive atlas records for zero-surface visualization.

    Solver queries keep the primary compact fields plus candidate counts.
    Rendering needs one scalar value, so multi-sector leaves use the candidate
    gap plane with the smallest absolute value at each query point.
    """
    pts = np.asarray(points, dtype=float)
    compact = atlas.eval_compact(pts)
    phi = compact.phi.copy()
    multi = compact.candidate_count > 1
    if not np.any(multi):
        return phi

    for lid in np.unique(compact.leaf_id[multi]):
        lid = int(lid)
        mask = compact.leaf_id == lid
        count = int(atlas.cand_count[lid])
        if count <= 1:
            continue
        dx = pts[mask] - atlas.leaf_center[lid]
        normals = atlas.cand_normal0[lid, :count]
        candidate_phi = atlas.cand_phi0[lid, :count][None, :] + dx @ normals.T
        best = np.argmin(np.abs(candidate_phi), axis=1)
        phi[mask] = candidate_phi[np.arange(candidate_phi.shape[0]), best]
    return phi


def surface_for_method(
    method: str,
    args: argparse.Namespace,
    named: NamedMesh,
    projector: MeshProjector,
    bbox: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if method == "original":
        return mesh_vertices_faces(named.mesh)
    if method == "feature_adaptive":
        atlas = load_or_build_adaptive_atlas(args, named, projector, bbox)
        clip_band = 1.35 * float(np.max(bbox[1] - bbox[0])) / float(args.probe_resolution - 1)
        return zero_gap_surface_from_mesh(
            named.mesh,
            lambda pts: adaptive_visual_phi(atlas, pts),
            search_radius=0.55 * clip_band,
            anchor_tol=args.anchor_tol_factor * 0.55 * clip_band,
        )
    raise ValueError(f"Unknown method: {method}")


def output_is_current(args: argparse.Namespace, named: NamedMesh, method: str, out: Path) -> bool:
    if not (out.with_suffix(".png").exists() and out.with_suffix(".pdf").exists()):
        return False
    if method != "feature_adaptive":
        return True
    return atlas_cache_path(args, named.stem).exists()


def main() -> None:
    args = parse_args()
    meshes = collect_meshes(args)
    total = len(meshes) * len(args.methods)
    done = 0
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"rendering {len(meshes)} RMD surface(s) to {args.out_dir}", flush=True)
    for named in meshes:
        bbox = cubic_bbox(named.mesh)
        t0 = time.perf_counter()
        projector = MeshProjector(named.mesh, k=min(args.projector_k, named.mesh.n_faces))
        print(
            f"[{named.stem}] {named.mesh.n_faces} faces, projector {time.perf_counter() - t0:.2f}s",
            flush=True,
        )
        for method in args.methods:
            done += 1
            out = args.out_dir / f"rmd_surface_{named.stem}_{method}"
            if output_is_current(args, named, method, out) and not args.force:
                print(f"[{done}/{total}] keep {out.name}", flush=True)
                continue
            t1 = time.perf_counter()
            vertices, faces = surface_for_method(method, args, named, projector, bbox)
            render_surface(vertices, faces, method, out)
            print(
                f"[{done}/{total}] wrote {out.name}.png/.pdf "
                f"({len(vertices)} vertices, {len(faces)} faces, {time.perf_counter() - t1:.1f}s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
