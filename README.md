# Feature-aware Contact-SDF Atlas demo

This package is a minimal, reproducible prototype for a **corner-normal mesh -> contact-suitable SDF atlas** pipeline.

It implements the idea discussed in the conversation:

- Input is a triangle mesh where each triangle stores **three vertex positions and three face-corner normals**.
- The same geometric vertex may have different normals in different triangles. These normals are preserved; they are not averaged.
- Ordinary scalar grid SDF is used only as a baseline.
- The proposed structure stores a local contact model per cell.  The adaptive version now supports **feature-specific refinement**:
  - smooth cell: `phi0 + n0^T dx + 1/2 dx^T H dx`;
  - sharp/ambiguous cell: multiple candidate normals, i.e. a normal-cone style contact record;
  - smooth cells stop at `max_depth`; edge/vertex/multi-normal-sector cells can refine deeper to `feature_max_depth`.
- Runtime query is cell lookup + polynomial evaluation, not online closest-point projection.

## What is generated

`scripts/generate_benchmarks.py` creates three analytic benchmark models:

1. `ellipsoid`: smooth surface with analytic corner normals.
2. `hex_prism`: flat faces and sharp edges; the same geometric nodes carry multiple face-corner normals.
3. `cone`: smooth side, sharp base rim and apex sectors.

Files are exported both as `.npz` and as a compact RMD-like text format:

```text
TRI x0 y0 z0 nx0 ny0 nz0 x1 y1 z1 nx1 ny1 nz1 x2 y2 z2 nx2 ny2 nz2
```

This is not the proprietary RecurDyn RMD grammar. It is a transparent corner-normal triangle block designed to match the mathematical structure of RMD-style mesh data.

## Run

```bash
pip install -r requirements.txt
python scripts/generate_benchmarks.py
python scripts/build_and_validate.py
python scripts/generate_paper_figures.py
pytest -q
```

Outputs are written under `results/`:

- `validation_summary.csv`
- `validation_summary.json`
- `*_uniform_atlas.npz`
- `*_feature_adaptive_atlas.npz`

## Baselines

The validation compares:

1. **Online projection baseline**: KD-tree candidate search + point-to-triangle closest projection + corner-normal interpolation.
2. **Scalar grid SDF baseline**: scalar sampled SDF with trilinear interpolation and trilinear-gradient normal.
3. **Uniform Contact-SDF Atlas**: precomputed cell records with local jets or multi-normal-cone records.
4. **Adaptive Contact-SDF Atlas**: dyadic/octal refinement of difficult cells while preserving O(1) runtime lookup through a finest-grid leaf table.

The atlas is expected to be faster than online projection because projection is moved offline.


## Adaptive and feature-specific refinement logic

`build_adaptive_contact_sdf_atlas(...)` starts from a coarse grid and recursively splits cells when one of the following offline checks fails:

- sampled gap error exceeds `gap_tol_factor * cell_size`;
- sampled best-candidate normal error exceeds `normal_tol_deg`;
- sampled points reveal competing normal sectors;
- the cell is a multi-normal-cone cell and has not reached the requested feature refinement level.

The key new parameters are:

```python
build_adaptive_contact_sdf_atlas(
    ...,
    max_depth=1,             # ordinary smooth leaves stop here
    feature_max_depth=3,     # edge/vertex/multi-sector leaves may go deeper
    sector_angle_deg=35.0,   # distinguish real normal sectors
    feature_enrichment=False # optional extra candidate-plane baking
)
```

A triangle edge/vertex hit alone is not treated as a physical sharp feature. The feature logic is driven by discontinuous or competing corner-normal sectors, which is the relevant information in the RMD-like corner-normal mesh.

A cheap signed-distance Lipschitz precheck skips sample projections for cells that are definitely outside the contact-relevant band.  Runtime evaluation performs only cell lookup and local polynomial/plane evaluation.

## Scope and limitations

This is a first verification package, not a complete production RMD parser. Once the exact RMD mesh-block grammar is available, the loader can be replaced while keeping the internal representation:

```python
triangles[f, i, :]       # triangle-local positions
corner_normals[f, i, :]  # triangle-local normals
```

The current atlas is intentionally second-order for smooth cells. Third-order tensors can be added later, but the second-order version is already sufficient to test the core claim: contact should use stored feature/normal/jet information, not gradients of a scalar interpolated SDF.


## Current validation result

After feature-specific refinement, `pytest -q` returns `9 passed`.  The sharp-body normal-cone hit rate improves from 0.677 to 0.877 on the hexagonal prism and from 0.559 to 0.831 on the cone, while runtime queries remain about 10x faster than online closest-point projection in this Python prototype.  See `VALIDATION_REPORT.md` for details.
