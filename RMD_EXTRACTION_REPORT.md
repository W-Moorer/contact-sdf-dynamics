# RecurDyn RMD surface extraction report

This project now includes a parser for the RecurDyn ASCII RMD surface blocks
found under `data/rmd`.  The extracted representation is exactly the current
backend input:

```text
triangles[f, i, :]       # three triangle-corner positions
corner_normals[f, i, :]  # three face-corner normals
```

## Supported surface blocks

The inspected files contain two compatible surface formats.

### `CSURFACE`

Observed in solid-patch records:

```text
CSURFACE / id
, NAME = '...'
, RM = marker_id
, TYPE = PATCH
, NO_PATCH = ...
, NO_NODE = ...
, PATCHTYPE = EACHNODENORMAL
, PATCH =
, n1, n2, n3, fnx, fny, fnz, n1x, n1y, n1z, n2x, n2y, n2z, n3x, n3y, n3z
, NODE =
, x, y, z
...
```

Extraction:

- triangle node ids are `n1,n2,n3`;
- `fnx,fny,fnz` is the patch/face normal;
- the last three triplets are the three corner normals;
- node ids are implicit 1-based indices into the following `NODE` list.

### `GGEOM`

Observed in triangle `GSURFACE` records:

```text
GGEOM / id
, NAME = '...'
, RM = marker_id
, SURF_TYPE = TRIANGLE
, NO_PATCH = ...
, NO_NODE = ...
, PATCHES =
3, n1, n2, n3, fnx, fny, fnz, n1x, n1y, n1z, n2x, n2y, n2z, n3x, n3y, n3z
, NODES =
, x, y, z
...
```

Extraction is the same as `CSURFACE`, except the leading `3` is the triangle
node count.

## Coordinate frames

Each surface references a marker through `RM`.  The node coordinates are stored
in that reference-marker frame.  The parser supports three output frames:

- `marker`: raw surface-reference-marker coordinates;
- `part`: apply the referenced marker transform, `x_part = QP_RM + R_RM x`;
- `world`: additionally apply the owning part transform, `x_world = QG_part + R_part x_part`.

For isolated atlas construction on a single body surface, `marker` or `part`
coordinates are sufficient as long as all points and normals are in the same
frame.  For multi-body contact-pair reconstruction from an assembled RMD model,
`world` is the appropriate frame, but the RecurDyn Euler convention should be
verified for nonzero `REULER` records before relying on global placement.

Normals are rotated with the same frame transform and then normalized.  They are
not averaged across welded vertices.

## Parsed inventory

Command used:

```powershell
python scripts\analyze_rmd_surfaces.py --input data\rmd --frame marker --export-dir results\rmd_extracted --out results\rmd_surface_inventory.csv
```

Summary:

| RMD file | surface blocks | total patches | note |
| --- | ---: | ---: | --- |
| `groove_cube.rmd` | 2 | 38,852 | two `CSURFACE` solid-patch bodies |
| `groove_sphere.rmd` | 2 | 23,102 | one grooved body and one ellipsoid |
| `jiandanjiaolian.rmd` | 2 | 45,976 | two `GGEOM` gear surfaces |
| `multibody_contact.rmd` | 7 | 75,106 | two complex solid patches and five ellipsoids |
| `rev_clearance_joint.rmd` | 2 | 6,542 | clearance-joint surfaces |

Overall inventory:

- surface blocks parsed: 15;
- triangle patches parsed: 183,368;
- welded vertices: 91,710;
- welded vertices with more than one corner-normal sector at 5 deg clustering: 11,769;
- patch formats found:
  - `CSURFACE_TRIANGLE_FACE_AND_CORNER_NORMALS`: 11 surfaces;
  - `GGEOM_TRIANGLE_COUNT_FACE_AND_CORNER_NORMALS`: 4 surfaces.

The nonzero multi-sector counts in grooved bodies, gears and clearance-joint
surfaces confirm that these RMD files preserve the corner-normal discontinuities
needed by the normal-sector atlas.  Ellipsoid surfaces report zero multi-sector
vertices, as expected for smooth geometry.

## Implementation status

- Parser module: `contact_sdf/rmd_parser.py`
- Inventory/export script: `scripts/analyze_rmd_surfaces.py`
- Tests: `tests/test_rmd_parser.py`
- Converted local outputs: `results/rmd_extracted/*.npz` and `*.cnmesh`

The converted `.npz` files can be loaded as `CornerNormalMesh` objects and used
directly by the existing projection, grid-SDF and atlas builders.
