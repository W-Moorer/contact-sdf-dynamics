from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contact_sdf.shapes import (
    cone_mesh,
    cylinder_mesh,
    ellipsoid_mesh,
    prism_mesh,
    sphere_mesh,
    torus_mesh,
    wedge_mesh,
)

OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)

meshes = [
    ellipsoid_mesh(n_lon=40, n_lat=20),
    sphere_mesh(n_lon=32, n_lat=16),
    prism_mesh(n_sides=6),
    wedge_mesh(angle_deg=90.0),
    cylinder_mesh(n_seg=32),
    cone_mesh(n_seg=48),
    torus_mesh(n_major=32, n_minor=12),
]
for m in meshes:
    m.save_npz(OUT / f"{m.name}.npz")
    m.save_rmd_like(OUT / f"{m.name}.cnmesh")
    print(f"wrote {m.name}: {m.n_faces} faces")
