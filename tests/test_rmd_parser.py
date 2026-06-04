from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from contact_sdf.rmd_parser import load_rmd_model


def test_parse_csurface_each_node_normal(tmp_path: Path):
    path = tmp_path / "tiny_csurface.rmd"
    path.write_text(
        "\n".join([
            "MARKER / 5",
            ", NAME = 'Body.Surface.Ref'",
            ", PART = 1",
            ", QP = 10 , 0 , 0",
            ", REULER = 0 , 0 , 0",
            "CSURFACE / 2",
            ", NAME = '##SOLID##_Body.Surface'",
            ", RM = 5",
            ", TYPE = PATCH",
            ", NO_PATCH = 1 , NO_LINE = 3 , NO_NODE = 3",
            ", PATCHTYPE = EACHNODENORMAL",
            ", PATCH = ",
            ", 1, 2, 3, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1",
            ", NODE = ",
            ", 0, 0, 0",
            ", 1, 0, 0",
            ", 0, 1, 0",
        ]),
        encoding="utf-8",
    )
    model = load_rmd_model(path, frame="marker")
    assert len(model.surfaces) == 1
    mesh = model.surfaces[0].mesh
    np.testing.assert_allclose(mesh.triangles[0], [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    np.testing.assert_allclose(mesh.corner_normals[0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    mesh_part = load_rmd_model(path, frame="part").surfaces[0].mesh
    np.testing.assert_allclose(mesh_part.triangles[0], [[10, 0, 0], [11, 0, 0], [10, 1, 0]])


def test_parse_ggeom_triangle_patch(tmp_path: Path):
    path = tmp_path / "tiny_ggeom.rmd"
    path.write_text(
        "\n".join([
            "GGEOM / 3",
            ", NAME = '##GSURFACE##_Body.Surface'",
            ", GEOM_TYPE = SURFACE",
            ", RM = 6",
            ", SURF_TYPE = TRIANGLE",
            ", NO_PATCH = 1",
            ", NO_NODE = 3",
            ", PATCHES = ",
            "3, 2, 3, 1, 0, 0, 1, -1, 0, 0, 0, -1, 0, 0, 0, -1",
            ", NODES = ",
            ", 0, 0, 0",
            ", 1, 0, 0",
            ", 0, 1, 0",
        ]),
        encoding="utf-8",
    )
    model = load_rmd_model(path)
    assert len(model.surfaces) == 1
    surface = model.surfaces[0]
    assert surface.patch_format == "GGEOM_TRIANGLE_COUNT_FACE_AND_CORNER_NORMALS"
    np.testing.assert_allclose(surface.mesh.triangles[0], [[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    np.testing.assert_allclose(surface.mesh.corner_normals[0], [[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
