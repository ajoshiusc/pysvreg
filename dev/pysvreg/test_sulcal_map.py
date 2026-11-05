from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import nibabel as nib
import numpy as np
from unittest import mock

from .dfsio import writedfs
from . import native
from .sulcal_map import (
    _curve_labels,
    _map_volume,
    _source_style_curve_distance,
    _source_style_curve_distances,
    _source_topology_labels,
)


class SulcalMapTests(unittest.TestCase):
    def test_batched_relaxation_is_identical_to_independent_source_sweeps(self) -> None:
        surface = SimpleNamespace(
            vertices=np.asarray(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64
            ),
            faces=np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int32),
        )
        adjacency = native.vertices_connectivity_fast(surface)[1]
        groups = [np.asarray([0]), np.asarray([3])]
        batched = _source_style_curve_distances(surface.vertices, adjacency, groups)
        independent = np.column_stack(
            [
                _source_style_curve_distance(surface.vertices, adjacency, seeds)
                for seeds in groups
            ]
        )
        np.testing.assert_array_equal(batched, independent)

    def test_curve_file_maps_to_source_encoded_surface_labels(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol.xml"
            protocol.write_text("<protocol/>", encoding="utf-8")
            curve_file = root / "curves.dfc"
            vertices = np.asarray(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32
            )
            surface = SimpleNamespace(
                vertices=vertices,
                faces=np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int32),
                labels=np.ones(4, dtype=np.uint16),
            )
            native.writedfc(str(curve_file), [vertices[[0]]], str(protocol))
            with mock.patch.object(
                native, "patchcurvature_mean", return_value=np.ones(4)
            ):
                labels, attributes, curves = _curve_labels(surface, curve_file, "left", 2.0)

        np.testing.assert_array_equal(labels, np.full(4, 4001, dtype=np.uint16))
        np.testing.assert_array_equal(attributes, np.ones(4, dtype=bool))
        self.assertEqual(len(curves), 1)

    def test_surface_labels_use_face_median_and_retain_curve_component(self) -> None:
        surface = SimpleNamespace(
            vertices=np.asarray(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32
            ),
            faces=np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int32),
            labels=np.ones(4, dtype=np.uint16),
        )
        result = _source_topology_labels(
            surface,
            curve_ids=np.asarray([1, 1, 0, 0]),
            distance=np.zeros(4),
            attributes=np.ones(4, dtype=bool),
            curvature_positive=np.ones(4, dtype=bool),
            source_labels=surface.labels,
            hemi="left",
            dist_threshold=10.0,
        )
        np.testing.assert_array_equal(result, np.asarray([4001, 4001, 4001, 0], dtype=np.uint16))

    def test_volume_map_preserves_input_label_datatype_and_sulcal_codes(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = root / "subject"
            data = np.zeros((4, 4, 4), dtype=np.int16)
            data[0, 0, 0] = 110
            data[3, 3, 3] = 2110
            image = nib.Nifti1Image(data, np.eye(4))
            image.header.set_data_dtype(np.int16)
            nib.save(image, f"{prefix}.svreg.label.nii.gz")

            for hemi, point, label in (
                ("left", (0, 0, 0), 4001),
                ("right", (3, 3, 3), 4000),
            ):
                for layer in ("mid.cortex.sulci", "inner.cortex.svreg", "pial.cortex.svreg"):
                    surface = SimpleNamespace(
                        vertices=np.asarray([point], dtype=np.float32),
                        faces=np.zeros((0, 3), dtype=np.int32),
                        labels=np.asarray([label], dtype=np.uint16),
                    )
                    writedfs(f"{prefix}.{hemi}.{layer}.dfs", surface)

            _map_volume(prefix)

            result = nib.load(f"{prefix}.sulci.label.nii.gz")
            labels = np.asanyarray(result.dataobj)
            self.assertEqual(result.get_data_dtype(), np.dtype(np.int16))
            self.assertEqual(int(labels[0, 0, 0]), 4001)
            self.assertEqual(int(labels[3, 3, 3]), 4000)
            self.assertEqual(int(labels[1, 1, 1]), 0)


if __name__ == "__main__":
    unittest.main()
