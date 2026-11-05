from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import nibabel as nib

from .air_cli import main


def _write_identity_warp(path: Path) -> None:
    """Create a minimal order-1 AIR warp with identity coordinates."""
    header = bytearray(368)
    struct.pack_into("<3i", header, 0, 1, 4, 3)
    struct.pack_into("<3i", header, 148, 10, 10, 10)
    struct.pack_into("<3d", header, 160, 1.0, 1.0, 1.0)
    struct.pack_into("<3i", header, 316, 10, 10, 10)
    struct.pack_into("<3d", header, 328, 1.0, 1.0, 1.0)
    coefficients = np.asarray(
        (
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype="<f8",
    )
    path.write_bytes(header + coefficients.tobytes())


class TestAirPointsOutput(unittest.TestCase):
    def test_output_text_matches_air_spacing_and_final_blank_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            warp = root / "identity.warp"
            source = root / "points.ucf"
            output = root / "mapped.ucf"
            _write_identity_warp(warp)
            source.write_text("1.234567 2.500000 3.750000\n", encoding="ascii")

            self.assertEqual(
                main([str(warp), str(source), str(output)], command="warp_points"), 0
            )
            self.assertEqual(
                output.read_bytes(),
                b"1.234567 2.500000 3.750000\n\n",
            )

    def test_coord_volume_identity_warp_matches_air_output_scale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            warp = root / "identity.warp"
            source = root / "source.nii.gz"
            output = root / "mapped.img"
            _write_identity_warp(warp)
            values = np.arange(1000, dtype=np.int16).reshape(10, 10, 10)
            values[0, 0, 0] = 20000  # AIR stores the doubled result as int16.
            values[-1, -1, -1] = -123
            nib.save(nib.Nifti1Image(values, np.eye(4)), source)

            self.assertEqual(
                main([str(warp), str(source), str(output)], command="warp_coord_vol"),
                0,
            )
            mapped = np.asanyarray(nib.load(output).dataobj).squeeze()
            expected = (np.maximum(values, 0).astype(np.int32) * 2).astype(np.int16)
            np.testing.assert_array_equal(mapped, expected)


if __name__ == "__main__":
    unittest.main()
