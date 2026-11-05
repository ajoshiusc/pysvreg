"""Pure Python command-line replacements for AIR helpers used by SVReg."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    from . import _deps  # noqa: F401
    import nibabel as nib
    from .native import air_warp_coord_volume, air_warp_points, _read_air_warp
except ImportError:
    import _deps  # noqa: F401
    import nibabel as nib
    from native import air_warp_coord_volume, air_warp_points, _read_air_warp


def main(argv: list[str] | None = None, *, command: str = "warp_points") -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 3:
        print(f"Usage: {command} .warp_file ucf_in ucf_out", file=sys.stderr)
        return 1
    warp_file, input_file, output_file = map(Path, argv)
    try:
        if not input_file.is_file():
            raise FileNotFoundError(f"missing UCF input file: {input_file}")
        if command == "warp_points":
            points = np.loadtxt(input_file, dtype=np.float64, ndmin=2)
            if points.shape[1] != 3:
                raise ValueError(f"UCF input must contain three coordinates per point: {input_file}")
            mapped = air_warp_points(points, warp_file, already_rounded=True)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            # AIR writes space-separated columns. Keep the byte-level text
            # format compatible as well as matching the mapped values.
            np.savetxt(output_file, mapped, fmt="%.6f", delimiter=" ")
            with output_file.open("a", encoding="ascii", newline="") as stream:
                stream.write("\n")
        elif command == "warp_coord_vol":
            image = nib.load(str(input_file))
            data = np.asanyarray(image.dataobj)
            if data.ndim == 4 and data.shape[3] == 1:
                data = data[..., 0]
            mapped = air_warp_coord_volume(data, warp_file)
            (_, _, standard_dimensions, standard_voxel_size, _, _) = _read_air_warp(warp_file)
            header = nib.AnalyzeHeader()
            output_data = mapped[..., np.newaxis]
            header.set_data_shape(output_data.shape)
            header.set_data_dtype(np.int16)
            header.set_zooms((*standard_voxel_size, 1.0))
            header["pixdim"][4] = 0.0
            result = nib.AnalyzeImage(output_data, None, header=header)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            nib.save(result, str(output_file))
        else:
            raise ValueError(f"unsupported AIR helper: {command}")
    except (OSError, ValueError) as exc:
        print(f"{command}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
