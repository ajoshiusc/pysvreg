from __future__ import annotations

import os
import sys
from pathlib import Path
import shutil
import traceback

import numpy as np

try:
    from . import _deps  # noqa: F401
except ImportError:
    import _deps  # noqa: F401

try:
    from .native import (
        align_uv_to_reference,
        air_warp_surface_guidance,
        calc_mid_surf,
        canonicalize_square_orientation,
        copy_attrib_colors,
        corr_topology_labels,
        coupled_square_map_from_guidance,
        ensure_target_metadata_copy,
        euler_number_mesh,
        icp_similarity,
        matlab_style_boundary_harmonic_map,
        map_curves_via_uv,
        map_thickness_to_atlas,
        parameterize_surface_uv,
        readdfc_sipi,
        readdfs,
        recolor_by_label,
        remove_curve_defects,
        save_surface,
        smooth_surf_hierarchy,
        transfer_surface_fields_by_uv,
        writedfc,
    )
    from .runtime import append_log, normalize_fileprefix, subject_log_path
except ImportError:
    from native import align_uv_to_reference, air_warp_surface_guidance, calc_mid_surf, canonicalize_square_orientation, copy_attrib_colors, corr_topology_labels, coupled_square_map_from_guidance, ensure_target_metadata_copy, euler_number_mesh, icp_similarity, matlab_style_boundary_harmonic_map, map_curves_via_uv, map_thickness_to_atlas, parameterize_surface_uv, readdfc_sipi, readdfs, recolor_by_label, remove_curve_defects, save_surface, smooth_surf_hierarchy, transfer_surface_fields_by_uv, writedfc
    from runtime import append_log, normalize_fileprefix, subject_log_path


def svreg_label_surf_hemi(subbasename: str, atlasbasename: str, hemi: str, *varargin: str) -> None:
    subbasename = normalize_fileprefix(subbasename)
    atlasbasename = normalize_fileprefix(atlasbasename)
    flags = "".join(varargin)
    sub_name = Path(subbasename).name
    tmp_dir = Path(subbasename).parent / f"{sub_name}.svreg.tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_base = str(tmp_dir / sub_name)

    append_log(
        subject_log_path(subbasename),
        "svreg_label_surf_hemi",
        [subbasename, atlasbasename, hemi, *varargin],
        subbasename=subbasename,
        atlasbasename=atlasbasename,
    )

    if hemi == "right":
        ensure_target_metadata_copy(tmp_base, atlasbasename)
        for suffix in (".cortex.dewisp.mask.nii.gz", ".bfc.nii.gz", ".pvc.frac.nii.gz"):
            src = Path(f"{atlasbasename}{suffix}")
            if src.is_file():
                shutil.copyfile(src, Path(f"{tmp_base}.target{suffix}"))
        label_src = Path(f"{atlasbasename}.label.nii.gz")
        if label_src.is_file():
            shutil.copyfile(label_src, Path(f"{tmp_base}.target.label.nii.gz"))
        hippo_src = Path(f"{atlasbasename}.hippo_carved.nii.gz")
        if hippo_src.is_file():
            shutil.copyfile(hippo_src, Path(f"{tmp_base}.target.hippo_carved.nii.gz"))

    for surf_type in ("mid", "inner", "pial"):
        atlas_src = f"{atlasbasename}.{hemi}.{surf_type}.cortex.dfs"
        sub_src = f"{subbasename}.{hemi}.{surf_type}.cortex.dfs"
        if surf_type == "mid":
            if not Path(sub_src).is_file():
                calc_mid_surf(
                    f"{subbasename}.{hemi}.inner.cortex.dfs",
                    f"{subbasename}.{hemi}.pial.cortex.dfs",
                    sub_src,
                )
        shutil.copyfile(atlas_src, f"{tmp_base}.target.{hemi}.{surf_type}.cortex.dfs")
        shutil.copyfile(sub_src, f"{tmp_base}.{hemi}.{surf_type}.cortex.dfs")

    subject_mid = readdfs(f"{tmp_base}.{hemi}.mid.cortex.dfs")
    atlas_mid = readdfs(f"{tmp_base}.target.{hemi}.mid.cortex.dfs")
    _ = euler_number_mesh(subject_mid)
    smooth_surf_hierarchy(f"{tmp_base}.{hemi}.mid.cortex.dfs", 10, flags)
    for smth in range(1, 11):
        atlas_smooth = Path(f"{atlasbasename}.{hemi}.mid.cortex_smooth{smth}.dfs")
        if atlas_smooth.is_file():
            shutil.copyfile(atlas_smooth, f"{tmp_base}.target.{hemi}.mid.cortex_smooth{smth}.dfs")

    atlas_guidance = None
    subject_guidance = None
    atlas_guidance_path = Path(f"{atlasbasename}.{hemi}.mid.cortex_smooth10.dfs")
    if atlas_guidance_path.is_file():
        atlas_guidance = readdfs(str(atlas_guidance_path))
    subject_guidance_path = Path(f"{tmp_base}.{hemi}.mid.cortex_smooth10.dfs")
    if subject_guidance_path.is_file():
        subject_guidance = readdfs(str(subject_guidance_path))

    if atlas_guidance is not None and not hasattr(atlas_guidance, "faces"):
        atlas_guidance.faces = atlas_mid.faces.copy()
    if subject_guidance is not None and not hasattr(subject_guidance, "faces"):
        subject_guidance.faces = subject_mid.faces.copy()

    raw_guidance_subject = subject_guidance if subject_guidance is not None else subject_mid
    raw_guidance_atlas = atlas_guidance if atlas_guidance is not None else atlas_mid
    field_transform = icp_similarity(raw_guidance_subject.vertices, raw_guidance_atlas.vertices)

    atlas_warp = Path(f"{atlasbasename}.warp")
    subject_warp = Path(f"{subbasename}.warp")
    air_env = os.environ.get("PYSVREG_USE_AIR_GUIDANCE")
    if air_env is None:
        use_air_guidance = (
            atlas_guidance is not None
            and subject_guidance is not None
            and atlas_warp.is_file()
            and subject_warp.is_file()
        )
    else:
        use_air_guidance = air_env == "1"
    guidance_subject = raw_guidance_subject
    guidance_atlas = raw_guidance_atlas
    if use_air_guidance:
        if atlas_guidance is not None and atlas_warp.is_file():
            guidance_atlas = air_warp_surface_guidance(
                atlas_guidance,
                atlas_warp,
                input_ucf=tmp_dir / f"{sub_name}.target.{hemi}.mid.cortex_smooth10.ucf",
                output_ucf=tmp_dir / f"{sub_name}.target.{hemi}.mid.cortex_smooth10_wrpd.ucf",
            )
            guidance_atlas.faces = atlas_mid.faces.copy()
        if subject_guidance is not None and subject_warp.is_file():
            guidance_subject = air_warp_surface_guidance(
                subject_guidance,
                subject_warp,
                input_ucf=tmp_dir / f"{sub_name}.{hemi}.mid.cortex_smooth10.ucf",
                output_ucf=tmp_dir / f"{sub_name}.{hemi}.mid.cortex_smooth10_wrpd.ucf",
            )
            guidance_subject.faces = subject_mid.faces.copy()

    # When AIR-warped guidance is available, the surfaces are already in the
    # shared frame MATLAB uses for flattening. Passing an explicit identity
    # avoids an extra ICP alignment inside coupled_square_map_from_guidance(),
    # which drifts the UV solve away from the MATLAB reference.
    transform = np.eye(4, dtype=np.float64) if use_air_guidance else field_transform
    map_mode = os.environ.get("PYSVREG_SURFACE_MAP_MODE", "coupled").lower()
    if map_mode == "coupled" or os.environ.get("PYSVREG_EXPERIMENTAL_COUPLED") == "1":
        try:
            atlas_mid, subject_mid = coupled_square_map_from_guidance(
                atlas_mid,
                subject_mid,
                atlas_guidance=guidance_atlas,
                subject_guidance=guidance_subject,
                transform_subject_to_atlas=transform,
                corr_step=4,
                sigma=10.0,
                maxiter=int(os.environ.get("PYSVREG_SURFACE_MAP_SOLVER_ITER", "2600")),
                solve_cycles=int(os.environ.get("PYSVREG_SURFACE_MAP_CYCLES", "4")),
                initial_map=os.environ.get("PYSVREG_SURFACE_MAP_INIT", "zero"),
                l2_iterations=int(os.environ.get("PYSVREG_SURFACE_MAP_L2_ITER", "63")),
            )
        except Exception as exc:
            if os.environ.get("PYSVREG_DEBUG_SURFACE_MAP") == "1":
                raise RuntimeError(
                    "coupled square-map path failed in svreg_label_surf_hemi"
                ) from exc
            if os.environ.get("PYSVREG_DEBUG_SURFACE_MAP") == "trace":
                traceback.print_exc()
            atlas_mid = parameterize_surface_uv(atlas_mid)
            subject_mid = parameterize_surface_uv(subject_mid, field_transform)
            subject_mid = align_uv_to_reference(subject_mid, atlas_mid, field_transform)
            atlas_mid = canonicalize_square_orientation(atlas_mid)
            subject_mid = canonicalize_square_orientation(subject_mid)
    elif map_mode == "boundary_harmonic" or os.environ.get("PYSVREG_EXPERIMENTAL_BOUNDARY_HARMONIC") == "1":
        atlas_mid, subject_mid = matlab_style_boundary_harmonic_map(
            atlas_mid,
            subject_mid,
            atlas_guidance=guidance_atlas,
            subject_guidance=guidance_subject,
        )
    else:
        atlas_mid = parameterize_surface_uv(atlas_mid)
        subject_mid = parameterize_surface_uv(subject_mid, field_transform)
        subject_mid = align_uv_to_reference(subject_mid, atlas_mid, field_transform)
        atlas_mid = canonicalize_square_orientation(atlas_mid)
        subject_mid = canonicalize_square_orientation(subject_mid)
    save_surface(f"{tmp_base}.target.{hemi}.mid.cortex.curvesmatch.dfs", atlas_mid)
    save_surface(f"{tmp_base}.{hemi}.mid.cortex.curvesmatch.dfs", subject_mid)

    subject_reg = transfer_surface_fields_by_uv(subject_mid, atlas_mid)
    save_surface(f"{tmp_base}.{hemi}.mid.cortex.reg.dfs", subject_reg)
    save_surface(f"{tmp_base}.target.{hemi}.mid.cortex.reg.dfs", atlas_mid)
    corr_topology_labels(f"{tmp_base}.{hemi}.mid.cortex.reg.dfs")
    recolor_by_label(f"{tmp_base}.{hemi}.mid.cortex.reg.dfs", atlasbasename)
    copy_attrib_colors(tmp_base, hemi, "reg.dfs")
    copy_attrib_colors(f"{tmp_base}.target", hemi, "reg.dfs")

    sub_inner = readdfs(f"{tmp_base}.{hemi}.inner.cortex.reg.dfs")
    sub_pial = readdfs(f"{tmp_base}.{hemi}.pial.cortex.reg.dfs")
    # Reload after topology cleanup and recoloring so we do not overwrite the
    # cleaned labels with the stale pre-clean transfer held in memory.
    subject_reg = readdfs(f"{tmp_base}.{hemi}.mid.cortex.reg.dfs")
    atlas_reg_base = readdfs(f"{tmp_base}.target.{hemi}.mid.cortex.reg.dfs")
    subject_reg, atlas_reg = map_thickness_to_atlas(subject_reg, sub_inner, sub_pial, atlas_reg_base, field_transform)
    save_surface(f"{tmp_base}.{hemi}.mid.cortex.reg.dfs", subject_reg)
    save_surface(f"{tmp_base}.target.{hemi}.mid.cortex.reg.dfs", atlas_reg)
    shutil.copyfile(f"{tmp_base}.{hemi}.mid.cortex.reg.dfs", f"{subbasename}.{hemi}.mid.cortex.svreg.init.dfs")

    if "r" not in flags:
        for surf_type in ("mid", "inner", "pial"):
            shutil.copyfile(
                f"{tmp_base}.{hemi}.{surf_type}.cortex.reg.dfs",
                f"{tmp_base}.{hemi}.{surf_type}.cortex.svreg.dfs",
            )

    protocol = tmp_dir / f"{sub_name}.sulcal_protocol_HD.xml"
    if protocol.is_file():
        curve_sets = [
            (Path(f"{atlasbasename}.{hemi}.dfc"), f"{tmp_base}.{hemi}.mapped.dfc", f"{subbasename}.{hemi}.mapped.dfc"),
            (Path(f"{atlasbasename}.{hemi}.all.dfc"), f"{tmp_base}.{hemi}.mapped.all.dfc", f"{subbasename}.{hemi}.mapped.all.dfc"),
        ]
        for atlas_curve, tmp_curve, out_curve in curve_sets:
            if not atlas_curve.is_file():
                continue
            curves, _ = readdfc_sipi(str(atlas_curve))
            mapped = map_curves_via_uv(curves, atlas_mid, subject_reg)
            writedfc(tmp_curve, mapped, str(protocol))
            shutil.copyfile(tmp_curve, out_curve)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 3:
        print("usage: svreg_label_surf_hemi.py SUBJECT_PREFIX ATLAS_PREFIX HEMI [flags ...]")
        return 1
    svreg_label_surf_hemi(argv[0], argv[1], argv[2], *argv[3:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
