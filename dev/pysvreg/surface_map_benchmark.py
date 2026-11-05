from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .dfsio import readdfs, writedfs
    from .native import (
        _solve_coupled_square_system,
        _surface_copy,
        air_warp_surface_guidance,
        boundary_loop,
        corr_topology_labels,
        harmonic_scalar_from_boundary,
        l2_surface_matching,
        matlab_boundary_square_values,
        orientation_tri,
        register_cc_boundary,
        transfer_surface_fields_by_uv,
    )
except ImportError:
    from dfsio import readdfs, writedfs
    from native import (
        _solve_coupled_square_system,
        _surface_copy,
        air_warp_surface_guidance,
        boundary_loop,
        corr_topology_labels,
        harmonic_scalar_from_boundary,
        l2_surface_matching,
        matlab_boundary_square_values,
        orientation_tri,
        register_cc_boundary,
        transfer_surface_fields_by_uv,
    )


def _tmp_base(prefix: str) -> str:
    path = Path(prefix)
    return str(path.parent / f"{path.name}.svreg.tmp" / path.name)


def _guidance_surfaces(subject_prefix: str, atlas_prefix: str, hemi: str):
    subject_base = _tmp_base(subject_prefix)
    atlas_mid = readdfs(f"{subject_base}.target.{hemi}.mid.cortex.dfs")
    subject_mid = readdfs(f"{subject_base}.{hemi}.mid.cortex.dfs")

    atlas_guidance_path = Path(f"{atlas_prefix}.{hemi}.mid.cortex_smooth10.dfs")
    subject_guidance_path = Path(f"{subject_base}.{hemi}.mid.cortex_smooth10.dfs")
    atlas_guidance = readdfs(str(atlas_guidance_path)) if atlas_guidance_path.is_file() else _surface_copy(atlas_mid)
    subject_guidance = readdfs(str(subject_guidance_path)) if subject_guidance_path.is_file() else _surface_copy(subject_mid)
    atlas_guidance.faces = np.asarray(atlas_mid.faces, dtype=np.int32).copy()
    subject_guidance.faces = np.asarray(subject_mid.faces, dtype=np.int32).copy()

    atlas_warp = Path(f"{atlas_prefix}.warp")
    subject_warp = Path(f"{subject_prefix}.warp")
    if atlas_warp.is_file():
        atlas_guidance = air_warp_surface_guidance(atlas_guidance, atlas_warp)
        atlas_guidance.faces = np.asarray(atlas_mid.faces, dtype=np.int32).copy()
    if subject_warp.is_file():
        subject_guidance = air_warp_surface_guidance(subject_guidance, subject_warp)
        subject_guidance.faces = np.asarray(subject_mid.faces, dtype=np.int32).copy()

    return atlas_mid, subject_mid, atlas_guidance, subject_guidance


def benchmark_surface_map(
    subject_prefix: str,
    atlas_prefix: str,
    reference_prefix: str,
    hemi: str,
    *,
    corr_step: int = 4,
    sigma: float = 10.0,
    maxiter: int = 1800,
    fraction: float = 0.05,
    niter: int = 80,
    step_size: float = 0.6,
    lap_reg: float = 25.0,
    loops: int = 1,
    init_mode: str = "harmonic",
) -> dict[str, float | int | str]:
    atlas_mid, subject_mid, atlas_guidance, subject_guidance = _guidance_surfaces(subject_prefix, atlas_prefix, hemi)
    reference_base = _tmp_base(reference_prefix)

    atlas_boundary, atlas_xb, atlas_yb = matlab_boundary_square_values(atlas_guidance)
    subject_boundary = boundary_loop(subject_guidance)
    subject_xb, subject_yb = register_cc_boundary(
        atlas_guidance,
        subject_guidance,
        atlas_boundary,
        subject_boundary,
        atlas_xb,
        atlas_yb,
        transform_subject_to_atlas=None,
    )

    if init_mode == "zero":
        x1 = np.zeros(atlas_mid.vertices.shape[0], dtype=np.float64)
        y1 = np.zeros(atlas_mid.vertices.shape[0], dtype=np.float64)
        x2 = np.zeros(subject_mid.vertices.shape[0], dtype=np.float64)
        y2 = np.zeros(subject_mid.vertices.shape[0], dtype=np.float64)
    elif init_mode == "harmonic":
        x1 = harmonic_scalar_from_boundary(atlas_guidance, atlas_boundary, atlas_xb)
        y1 = harmonic_scalar_from_boundary(atlas_guidance, atlas_boundary, atlas_yb)
        x2 = harmonic_scalar_from_boundary(subject_guidance, subject_boundary, subject_xb)
        y2 = harmonic_scalar_from_boundary(subject_guidance, subject_boundary, subject_yb)
    else:
        raise ValueError(f"unsupported init_mode: {init_mode}")

    corr_atlas, corr_subject = l2_surface_matching(
        atlas_guidance,
        subject_guidance,
        fraction=fraction,
        niter=niter,
        step_size=step_size,
        lap_reg=lap_reg,
        corr_step=max(1, int(corr_step)),
    )
    sigma_vec = np.full(corr_atlas.shape[0], float(sigma), dtype=np.float64)

    for _ in range(int(loops)):
        x1, y1, x2, y2 = _solve_coupled_square_system(
            atlas_guidance,
            subject_guidance,
            atlas_boundary,
            subject_boundary,
            atlas_xb,
            atlas_yb,
            subject_xb,
            subject_yb,
            corr_atlas,
            corr_subject,
            sigma_vec,
            x1,
            y1,
            x2,
            y2,
            maxiter=maxiter,
        )
        if corr_atlas.size == 0:
            continue
        ori1 = orientation_tri(np.asarray(atlas_mid.faces, dtype=np.int64), x1, y1)
        wrong1 = 1 if np.count_nonzero(ori1 == 1) < np.count_nonzero(ori1 == -1) else -1
        bad_vertices1 = np.unique(np.asarray(atlas_mid.faces, dtype=np.int64)[ori1 == wrong1].reshape(-1))
        ori2 = orientation_tri(np.asarray(subject_mid.faces, dtype=np.int64), x2, y2)
        wrong2 = 1 if np.count_nonzero(ori2 == 1) < np.count_nonzero(ori2 == -1) else -1
        bad_vertices2 = np.unique(np.asarray(subject_mid.faces, dtype=np.int64)[ori2 == wrong2].reshape(-1))
        if bad_vertices1.size == 0 and bad_vertices2.size == 0:
            continue
        bad_corr = np.zeros(corr_atlas.shape[0], dtype=bool)
        if bad_vertices1.size:
            bad_corr |= np.isin(corr_atlas, bad_vertices1)
        if bad_vertices2.size:
            bad_corr |= np.isin(corr_subject, bad_vertices2)
        sigma_vec[bad_corr] *= 0.5

    atlas_out = _surface_copy(atlas_mid)
    subject_out = _surface_copy(subject_mid)
    atlas_out.u = (0.5 * (1.0 + x1)).astype(np.float32)
    atlas_out.v = (0.5 * (1.0 + y1)).astype(np.float32)
    subject_out.u = (0.5 * (1.0 + x2)).astype(np.float32)
    subject_out.v = (0.5 * (1.0 + y2)).astype(np.float32)

    ref_subject_curves_path = Path(f"{reference_base}.{hemi}.mid.cortex.curvesmatch.dfs")
    ref_atlas_curves_path = Path(f"{reference_base}.target.{hemi}.mid.cortex.curvesmatch.dfs")
    reference_stage = "curvesmatch"
    if ref_subject_curves_path.is_file() and ref_atlas_curves_path.is_file():
        ref_subject_curves = readdfs(str(ref_subject_curves_path))
        ref_atlas_curves = readdfs(str(ref_atlas_curves_path))
    else:
        # The supplied MATLAB test archive contains final mapped surfaces but
        # not the temporary curvesmatch/reg intermediates.
        ref_subject_curves = readdfs(
            f"{reference_prefix}.{hemi}.mid.cortex.svreg.init.dfs"
        )
        ref_atlas_curves = readdfs(
            str(Path(reference_prefix).parent / f"atlas.{hemi}.mid.cortex.svreg.dfs")
        )
        reference_stage = "final"
    subject_uv = np.linalg.norm(
        np.column_stack([subject_out.u, subject_out.v]) - np.column_stack([ref_subject_curves.u, ref_subject_curves.v]),
        axis=1,
    )
    atlas_uv = np.linalg.norm(
        np.column_stack([atlas_out.u, atlas_out.v]) - np.column_stack([ref_atlas_curves.u, ref_atlas_curves.v]),
        axis=1,
    )

    mapped = transfer_surface_fields_by_uv(subject_out, atlas_out)
    tmp_out = Path("/tmp") / f"pysvreg_bench_{hemi}.dfs"
    writedfs(str(tmp_out), mapped)
    corr_topology_labels(str(tmp_out))
    mapped_clean = readdfs(str(tmp_out))
    ref_reg_path = Path(f"{reference_base}.{hemi}.mid.cortex.reg.dfs")
    ref_reg = (
        readdfs(str(ref_reg_path))
        if ref_reg_path.is_file()
        else readdfs(f"{reference_prefix}.{hemi}.mid.cortex.svreg.init.dfs")
    )

    return {
        "reference_stage": reference_stage,
        "init_mode": init_mode,
        "loops": int(loops),
        "corr_step": int(corr_step),
        "fraction": float(fraction),
        "niter": int(niter),
        "step_size": float(step_size),
        "lap_reg": float(lap_reg),
        "sigma": float(sigma),
        "maxiter": int(maxiter),
        "subject_uv_mean": float(np.mean(subject_uv)),
        "subject_uv_max": float(np.max(subject_uv)),
        "atlas_uv_mean": float(np.mean(atlas_uv)),
        "atlas_uv_max": float(np.max(atlas_uv)),
        "label_diff": int(np.count_nonzero(np.asarray(mapped_clean.labels) != np.asarray(ref_reg.labels))),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark alternative surface-map settings against MATLAB reference outputs.")
    parser.add_argument("subject_prefix")
    parser.add_argument("atlas_prefix")
    parser.add_argument("reference_prefix")
    parser.add_argument("hemi", choices=["left", "right"])
    parser.add_argument("--corr-step", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--maxiter", type=int, default=1800)
    parser.add_argument("--fraction", type=float, default=0.05)
    parser.add_argument("--niter", type=int, default=80)
    parser.add_argument("--step-size", type=float, default=0.6)
    parser.add_argument("--lap-reg", type=float, default=25.0)
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--init-mode", choices=["harmonic", "zero"], default="harmonic")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            benchmark_surface_map(
                args.subject_prefix,
                args.atlas_prefix,
                args.reference_prefix,
                args.hemi,
                corr_step=args.corr_step,
                sigma=args.sigma,
                maxiter=args.maxiter,
                fraction=args.fraction,
                niter=args.niter,
                step_size=args.step_size,
                lap_reg=args.lap_reg,
                loops=args.loops,
                init_mode=args.init_mode,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
