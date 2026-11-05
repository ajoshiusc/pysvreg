# SVReg replacement compatibility map

This file maps the launcher files in `scripts/` and the bundled AIR helpers to
the Python dispatch functions. The 51 Linux/macOS shell launchers and 28 Windows
command launchers resolve to the 28 commands in `compat_cli.COMMANDS`. This is
a dispatch map, not a claim that all commands have passed MATLAB output parity
tests; known fidelity limits are in
[`README.md`](README.md).

## SVReg launchers

Linux and macOS shell launchers pass their own name to `compat_cli.py`; the
dispatcher removes `_linux` or `_mac` before looking up the command. Windows
command wrappers pass the normalized command name directly. Every command has
a same-named wrapper in `scripts/`; the table lists its Python entry point and
platform-specific launchers. Set `PYTHON` to choose the interpreter.
Running `python -m pip install .` from the repository root installs all 28
commands as Python console scripts. Windows installs create `.exe` console
launchers in the environment's `Scripts` directory; each starts its Python
handler without MATLAB or MCR.

For an existing POSIX BrainSuite installation, `dev/deploy_python_svreg.py`
provides a dry-run-first in-place deployment. It installs the Python modules
under the BrainSuite SVReg directory and replaces its extensionless commands,
MCR `.sh` launchers, and Linux AIR helpers. The selected Python interpreter
must already include NumPy, SciPy, and NiBabel. Original files are preserved
with a `.pysvreg-original` suffix; see the package README for invocation.

| Command | Python implementation | Source launchers |
| --- | --- | --- |
| `clean_intermediate_files` | `clean_intermediate_files.main` | `clean_intermediate_files_linux.sh`, `clean_intermediate_files_mac.sh`, `clean_intermediate_files.cmd` |
| `generate_stats_xls` | `statistics.generate_stats_xls` | `generate_stats_xls_linux.sh`, `generate_stats_xls_mac.sh`, `generate_stats_xls.cmd` |
| `generate_vol_param_stats_xls` | `statistics.generate_vol_param_stats_xls` | `generate_vol_param_stats_xls_linux.sh`, `generate_vol_param_stats_xls_mac.sh`, `generate_vol_param_stats_xls.cmd` |
| `gui_bias_correct` | `gui_bias_correct.main` | `gui_bias_correct_linux.sh`, `gui_bias_correct_mac.sh`, `gui_bias_correct.cmd` |
| `refine_ROIs2` | `refine_ROIs2.main` | `refine_ROIs2_linux.sh`, `refine_ROIs2_mac.sh`, `refine_ROIs2.cmd` |
| `refine_sulci_hemi` | `refine_sulci_hemi.main` | `refine_sulci_hemi_linux.sh`, `refine_sulci_hemi_mac.sh`, `refine_sulci_hemi.cmd` |
| `register_cc_curve` | `standalone.main` | `register_cc_curve_linux.sh`, `register_cc_curve_mac.sh`, `register_cc_curve.cmd` |
| `svreg` | `svreg.main` | `svreg.cmd`, `svreg_linux.sh`, `svreg_mac.sh` |
| `svreg_apply_map` | `standalone.main` | `svreg_apply_map_linux.sh`, `svreg_apply_map_mac.sh`, `svreg_apply_map.cmd` |
| `svreg_get_mni_tal` | `standalone.main` | `svreg_get_mni_tal_linux.sh`, `svreg_get_mni_tal_mac.sh`, `svreg_get_mni_tal.cmd` |
| `svreg_label_surf_hemi` | `svreg_label_surf_hemi.main` | `svreg_label_surf_hemi_linux.sh`, `svreg_label_surf_hemi_mac.sh`, `svreg_label_surf_hemi.cmd` |
| `svreg_labelwith_atlas` | `standalone.main` | `svreg_labelwith_atlas_linux.sh`, `svreg_labelwith_atlas_mac.sh`, `svreg_labelwith_atlas.cmd` |
| `svreg_make_atlas` | `make_atlas.main` | `svreg_make_atlas_linux.sh`, `svreg_make_atlas_mac.sh`, `svreg_make_atlas.cmd` |
| `svreg_mni2sub` | `standalone.main` | `svreg_mni2sub_linux.sh`, `svreg_mni2sub_mac.sh`, `svreg_mni2sub.cmd` |
| `svreg_multiparc` | `multiparc.main` | `svreg_multiparc_linux.sh`, `svreg_multiparc_mac.sh`, `svreg_multiparc.cmd` |
| `svreg_prepare_files` | `prepare_files.prepare_files` | `svreg_prepare_files_linux.sh`, `svreg_prepare_files_mac.sh`, `svreg_prepare_files.cmd` |
| `svreg_refinements` | `svreg_refinements.main` | `svreg_refinements_linux.sh`, `svreg_refinements_mac.sh`, `svreg_refinements.cmd` |
| `svreg_resample` | `standalone.main` | `svreg_resample_linux.sh`, `svreg_resample_mac.sh`, `svreg_resample.cmd` |
| `svreg_runall` | `runall.main` | `svreg_runall.cmd`, `svreg_runall.sh` |
| `svreg_smooth_surf_function` | `standalone.main` | `svreg_smooth_surf_function_linux.sh`, `svreg_smooth_surf_function_mac.sh`, `svreg_smooth_surf_function.cmd` |
| `svreg_smooth_vol_function` | `standalone.main` | `svreg_smooth_vol_function_linux.sh`, `svreg_smooth_vol_function_mac.sh`, `svreg_smooth_vol_function.cmd` |
| `svreg_sulcal_map` | `sulcal_map.main` | `svreg_sulcal_map_linux.sh`, `svreg_sulcal_map_mac.sh`, `svreg_sulcal_map.cmd` |
| `svreg_thickness2atlas` | `standalone.main` | `svreg_thickness2atlas_linux.sh`, `svreg_thickness2atlas_mac.sh`, `svreg_thickness2atlas.cmd` |
| `svreg_volreg` | `svreg_volreg.main` | `svreg_volreg_linux.sh`, `svreg_volreg_mac.sh`, `svreg_volreg.cmd` |
| `thicknessPVC` | `thickness_pvc.main` | `thicknessPVC_linux.sh`, `thicknessPVC_mac.sh`, `thicknessPVC.cmd` |
| `volmap_ball` | `volmap_ball.main` | `volmap_ball_linux.sh`, `volmap_ball_mac.sh`, `volmap_ball.cmd` |

The `standalone.main` and `air_cli.main` functions branch on the normalized
command name supplied by `compat_cli.py`.

## Legacy Windows executables

`compile_scripts/compile_all.m` copies 26 `.exe` basenames into its Windows
package: 24 MATLAB Compiler commands and the two AIR helpers. Every copied
basename exactly matches a `[project.scripts]` console entry point in
`pyproject.toml`, so a Python installation generates a same-name launcher for
each. `svreg_prepare_files` and `svreg_runall` are additional Python commands
that were not in that older executable list. Put the Python environment's
`Scripts` directory before the legacy SVReg `bin` directory on `PATH` to
select the Python launchers by command name.

The `refine_ROIs2` path also promotes the registered target mid-surface to
`atlas.<hemi>.mid.cortex.svreg.dfs` and writes
`atlas.<hemi>.mid.cortex.jacobian.dfs`, matching side-effects in the MATLAB
implementation. The Jacobian mesh geometry matches the a USCBrain validation subject reference; its
vertex attributes still show numerical differences and are not yet parity
certified.

`refine_ROIs2` and `refine_ROIs_atlas2` now include the source-derived
curvature-weighted geodesic level-set update, including MATLAB's ROI-pair
filters, contour-based iteration count, warm-started diagonally preconditioned
CG with its 300-iteration cap, and zero-iteration skip. The USCBrain
`svreg_multiparc` path completed for a USCBrain validation subject in 2 min 28 s on the validation
host. Its surface labels differed from archived generic MATLAB
registered-subject surfaces (used only as a proxy) at 746/108,858 left and
1,359/113,265 right vertices; topology-only baseline counts were 666 and 579.
The archive has no multiparc-specific references, so direct output parity is
unmeasured. The `-E` atlas path uses the same refinement with all label pairs;
subject-level output parity remains unmeasured.

`svreg_sulcal_map` uses a Python port of `3rdParty/patchcurvature.m`'s
corner-angle-weighted vertex normals and two-ring quadratic fit. It replaces
the MATLAB `patchnormals_double` MEX without calling SVReg C/MEX code. The
complete a USCBrain validation subject scratch run took 54.8 seconds; no MATLAB sulcal-map output is
archived for direct numerical comparison, and the curvature correction changed
labels substantially relative to the prior approximation.

## Bundled AIR helper executables

| Legacy helper | Python replacement | Status |
| --- | --- | --- |
| `3rdParty/AIR_bin/warp_points_linux` | `air_cli.main` via `warp_points` | Python launcher; reference comparisons on seven subjects are recorded in `README.md`. |
| `3rdParty/AIR_bin/warp_points_mac` | `air_cli.main` via `warp_points` | Python launcher; same implementation as Linux. |
| `3rdParty/AIR_bin/warp_coord_vol_linux` | `air_cli.main` via `warp_coord_vol` | Python launcher; reference comparisons are recorded in `README.md`. |
| `3rdParty/AIR_bin/warp_coord_vol_mac` | `air_cli.main` via `warp_coord_vol` | Python launcher; same implementation as Linux. |
| `3rdParty/AIR_bin/warp_points.exe` | Installed `warp_points.exe` console launcher or `.cmd` wrapper | The repository binary remains for legacy MATLAB workflows. Install the Python package and put its environment's `Scripts` directory first on `PATH`, or invoke `warp_points.cmd` explicitly. |
| `3rdParty/AIR_bin/warp_coord_vol.exe` | Installed `warp_coord_vol.exe` console launcher or `.cmd` wrapper | The repository binary remains for legacy MATLAB workflows. Install the Python package and put its environment's `Scripts` directory first on `PATH`, or invoke `warp_coord_vol.cmd` explicitly. |

There are also `scripts/warp_points.cmd` and `scripts/warp_coord_vol.cmd`
wrappers. A direct path to the repository's AIR `.exe` still selects the
legacy binary; installed console launchers replace bare command names when
their Python environment is first on `PATH`.

## MATLAB MEX routines

The Python code does not load any of these MEX binaries. Linux MEX files are
duplicated in `3rdParty/` and `MEX_Files/` where noted.

| MEX binary | Python-side replacement | Coverage |
| --- | --- | --- |
| `cpd_P.mexa64` (`3rdParty/`, `MEX_Files/`) | `native.cpd_P` | Dense CPD posterior calculation; the MATLAB MEX itself is not loaded. |
| `cpd_Pcorrespondence.mexa64` (`3rdParty/`, `MEX_Files/`) | `native.cpd_Pcorrespondence` | Nearest correspondence calculation; the MATLAB MEX itself is not loaded. |
| `cpd_Pappmex.mexa64` (`3rdParty/`, `MEX_Files/`) | `native.cpd_P` in Python registration paths | Algorithmic replacement uses the dense posterior path; the truncated approximation is not a one-to-one port. |
| `fgt_model.mexa64` (`3rdParty/`, `MEX_Files/`) | CPD Gaussian-kernel routines in `native.py` | Dense/vectorized Gaussian kernels are used instead of the Fast Gauss Transform; large registration cases may run slower. |
| `fgt_predict.mexa64` (`3rdParty/`, `MEX_Files/`) | CPD Gaussian-kernel routines in `native.py` | Algorithm-level substitution, not a separate accelerated FGT predictor. |
| `trilinear.mexa64` (`MEX_Files/`) | SciPy interpolation in `native.py` and `standalone.py` | Python workflow uses SciPy sampling; the MATLAB MEX is not loaded. |

The repository includes C source for `cpd_P`, `cpd_Pappmex`,
`cpd_Pcorrespondence`, `fgt_model`, and `fgt_predict` under `3rdParty/`. Those
sources and compiled MEX files remain in the checkout for legacy MATLAB use;
they are not build or runtime requirements for `dev/pysvreg`.

The 12 additional `*.mex*` files under `dev/matlab_gifti/` belong to the
vendored MATLAB GIFTI utility. SVReg's Python path does not import that utility;
the required DFS geometry I/O is implemented in `dfsio.py`, and NIfTI I/O uses
NiBabel. Those vendor MEX files are not SVReg command dependencies.


## Native-library boundary

SVReg-specific implementation code and dispatch are Python. Runtime numerical
dependencies include NumPy, SciPy, and NiBabel; NumPy and SciPy ship compiled
native kernels. The replacement avoids MATLAB, MATLAB Runtime, this repository's
C/MEX code, and the bundled AIR `.exe` files when the Python launchers are used.
