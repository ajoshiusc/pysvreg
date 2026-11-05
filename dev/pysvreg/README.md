# Python SVReg replacement

This package is a native Python implementation of SVReg functionality for
BrainSuite-produced files. The entry points below do not start MATLAB or the
MATLAB Runtime. Launchers for Linux and macOS use `python3` and this package.

See [`COMPATIBILITY.md`](COMPATIBILITY.md) for the per-launcher map, AIR helper
replacements, and MEX replacement coverage.

## Install dependencies

Use Python 3 and install the package from the repository root with
`python3 -m pip install .`. This installs NumPy, SciPy, NiBabel, and all 28
command entry points. On Windows, the installer creates command-name `.exe`
launchers in the environment's `Scripts` directory; they invoke the Python
entry points without MATLAB or MCR. Alternatively, install the runtime packages with
`python3 -m pip install -r dev/pysvreg/requirements.txt` and use the
source-tree launchers. SVReg's replacement code does not load C/MEX helpers;
NumPy and SciPy provide their own compiled numerical kernels.

The wheel does not duplicate atlas data. When using installed commands, set
`PYSVREG_ROOT` to the checkout or BrainSuite `svreg` directory to use its
bundled default atlas, or pass an atlas prefix explicitly to commands that
accept one.

To prepare an existing POSIX BrainSuite installation, run
`python3 dev/deploy_python_svreg.py /path/to/BrainSuite23a/svreg --python /path/to/python`
for a dry run, then add `--apply` after reviewing the paths. The selected
Python must already have NumPy, SciPy, and NiBabel installed. The deployer
copies the package into the BrainSuite tree, replaces its extensionless
commands, `.sh` launchers, and Linux AIR helpers, and sets `PYSVREG_ROOT` to
the BrainSuite SVReg directory at launch. Replaced files are kept beside the
new launchers with a `.pysvreg-original` suffix. On Windows, install the wheel
and put its environment's `Scripts` directory before the legacy `bin`
directory on `PATH`.

## Commands with Python entry points

| Command | Python implementation |
| --- | --- |
| `svreg` | `svreg.py` and the native surface/volume stages |
| `svreg_label_surf_hemi` | `svreg_label_surf_hemi.py` |
| `refine_ROIs2` | `refine_ROIs2.py` |
| `refine_sulci_hemi` | `refine_sulci_hemi.py` |
| `volmap_ball` | `volmap_ball.py` |
| `svreg_volreg` | `svreg_volreg.py` |
| `svreg_refinements` | `svreg_refinements.py` |
| `clean_intermediate_files` | `clean_intermediate_files.py` |
| `svreg_smooth_vol_function` | `standalone.py` |
| `svreg_resample` | `standalone.py` |
| `svreg_get_mni_tal` | `standalone.py` |
| `svreg_mni2sub` | `standalone.py` |
| `svreg_apply_map` (scalar 3D/4D and BrainSuite `.eig`) | `standalone.py` |
| `svreg_smooth_surf_function` | `standalone.py` |
| `svreg_thickness2atlas` | `standalone.py` |
| `svreg_labelwith_atlas` | `standalone.py` |
| `register_cc_curve` | `standalone.py` |
| `generate_stats_xls` | `statistics.py` |
| `generate_vol_param_stats_xls` | `statistics.py` |
| `svreg_prepare_files` | `prepare_files.py` |
| `svreg_make_atlas` | `make_atlas.py` |
| `svreg_multiparc` | `multiparc.py` |
| `svreg_sulcal_map` | `sulcal_map.py` |
| `thicknessPVC` | `thickness_pvc.py` |
| `svreg_runall` | `runall.py` |
| `gui_bias_correct` | `gui_bias_correct.py` |
| `warp_points` (AIR helper) | `air_cli.py` |
| `warp_coord_vol` (AIR helper) | `air_cli.py` |

The command-compatible launchers are in `scripts/`. Linux and macOS use the
existing `.sh` entry points; each command also has a same-name Windows `.cmd`
wrapper. Set `PYTHON` to select an interpreter, for example
`PYTHON=/path/to/python scripts/svreg_linux.sh SUBJECT_PREFIX ATLAS_PREFIX -k`.

## Coverage and remaining work

Every command name used by the `scripts/` launchers has a Python dispatch entry
point. This means the launchers run without MATLAB or MCR; it does not yet mean
that every command has been certified as a drop-in numerical replacement.
Scalar and BrainSuite `.eig` tensor map application are implemented. Numerical
parity still needs broader reference runs.

The Python `refine_ROIs2` command now ports MATLAB's curvature-weighted
geodesic level-set boundary update, including its ROI-pair exclusions,
contour-derived iteration count, warm-started diagonal-PCG solver with a
300-iteration cap, and zero-iteration skip. `svreg_make_atlas -E` uses the
same update across all label pairs. A complete a USCBrain validation subject USCBrain multiparc run
took 2 min 28 s on the validation host. Its labels differed from archived
generic MATLAB registered-subject surfaces (only a proxy for multiparc) at
746/108,858 left and 1,359/113,265 right vertices; the topology-only baseline
differed at 666 and 579. The archive contains no multiparc-specific reference
surfaces, so direct output parity for this update remains unverified.

Both AIR helpers have Python implementations in `native.py` and are available
through `air_cli.py`. Python registration implements the CPD routines formerly
supplied by the `cpd_*.mexa64` and `fgt_*.mexa64` files. It uses SciPy
interpolation in place of the legacy `trilinear.mexa64`; DFS I/O is Python and
NIfTI I/O uses NiBabel. None of those MATLAB MEX files is loaded by the Python
commands.

`warp_points` was compared with the installed AIR reference on all seven
subjects in `SVRegTestData_orig.zip`: 2,837,973 output coordinates matched
exactly at six-decimal text precision, covering order-2, order-3, and order-4
warps. For one 304,882-vertex order-4 surface, the Python routine ran in about
0.91 seconds and the AIR binary in 0.94–1.00 seconds on the validation host.
After matching AIR's space-separated columns and trailing blank line, a fresh
command-line comparison against the BrainSuite23a `warp_points_linux` binary
was byte-identical on 1,884,339 left/right inner-surface vertices from all
seven subjects. End-to-end Python command times were 1.15–4.14 seconds versus
0.48–2.07 seconds for AIR on this host; the Python times include interpreter
startup. `dev/pysvreg/test_air_cli.py` guards the AIR text format.
`warp_coord_vol` produced voxel-for-voxel identical images for order-2,
order-3, and order-4 examples, including full output grids of up to 21 million
voxels. On the validation host, Python command times were about 2.0, 2.5, and
4.3 seconds for those three cases, versus 0.5, 0.8, and 2.3 seconds for AIR.
An additional comparison against the installed AIR binary used a a USCBrain validation subject order-4
warp: a random signed-short volume and the X/Y/Z coordinate fields used by
`get_air_map` matched exactly across all 11.5 million output voxels per field.
That case took 3.8 seconds in Python and 0.86 seconds in AIR on this host. The
Python point helper is close to AIR's runtime; volume-helper time is still
several seconds and has more overhead on smaller grids.

## Fidelity and runtime

The target is MATLAB output parity, including NIfTI voxel order, coordinate
conventions, interpolation, header fields, filenames, and exit behavior. The
Python parity utilities in `parity.py` and `surface_stage_compare.py` compare
representative outputs. `svreg_apply_map` supports scalar 3D/4D images with
MATLAB-compatible nearest, linear, cubic-convolution, and spline interpolation.
On a seeded random-volume fixture, nearest and linear matched MATLAB exactly;
cubic-convolution differed by at most `2.4e-7`, and spline interpolation by at
most `3.4e-6`. The same command supports BrainSuite's custom 12-float
`.eig.nii.gz` files, including nearest-neighbor eigenvalue/vector warping and
PPD vector reorientation. Identity and rigid-rotation tensor fixtures pass; a
deployed MATLAB reference run could not resolve its `load_untouch_eig_gz`
dependency, so tensor numerical parity is not yet confirmed. Other resampling
commands still need wider parity checks. `svreg_resample` now
routes cubic-convolution and spline requests through the same MATLAB-matched
samplers and passes its corner-median or explicit extrapolation value through;
small 3D/4D dispatch and extrapolation fixtures pass. A direct MATLAB reference
run for that command is still pending. `svreg_smooth_vol_function` now follows
MATLAB's documented `imgaussfilt3` default filter size and replicate padding.
Its 3D boundary and 4D frame behavior have regression coverage, but direct
MATLAB numeric comparison is pending. The standalone
`svreg_smooth_surf_function` solve is checked against the MATLAB source's
weighted-stiffness normal equations on a synthetic mesh; direct MATLAB binary
comparison is unavailable on this host. Treat this package as an in-progress
replacement until the remaining commands are ported and those comparisons are
reviewed.

`svreg_get_mni_tal` and `svreg_mni2sub` use MATLAB's half-away-from-zero voxel
rounding. The MNI-to-Talairach rotation follows the bundled SPM matrix
convention, with regression coverage for points above and below the AC. A
a USCBrain validation subject coordinate lookup was also exercised against the supplied subject map
and USC atlas map.

The map regression fixtures can be rerun from the repository root with
`python3 dev/pysvreg/test_apply_map.py` and
`python3 dev/pysvreg/test_eig_map.py`. The resampler
dispatch and extrapolation checks are in
`python3 -m dev.pysvreg.test_svreg_resample`.

An earlier `svreg_label_surf_hemi` left-hemisphere run on a USCBrain validation subject took 13
minutes 16 seconds with the 1,200-iteration map setting. Its output mid-surface
vertex coordinates matched MATLAB exactly, while mean UV error was `0.00310`
(maximum `0.08808`) and 5,139 of 108,858 vertex labels differed. With the new
1,800-iteration default, the deployed Python command took 7 minutes 12 seconds
on a fresh copy of the same subject. Vertex coordinates again matched exactly;
mean UV error was `0.00204` (maximum `0.08928`), and 5,110 labels differed.
These runtimes come from different code states and are not a controlled speed
comparison. The MATLAB archive lacks intermediate curve-match and registration
surfaces, so output comparison uses the final registered surfaces.

The coupled surface-map solver exposes experimental schedule overrides through
`PYSVREG_SURFACE_MAP_INIT` (`harmonic` or `zero`),
`PYSVREG_SURFACE_MAP_CYCLES` (default `1`),
`PYSVREG_SURFACE_MAP_SOLVER_ITER` (default `1800`), and
`PYSVREG_SURFACE_MAP_L2_ITER` (default `80`). The MATLAB source uses zero maps,
four correction cycles, 2,600 PCG iterations, and 62 L2 iterations for this
path. A a USCBrain validation subject run with those source-derived settings had not finished after 20
minutes and was stopped before output comparison. The faster defaults remain
active; output parity for the source-derived schedule is unverified.

The left-hemisphere benchmark now compares against the final MATLAB subject and
atlas surfaces when the archive lacks temporary mapping files. On that a USCBrain validation subject
reference, the previous harmonic/1,200/80 schedule had mean subject UV error
`0.00310`, mean atlas UV error `0.00375`, and 5,139 differing subject labels.
Harmonic initialization with 1,800/80 reduced those to `0.00204`, `0.00239`,
and 5,108. A 2,600/80 cap changed the label count to 5,104 but had slightly
higher mean UV errors and took longer. Zero initialization with 1,200/80
produced 5,147 differing labels and higher UV error. The default solver cap is
now 1,800; these isolated schedule benchmarks are distinct from the deployed
single-hemisphere run above and do not constitute a new full-pipeline parity
run.

The ROI statistics commands are implemented in Python and their Linux/macOS
launchers no longer require MATLAB Runtime. `generate_stats_xls` calculates
surface area, mean PVC thickness, and tissue volumes. It was run on a scratch
copy of the MATLAB-processed a USCBrain validation subject subject; all 163 report rows matched, with
the largest numeric difference `0.000394` in area or volume fields, caused by
surface arithmetic precision. `generate_vol_param_stats_xls` calculates
tissue-weighted parameter means and standard deviations. A synthetic fixture
now checks all ROI rows and the special pure-CSF/GM/WM rows against the MATLAB
source formula; this exposed and fixed a Python column-placement error. The
installed MATLAB executable cannot provide a direct file comparison because it
fails before calculation with `vl` undefined in the MATLAB source.
`python3 dev/pysvreg/test_vol_param_stats.py` reruns the
formula-level fixture.

`svreg_prepare_files` stages atlas metadata, target volumes and surfaces in the
subject's `.svreg.tmp` directory. It creates a missing subject mid surface from
the inner and pial surfaces. Its Linux/macOS launchers use Python and report all
missing required inputs before returning an error.

`svreg_make_atlas` promotes registered surfaces and labels into atlas-ready
files, builds surface smoothing hierarchies, transfers curvature data when
available, and writes the ROI list. A synthetic surface fixture exercised the
Linux command path. Subject-level parity against the MATLAB command is still
unmeasured, including its manually edited `-E` labeling branch.

`svreg_multiparc` transfers labeled atlas surfaces through the SVReg flat map
and applies the subject coordinate map to atlas label volumes when available.
It now stages atlas-specific surface labels, applies the volumetric refinement
stage, and publishes the MATLAB-compatible mid-surface and label-volume names.
A a USCBrain validation subject run with `Yeo2011_7Networks` produced both hemisphere surfaces and a
256 x 176 x 256 label volume with all eight atlas labels. A topology-only
USCBrain baseline completed in 65 seconds; with MATLAB-style ROI refinement,
the full run completed in 2 min 28 s. Refined labels differed from the generic
archived MATLAB subject surfaces at 746/108,858 left and 1,359/113,265 right
vertices. These references are only a proxy, since the archive has no
multiparc-specific surfaces or label volume. A synthetic volume-transfer test
checks MATLAB `interp3` axis order, nearest sampling, zero extrapolation, and
output datatype.

`svreg_sulcal_map` creates per-hemisphere sulcal surface labels and a volume
label map from refined curve files. Its surface path now follows the MATLAB
face-mask, triangle-median, and largest connected face-component rules; its
volume path retains the input label datatype. Synthetic topology and volume
fixtures cover those rules. The source-style two-pass distance sweeps are
batched across curves without changing their ordered updates; on two real
a USCBrain validation subject curves, batched and independent distances were exactly equal. The local
quadratic curvature fit now ports `3rdParty/patchcurvature.m`, including its
corner-angle-weighted normals and two-ring least-squares patch; it does not
load the MATLAB `patchnormals_double` MEX. On a unit-sphere fixture its mean
curvature is positive and close to the analytic value, and it scales inversely
with radius. For a 108,858-vertex a USCBrain validation subject hemisphere the fit took 13.9 seconds.
The complete scratch `svreg_sulcal_map` run took 54.8 seconds and wrote both
surfaces and an int16 label volume. Replacing the previous fast curvature
estimate changed 51,002 left and 56,735 right surface labels and 386,474 volume
voxels. No MATLAB sulcal-map outputs are archived for direct comparison, so
these changes are source-aligned but output parity remains unverified.

`refine_sulci_hemi` now ports the surface-based geodesic curvature-flow path,
including curve defect removal, the local surface patch, signed geodesic field,
finite-element flow, zero-level contour extraction, and MATLAB-compatible
temporary-directory naming. A sparse LU factorization is reused for the ten
implicit flow solves on each curve patch; this reduced a representative
a USCBrain validation subject curve from about 22 seconds with iterative solves to about 1.4 seconds.
The attached `SVRegTestData_orig.zip` does not contain mapped/refined DFCs, so
this comparison uses the separate expanded a USCBrain validation subject MATLAB reference set at
`SVRegTestData_svreg26a_linux/a USCBrain validation subject` available on the validation host. Across
both hemispheres, the source-tree run took 68.7 seconds; the installed
BrainSuite launchers took 49.7 seconds left and 34.3 seconds right in a
separate run; a later 14.3-second left temp-directory smoke run also succeeded.
These timings were not controlled. Python produced 2,281 left and 2,297 right
points versus 2,330 and 2,363 in the archived MATLAB DFCs. All 36 curves per
hemisphere agreed on whether each result was empty. Across nonempty curves,
mean bidirectional nearest-point distance was 0.435 mm on the left and 0.454 mm
on the right; mean per-curve 95th-percentile distances were 2.24 mm and 2.16 mm.
Outputs are close, but
not identical; nearest-vertex ties, graph-path tie breaking, and numerical
differences in the flow can change points by several millimeters on some
curves. The numerical results and runtime are measured on this a USCBrain validation subject set and
are not a general certification for every subject.

`thicknessPVC` has a Python implementation of the tissue-fraction heat solver,
surface upsampling, output NIfTIs, thickness surfaces, and hemisphere maps.
Copied a USCBrain validation subject inputs from the supplied archive were compared with the installed
BrainSuite23a MATLAB executable at 4 mm and 2 mm computation resolutions. At
4 mm, Python ran in about 19 seconds and MATLAB in about 39 seconds; their heat
maps differed by at most `4.3e-8`, gray-matter fraction maps matched exactly,
and mid-surface thickness attributes differed by less than `4.8e-7 mm` at the
95th percentile. At 2 mm, Python ran in about 22 seconds and MATLAB in about
39 seconds; the heat-map maximum absolute difference was `1.1e-6`, gray-matter
fraction maps matched exactly, and mid-surface thickness attributes differed
by less than `7.2e-7 mm` at the 95th percentile. The isosurface vertex sets
matched within `1.6e-5 mm` at both resolutions. Triangle connectivity still
differs even though the vertex sets and triangle counts are close. MATLAB's
PCG stopped at 15 iterations at 4 mm and at its 100-iteration limit at 2 mm,
with a warning that its requested tolerance was not achievable. Both programs
produce extreme thickness outliers at these coarse resolutions (up to about
1.65 million mm at 2 mm), so these comparisons validate Python parity with the
reference behavior, not useful thickness accuracy at the reduced resolutions.
At the default 0.48 mm resolution, Python completed a USCBrain validation subject in 4 min 40 s on the
validation host (291 x 349 x 247 computation grid, 377 PCG iterations). The
input volumes and surfaces matched the MATLAB reference inputs byte-for-byte.
Python produced the same isosurface vertex counts, triangle counts, and face
connectivity. Its thickness attributes differed by less than 9.6e-7 mm at the
95th percentile. The resampled heat maps had a mean absolute difference of
2.4e-8; three voxels differed by more than 0.01, with a maximum difference of
0.0176. This full-resolution run also exposed and fixed a NumPy 2.x overflow
in the isosurface chunk offsets, now stored as 16-bit integers.

`svreg_runall` completed all four stages on the supplied a USCBrain validation subject subject using
the same `USCBrain/USCBrain` atlas recorded in the MATLAB reference log. The
Python run took 34 min 28 s from thickness through ROI statistics on the
validation host; the historical MATLAB logs span about 6 h 35 min from
thickness to statistics (the log times are not a controlled same-host runtime
benchmark). Final registered subject mid-surface coordinates matched the
reference exactly, vertex-for-vertex. Surface parcel labels differed at 10,671
of 222,123 vertices (4.8%). The final volume label maps differed at 122,550 of
11,534,336 voxels (1.06%), compared with 173,653 differing voxels in the
initial maps. The final SVReg coordinate map had mean absolute component
differences of 0.43, 0.65, and 0.68 voxels (x/y/z); mean vector difference was
1.16 voxels over the full volume and 1.99 voxels inside the cerebrum mask
(95th percentile 5.55). Surface thickness attributes matched within a mean
absolute difference of 6.2e-8 mm. ROI mean thickness differed by 0.022 mm on
average, while mean absolute GM, CSF, WM, total, and surface-area differences
were 134, 143, 93, 215 mm^3, and 23–37 mm^2 per ROI. The end-to-end run is
complete and substantially faster than the elapsed time recorded in the MATLAB
reference logs; volume map, parcel label, and ROI volume parity still need
improvement.

`volmap_ball` creates the surface-constrained unit-ball coordinates used by
MATLAB SVReg. It maps each hemisphere's square coordinates to the sphere,
interpolates those coordinates onto the cortex-mask boundary, and solves the
interior six-neighbor harmonic problem with a sparse preconditioned conjugate
gradient solver. The generated MAT file includes the `Brain` fields consumed
by `svreg_volreg`, including one-based MATLAB mask indices and separate subject
and target surface maps. On the supplied a USCBrain validation subject run, generation took 31 seconds
for the 401,413-voxel subject mask and 2 min 50 seconds for the 1,991,748-voxel
target mask; all coordinates were finite and had norms no greater than one.
There was no saved MATLAB unit-ball MAT file in the reference data, so direct
numeric parity for this intermediate remains unmeasured. The small fixture can
be rerun with
`python3 -m dev.pysvreg.test_volmap_ball`.

`svreg_volreg` now uses the subject and target unit-ball maps to initialize its
volumetric correspondences, maps pial/inner/mid surfaces through the same
spherical coordinates, and extends their displacement through the cerebrum
with the source Laplacian/data-fit system. The MATLAB source uses global 3D
scattered linear interpolation; for full-sized maps, Python uses a local
weighted affine interpolator to avoid materializing a multi-million-point
Delaunay mesh. It now replaces an affine estimate with barycentric
interpolation when it finds a query-containing local tetrahedron whose
circumsphere is empty across the nearest 32 samples; other queries retain the
affine estimate and its nearest-neighbor hull fallback. On a deterministic
5,000-source/10,000-query nonlinear fixture, local simplex corrections matched
global Delaunay values where accepted (46.4% of queries) and reduced mean
interpolation error from 0.00516 to 0.00279 output units in 2.76 seconds.
On large volume clouds, the fast hull check tests likely extrapolation points
against a Delaunay triangulation of the source cloud's convex-hull vertices
and uses nearest neighbors outside the hull. The large-cloud candidate
threshold is `-0.005` on the minimum local affine weight. In a seeded
30,000-query a USCBrain validation subject sample, it selected 12,867 candidates and included all
1,451 points that a full hull check classified as outside. For the 222,123
subject and 696,035 atlas surface samples, a surface-specific path uses nearest-
neighbor extrapolation: the points lie on the unit sphere, so non-vertex
queries fall outside the convex hull and `mygriddata3` uses its nearest-value
fallback. This implementation matched global SciPy Delaunay plus nearest
extrapolation exactly on a float32 unit-sphere fixture. The prior local-affine
surface estimate had a mean error of 0.058 output units. Direct nearest search
also avoids local simplex refinement for all six surface transfers. On the
a USCBrain validation subject volume map, global SciPy Delaunay took 100 s to triangulate the
1,991,748 target samples and another 83 s to locate only 9,959 queries;
querying the 401,413-voxel subject cortex mask at that rate would take about
56 min.

A separate seeded sample of 1,000 real a USCBrain validation subject volume queries was compared
against SciPy's full global Delaunay interpolation with nearest extrapolation.
The reference triangulation took 85 seconds and query evaluation 75 seconds;
the local path took 7.7 seconds. It classified 52 points outside the source
hull; after tightening the candidate threshold, all 52 received the exact
nearest value. The previous wider threshold missed one by 0.087 voxels. This
checks the hull fallback against real data while keeping full-subject
interpolation within practical runtime.

The local volume interpolator is exact on affine fields and approximates the
smooth harmonic map locally. The stage also runs four forward/inverse
consistency cycles, the single-pass negative-Jacobian correction, nonlinear
intensity refinement (100 iterations by default), and computes forward and
inverse Jacobians and the atlas-space inverse map. The tightened hull check
was rerun through the installed BrainSuite launcher on a USCBrain validation subject with the
matching USCBrain atlas. It completed volume registration in 15 min 32 s and
`svreg_refinements` in 25 s. Against MATLAB, forward-map error was 0.770
voxels mean/2.403 p95 in the 401,413-voxel unit-ball cortex mask and
1.398/4.639 in the 1,046,192-voxel cerebrum mask; inverse-map error was
0.822/2.657 in the 4,355,415-voxel atlas cerebrum mask. The prior wider hull
threshold measured 0.789/2.467 and 1.405/4.650 in the corresponding forward
masks, so the new output is closer to MATLAB. The initial label volume
differed from MATLAB at 139,155 of 11,534,336 voxels. After
`svreg_refinements`, the final label difference was 91,936 voxels (0.80%
overall), down from 92,280 with the wider hull threshold. This refinement
ports MATLAB's 6-connected
hole filling, 26-connected small-component removal, pial-mask correction,
deduplicated nearest samples, and non-integer nearest-fill offsets. The
refinement stage took 25 seconds on this scratch run. Volume interpolation
remains a local approximation to MATLAB's global Delaunay interpolation, so
other subjects still need parity checks. The solver fixtures can be rerun with
`python3 -m dev.pysvreg.test_svreg_volreg` and
`python3 -m dev.pysvreg.test_svreg_elastic`.
`gui_bias_correct` uses Tkinter to browse the image and tissue labels, adjust
target tissue medians, preview a smooth multiplicative correction, and save a
corrected NIfTI. The source's `view_nii_bias_corr` viewer is not present in this
repository, so this is a Python workflow with different controls and still
needs review against the intended manual correction process.

The Linux and macOS AIR helper paths in `3rdParty/AIR_bin/` are executable
Python launchers. Windows Python command wrappers are available both beside
the bundled helpers in `3rdParty/AIR_bin/` and under `scripts/`; set `PYTHON`
to choose the interpreter. The original AIR `.exe` files and MATLAB MEX
binaries remain for legacy workflows. The Python SVReg commands do not load or
invoke those files; their AIR warp operations use the Python implementations.

## Deploying into BrainSuite23a

Use `dev/deploy_python_svreg.py` to review and apply replacements in an existing POSIX BrainSuite SVReg directory. It performs a dry run unless `--apply` is supplied and preserves replaced launchers with a `.pysvreg-original` suffix. See the install instructions above.

## Direct invocation

```bash
python3 -m dev.pysvreg SUBJECT_PREFIX [ATLAS_PREFIX] [flags ...]
python3 dev/pysvreg/compat_cli.py svreg_resample IN.nii.gz OUT.nii.gz -res 1 1 1
```
