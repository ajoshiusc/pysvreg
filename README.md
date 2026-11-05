# pysvreg

A MATLAB and MCR independent Python implementation of BrainSuite SVReg commands. The implementation uses Python with NumPy, SciPy, and NiBabel; it does not load SVReg C/MEX binaries or the AIR executables. Output compatibility is still being validated, so see [the compatibility notes](dev/pysvreg/COMPATIBILITY.md) for coverage and known limitations.

## Install

Requires Python 3.10 or newer. From a checkout:

```bash
python3 -m pip install .
```

This installs the SVReg command entry points into the active Python environment. The package does not include BrainSuite atlases; use an atlas from a compatible BrainSuite distribution and pass its prefix where required, or set `PYSVREG_ROOT` to an SVReg installation containing the atlas.

To invoke the source tree without installing:

```bash
PYTHON=/path/to/python scripts/svreg_linux.sh SUBJECT_PREFIX ATLAS_PREFIX -k
```

## BrainSuite deployment

`dev/deploy_python_svreg.py` supports a dry run against a POSIX BrainSuite SVReg directory. Review the proposed replacements before applying them:

```bash
python3 dev/deploy_python_svreg.py /path/to/BrainSuite23a/svreg --python /path/to/python
python3 dev/deploy_python_svreg.py /path/to/BrainSuite23a/svreg --python /path/to/python --apply
```

The deployer preserves replaced launchers with a `.pysvreg-original` suffix. The selected Python interpreter must have NumPy, SciPy, and NiBabel installed.

## License and notices

SVReg is distributed under GPL-2.0; see [LICENSE](LICENSE). Third-party notices are preserved in [NOTICE_src.txt](NOTICE_src.txt).
