from __future__ import annotations

from pathlib import Path
import re
import unittest

from .compat_cli import COMMANDS


REPO_ROOT = Path(__file__).resolve().parents[2]


class TestCompatibilityInventory(unittest.TestCase):
    def test_every_posix_and_windows_command_wrapper_has_a_handler(self) -> None:
        scripts = REPO_ROOT / "scripts"
        if not scripts.is_dir():
            bin_dir = REPO_ROOT / "bin"
            if not bin_dir.is_dir():
                self.skipTest("launcher inventory is not included in this Python-only snapshot")
            missing = sorted(command for command in COMMANDS if not (bin_dir / command).is_file())
            self.assertFalse(missing, f"missing deployed command launchers: {missing}")
            for command in COMMANDS:
                launcher = bin_dir / command
                self.assertTrue(launcher.stat().st_mode & 0o111, f"{launcher} is not executable")
                self.assertIn("dev.pysvreg.compat_cli", launcher.read_text(encoding="utf-8"))
                shell_alias = bin_dir / f"{command}.sh"
                self.assertTrue(shell_alias.is_file(), f"missing deployed shell alias: {shell_alias}")
            return

        shell_names = set()
        for path in scripts.glob("*.sh"):
            name = path.stem
            if name.endswith("_linux"):
                name = name[:-6]
            elif name.endswith("_mac"):
                name = name[:-4]
            shell_names.add(name)

        windows_names = {path.stem for path in scripts.glob("*.cmd")}
        self.assertEqual(len(shell_names), 26)
        self.assertEqual(len(windows_names), 28)
        self.assertFalse(shell_names - COMMANDS.keys())
        self.assertFalse(windows_names - COMMANDS.keys())

    def test_matlab_compiled_command_names_have_python_console_entries(self) -> None:
        compile_script = REPO_ROOT / "compile_scripts" / "compile_all.m"
        air_bin = REPO_ROOT / "3rdParty" / "AIR_bin"
        if not compile_script.is_file() or not air_bin.is_dir():
            self.skipTest("legacy source inventory is not included in this Python-only snapshot")
        source = compile_script.read_text(encoding="utf-8", errors="replace")
        compiled = set(re.findall(r"mcc\s+-m(?:\s+-v)?\s+(\w+)\.m", source))
        packaged = set(re.findall(r"copyfile\('(?:[^']*/)?([^/']+)\.exe'", source))
        air = {path.stem for path in air_bin.glob("*.exe")}

        self.assertEqual(len(compiled), 24)
        self.assertEqual(len(packaged), 26)
        self.assertEqual(packaged - compiled, air)
        self.assertFalse((compiled | air) - COMMANDS.keys())

    def test_matlab_mex_files_have_python_replacements_or_are_unneeded(self) -> None:
        required_inventory = (
            REPO_ROOT / "3rdParty",
            REPO_ROOT / "MEX_Files",
            REPO_ROOT / "dev" / "matlab_gifti",
        )
        if not all(path.is_dir() for path in required_inventory):
            self.skipTest("legacy MEX inventory is not included in this Python-only snapshot")
        expected = {
            "cpd_P.mexa64",
            "cpd_Pappmex.mexa64",
            "cpd_Pcorrespondence.mexa64",
            "fgt_model.mexa64",
            "fgt_predict.mexa64",
            "trilinear.mexa64",
        }
        mex_dirs = (REPO_ROOT / "3rdParty", REPO_ROOT / "MEX_Files")
        actual = {p.name for directory in mex_dirs for p in directory.glob("*.mex*")}
        self.assertEqual(actual, expected)

        package = REPO_ROOT / "dev" / "pysvreg"
        source = "\n".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in package.glob("*.py")
            if not path.name.startswith("test_")
        )
        compatibility = (package / "COMPATIBILITY.md").read_text(
            encoding="utf-8", errors="replace"
        )
        for name in expected:
            self.assertIn(name, compatibility)

        vendor_mex = list((REPO_ROOT / "dev" / "matlab_gifti").rglob("*.mex*"))
        self.assertEqual(len(vendor_mex), 12)
        self.assertFalse(
            any(
                re.search(r"\b(?:from|import)\s+(?:dev\.)?matlab_gifti\b", line)
                for line in source.splitlines()
            )
        )
        self.assertTrue((package / "dfsio.py").is_file())


if __name__ == "__main__":
    unittest.main()
