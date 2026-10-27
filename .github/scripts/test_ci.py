# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ci.py: discovery, the ci mode, and empty collections. No SDK needed."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ci


def write(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def module(root: Path, name: str) -> None:
    write(root / "modules" / name / "metadata.json", json.dumps({"module": {"name": name}}))
    write(root / "modules" / name / "CMakeLists.txt")


def application(root: Path, name: str, modes: dict) -> None:
    metadata = {"application": {"name": name, "modes": modes}}
    write(root / "applications" / name / "metadata.json", json.dumps(metadata))
    write(root / "applications" / name / "CMakeLists.txt")


CI_MODE = {
    "ci": {"run": {"command": '"<holohub_app_bin>/demo" --count 3', "workdir": "holohub_app_bin"}}
}


class DiscoveryTest(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)

    def tearDown(self):
        self._directory.cleanup()

    def test_empty_collection_discovers_nothing(self):
        (self.root / "modules").mkdir()
        self.assertEqual(ci.discover(self.root, "modules", "module"), [])
        self.assertEqual(ci.discover(self.root, "applications", "application"), [])

    def test_empty_collection_builds_nothing(self):
        build = self.root / "build"
        self.assertEqual(ci.host(self.root, build), 0)
        self.assertEqual(ci.sdk(self.root, build, "x86_64"), 0)
        self.assertFalse(build.exists())

    def test_finds_modules_in_name_order(self):
        module(self.root, "zeta")
        module(self.root, "alpha")
        write(self.root / "modules" / "README.md")
        (self.root / "modules" / "no_metadata").mkdir()
        names = [p.name for p in ci.discover(self.root, "modules", "module")]
        self.assertEqual(names, ["alpha", "zeta"])

    def test_rejects_wrong_envelope(self):
        write(self.root / "modules" / "x" / "metadata.json", json.dumps({"operator": {}}))
        write(self.root / "modules" / "x" / "CMakeLists.txt")
        with self.assertRaisesRegex(ci.CiError, "'module' metadata envelope"):
            ci.discover(self.root, "modules", "module")

    def test_rejects_missing_cmakelists(self):
        write(self.root / "modules" / "x" / "metadata.json", json.dumps({"module": {}}))
        with self.assertRaisesRegex(ci.CiError, "no CMakeLists.txt"):
            ci.discover(self.root, "modules", "module")

    def test_application_needs_ci_mode(self):
        application(self.root, "demo", {"default": {"run": {"command": "demo"}}})
        (app,) = ci.discover(self.root, "applications", "application")
        with self.assertRaisesRegex(ci.CiError, "modes.ci.run.command"):
            ci.ci_mode(app)
        # The run fails before building anything.
        with self.assertRaisesRegex(ci.CiError, "modes.ci.run.command"):
            ci.sdk(self.root, self.root / "build", "x86_64")


class CiModeTest(unittest.TestCase):
    def test_command_resolves_the_build_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            application(root, "demo", CI_MODE)
            (app,) = ci.discover(root, "applications", "application")
            command, cwd = ci.ci_command(ci.ci_mode(app), app, Path("/b/demo"))
            self.assertEqual(command, ["/b/demo/demo", "--count", "3"])
            self.assertEqual(cwd, Path("/b/demo"))

    def test_cmake_options(self):
        mode = {"build": {"cmake_options": ["-DA=ON"]}, "run": {"command": "x"}}
        self.assertEqual(ci.ci_cmake_options(mode), ["-DA=ON"])
        self.assertEqual(ci.ci_cmake_options({"run": {"command": "x"}}), [])
        with self.assertRaises(ci.CiError):
            ci.ci_cmake_options({"build": {"cmake_options": "-DA=ON"}})


if __name__ == "__main__":
    unittest.main()
