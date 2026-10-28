# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for release.py: tags, the package build and the consumer, with the builds
and dpkg faked. No SDK needed."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import release
from ci import CiError


def module(root: Path, folder: str, version: str = "1.2.3", **fields) -> None:
    metadata = {"module": {"name": f"holoscan-{folder}", "version": version, **fields}}
    path = root / "modules" / folder
    path.mkdir(parents=True)
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (path / "CMakeLists.txt").write_text("", encoding="utf-8")
    consumer = path / "tests" / "installed_consumer"
    consumer.mkdir(parents=True)
    (consumer / "CMakeLists.txt").write_text("", encoding="utf-8")


class Fixture(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)

    def tearDown(self):
        self._directory.cleanup()


class TagTest(Fixture):
    def test_tag_names_the_module_and_version(self):
        module(self.root, "apriltag", "1.0.0")
        project, version = release.parse_tag(self.root, "apriltag-v1.0.0")
        self.assertEqual((project.name, version), ("apriltag", "1.0.0"))

    def test_module_names_may_have_hyphens(self):
        module(self.root, "point-cloud", "2.10.0")
        project, version = release.parse_tag(self.root, "point-cloud-v2.10.0")
        self.assertEqual((project.name, version), ("point-cloud", "2.10.0"))

    def test_malformed_tags_fail(self):
        module(self.root, "apriltag", "1.0.0")
        for tag in (
            "apriltag-1.0.0",
            "v1.0.0",
            "apriltag-v1.0",
            "apriltag-v01.0.0",
            "apriltag-v1.0.0-rc.1",
            "Apriltag-v1.0.0",
        ):
            with self.subTest(tag=tag), self.assertRaisesRegex(CiError, "is not <module>-v"):
                release.parse_tag(self.root, tag)

    def test_unknown_module_fails(self):
        module(self.root, "apriltag", "1.0.0")
        with self.assertRaisesRegex(CiError, "no module modules/yolo26/"):
            release.parse_tag(self.root, "yolo26-v1.0.0")

    def test_version_must_match_the_metadata(self):
        module(self.root, "apriltag", "1.0.0")
        with self.assertRaisesRegex(CiError, "the tag is for 1.0.1"):
            release.parse_tag(self.root, "apriltag-v1.0.1")

    def test_main_prints_github_outputs(self):
        module(self.root, "point-cloud", "1.0.0")
        output = io.StringIO()
        with mock.patch.object(release, "ROOT", self.root), contextlib.redirect_stdout(output):
            self.assertEqual(release.main(["tag", "point-cloud-v1.0.0"]), 0)
        self.assertEqual(
            output.getvalue().splitlines(),
            ["module=point-cloud", "version=1.0.0", "package=holoscan-point-cloud"],
        )


class PackageTest(Fixture):
    def setUp(self):
        super().setUp()
        self.commands = []
        self.fields = None
        for patch in (
            mock.patch.object(release, "run", self._run),
            mock.patch.object(release, "deb_architecture", return_value="arm64"),
            mock.patch.object(release, "deb_fields", lambda deb: self.fields),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def _run(self, command, **_kwargs):
        self.commands.append(command)
        if command[0] == "cmake" and "-B" in command and "-S" in command:
            # The module's pkg/ configuration writes the CPack configuration.
            build = Path(command[command.index("-B") + 1])
            (build / "pkg").mkdir(parents=True, exist_ok=True)
            (build / "pkg" / "CPackConfig-holoscan-point-cloud.cmake").write_text("")
        if command[0] == "cpack":
            staging = Path(command[command.index("-B") + 1])
            staging.mkdir(parents=True, exist_ok=True)
            (staging / "holoscan-point-cloud_1.0.0_arm64.deb").write_text("deb")

    def package(self):
        return release.package(
            self.root, "point-cloud", "1.0.0", self.root / "dist", self.root / "build"
        )

    def test_builds_the_package_configuration_and_checks_the_deb(self):
        module(self.root, "point-cloud", "1.0.0")
        self.fields = {
            "Package": "holoscan-point-cloud",
            "Version": "1.0.0",
            "Architecture": "arm64",
        }
        deb = self.package()
        self.assertEqual(deb, self.root / "dist" / "holoscan-point-cloud_1.0.0_arm64.deb")
        self.assertTrue(deb.is_file())
        configure = self.commands[0]
        self.assertIn(str(self.root / "modules" / "point-cloud"), configure)
        self.assertIn("-DPKG_holoscan_point_cloud=ON", configure)
        self.assertIn("-DBUILD_TESTING=OFF", configure)
        self.assertEqual(self.commands[2][:2], ["cpack", "--config"])

    def test_wrong_control_fields_fail(self):
        module(self.root, "point-cloud", "1.0.0")
        self.fields = {
            "Package": "holoscan-point-cloud",
            "Version": "0.9.0",
            "Architecture": "arm64",
        }
        with self.assertRaisesRegex(CiError, "control fields"):
            self.package()

    def test_version_must_match_the_metadata(self):
        module(self.root, "point-cloud", "0.9.0")
        with self.assertRaisesRegex(CiError, "module.version is '0.9.0'"):
            self.package()
        self.assertEqual(self.commands, [])


class ConsumeTest(Fixture):
    def test_installs_the_file_and_builds_the_consumer_without_a_prefix(self):
        module(self.root, "apriltag", "1.0.0")
        commands = []
        with mock.patch.object(release, "run", lambda command, **_: commands.append(command)):
            release.consume(
                self.root, "apriltag", Path("dist/holoscan-apriltag_1.0.0_amd64.deb"), self.root
            )
        install = next(c for c in commands if c[:2] == ["apt-get", "install"])
        self.assertTrue(Path(install[-1]).is_absolute())
        configure = next(c for c in commands if c[0] == "cmake" and "-S" in c)
        self.assertEqual(
            Path(configure[configure.index("-S") + 1]),
            self.root / "modules" / "apriltag" / "tests" / "installed_consumer",
        )
        self.assertFalse(any("CMAKE_PREFIX_PATH" in argument for argument in configure))
        self.assertEqual(commands[-1][0], "ctest")


if __name__ == "__main__":
    unittest.main()
