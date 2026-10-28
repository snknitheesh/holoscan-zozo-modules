# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ci.py: discovery, the ci mode, and empty collections. No SDK needed."""

import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

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
        (self.root / "modules" / ".hidden").mkdir()
        names = [p.name for p in ci.discover(self.root, "modules", "module")]
        self.assertEqual(names, ["alpha", "zeta"])

    def test_rejects_folder_without_metadata(self):
        # A misspelt or forgotten metadata.json must fail the run, not skip the folder.
        module(self.root, "alpha")
        write(self.root / "modules" / "typo" / "metadata.jsn", "{}")
        write(self.root / "modules" / "typo" / "CMakeLists.txt")
        with self.assertRaisesRegex(ci.CiError, "typo: has no metadata.json"):
            ci.discover(self.root, "modules", "module")
        (self.root / "applications" / "empty").mkdir(parents=True)
        with self.assertRaisesRegex(ci.CiError, "empty: has no metadata.json"):
            ci.discover(self.root, "applications", "application")

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
    def test_malformed_run_is_a_ci_error(self):
        for run in ("demo", ["demo"], None, {"command": ["demo"]}):
            app = ci.Project("application", "demo", Path("/a/demo"), {})
            app.metadata["application"] = {"modes": {"ci": {"run": run}}}
            with self.subTest(run=run), self.assertRaisesRegex(ci.CiError, "modes.ci.run.command"):
                ci.ci_mode(app)

    def test_timeout(self):
        self.assertEqual(ci.ci_timeout({"run": {"command": "x"}}), ci.CI_MODE_TIMEOUT)
        self.assertEqual(ci.ci_timeout({"run": {"command": "x", "timeout": 30}}), 30)
        self.assertEqual(ci.ci_timeout({"run": {"command": "x", "timeout": 2.5}}), 2.5)
        for bad in (0, -1, "60", True, None, float("nan"), float("inf")):
            with self.subTest(timeout=bad), self.assertRaisesRegex(ci.CiError, "timeout"):
                ci.ci_timeout({"run": {"command": "x", "timeout": bad}})

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


class RunTest(unittest.TestCase):
    def test_success(self):
        ci.run([sys.executable, "-c", "pass"], timeout=30)

    def test_failure_is_a_ci_error(self):
        with self.assertRaisesRegex(ci.CiError, "exit status 3"):
            ci.run([sys.executable, "-c", "raise SystemExit(3)"])

    def test_timeout_kills_the_whole_process_group(self):
        # The command starts a grandchild that would outlive a kill of the child alone.
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / "pid"
            script = (
                "import subprocess, sys, time\n"
                "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                f"open({str(pid_file)!r}, 'w').write(str(p.pid))\n"
                "time.sleep(60)\n"
            )
            started = time.monotonic()
            with self.assertRaisesRegex(ci.CiError, "timed out after 1 s"):
                ci.run([sys.executable, "-c", script], timeout=1)
            self.assertLess(time.monotonic() - started, 20)
            grandchild = int(pid_file.read_text())
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and _alive(grandchild):
                time.sleep(0.05)
            self.assertFalse(_alive(grandchild), "the grandchild survived the timeout")

    def test_timeout_kills_what_ignores_sigterm(self):
        # The grandchild ignores SIGTERM; the leader exits on it. SIGKILL must follow.
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / "pid"
            command = [sys.executable, "-c", _spawner(pid_file, ignore_sigterm=True)]
            with self.assertRaisesRegex(ci.CiError, "timed out"):
                ci.run(command, timeout=1)
            _assert_gone(self, int(pid_file.read_text()))

    def test_command_without_timeout_stays_in_the_callers_group(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "pgid"
            script = f"import os; open({str(out)!r}, 'w').write(str(os.getpgid(0)))"
            ci.run([sys.executable, "-c", script])
            self.assertEqual(int(out.read_text()), os.getpgid(0))
            ci.run([sys.executable, "-c", script], timeout=30)
            self.assertNotEqual(int(out.read_text()), os.getpgid(0))

    def test_signal_to_ci_stops_the_command(self):
        # ci.py itself is told to stop while a command runs: the command and what it
        # started must not outlive it. Runs ci.py in a child, so the signal is its own.
        for timeout, sig in ((None, signal.SIGTERM), (60, signal.SIGTERM), (60, signal.SIGHUP)):
            with (
                self.subTest(timeout=timeout, signal=sig.name),
                tempfile.TemporaryDirectory() as directory,
            ):
                pid_file = Path(directory) / "pid"
                leader_file = Path(directory) / "leader"
                spawner = _spawner(pid_file, leader_file=leader_file)
                driver = (
                    "import sys\n"
                    f"sys.path.insert(0, {str(Path(ci.__file__).parent)!r})\n"
                    "import ci\n"
                    "ci.exit_on_signals()\n"
                    f"ci.run([sys.executable, '-c', {spawner!r}], timeout={timeout})\n"
                )
                process = subprocess.Popen([sys.executable, "-c", driver])
                try:
                    _wait_for(pid_file)
                    process.send_signal(sig)
                    self.assertEqual(process.wait(timeout=30), 128 + sig)
                finally:
                    process.kill()
                    process.wait()
                _assert_gone(self, int(leader_file.read_text()))
                if timeout is not None:
                    # In its own group, everything the command started goes too.
                    _assert_gone(self, int(pid_file.read_text()))
                else:
                    with contextlib.suppress(ProcessLookupError):
                        os.kill(int(pid_file.read_text()), signal.SIGKILL)


def _spawner(pid_file: Path, *, ignore_sigterm: bool = False, leader_file: Path | None = None):
    """A command that starts a grandchild, records its PID, and sleeps."""
    child = "import signal, time\n"
    if ignore_sigterm:
        child += "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    child += f"open({str(pid_file) + '.tmp'!r}, 'w').write('x')\ntime.sleep(60)\n"
    script = (
        "import os, subprocess, sys, time\n"
        f"p = subprocess.Popen([sys.executable, '-c', {child!r}])\n"
        f"while not os.path.exists({str(pid_file) + '.tmp'!r}): time.sleep(0.02)\n"
    )
    if leader_file is not None:
        script += f"open({str(leader_file)!r}, 'w').write(str(os.getpid()))\n"
    script += (
        f"open({str(pid_file) + '.part'!r}, 'w').write(str(p.pid))\n"
        f"os.replace({str(pid_file) + '.part'!r}, {str(pid_file)!r})\n"
        "time.sleep(60)\n"
    )
    return script


def _wait_for(path: Path, seconds: float = 20) -> None:
    deadline = time.monotonic() + seconds
    while not path.exists():
        if time.monotonic() > deadline:
            raise AssertionError(f"{path} never appeared")
        time.sleep(0.02)


def _assert_gone(test: unittest.TestCase, pid: int) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and _alive(pid):
        time.sleep(0.05)
    test.assertFalse(_alive(pid), f"process {pid} survived")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A killed child of an exited parent may linger as a zombie until init reaps it.
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as stat:
            return stat.read().split(") ")[-1].split()[0] != "Z"
    except OSError:
        return False


class SdkTest(unittest.TestCase):
    """What `ci.py sdk` passes to CMake and to the ci mode, with the builds faked."""

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.builds = []
        self.runs = []
        patches = [
            mock.patch.object(ci, "cmake_build_and_test", self._build),
            mock.patch.object(ci, "run", self._run),
            mock.patch.dict(os.environ),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def tearDown(self):
        self._directory.cleanup()

    def _build(self, source, build, options, no_tests):
        self.builds.append((Path(source).name, options))

    def _run(self, command, *, cwd=None, timeout=None):
        self.runs.append((command, timeout))

    def test_modules_find_the_modules_installed_before_them(self):
        module(self.root, "alpha")
        write(self.root / "modules" / "alpha" / "tests" / "installed_consumer" / "CMakeLists.txt")
        ci.sdk(self.root, self.root / "build", "x86_64")
        prefix = self.root / "build" / "install"
        (name, options) = self.builds[0]
        self.assertEqual(name, "alpha")
        self.assertIn(f"-DCMAKE_PREFIX_PATH={prefix}", options)
        self.assertIn(f"-DCMAKE_INSTALL_PREFIX={prefix}", options)

    def test_ci_mode_runs_with_its_timeout(self):
        application(self.root, "demo", CI_MODE)
        application(self.root, "slow", {"ci": {"run": {"command": "slow", "timeout": 1800}}})
        ci.sdk(self.root, self.root / "build", "x86_64")
        self.assertEqual([timeout for _, timeout in self.runs], [ci.CI_MODE_TIMEOUT, 1800])


if __name__ == "__main__":
    unittest.main()
