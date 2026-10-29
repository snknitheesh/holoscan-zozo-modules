# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ci.py: discovery, the dependency graph, selection, the ci mode, and empty
collections. No SDK needed."""

import contextlib
import json
import os
import shutil
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


def module(root: Path, name: str, deps: tuple[str, ...] = ()) -> None:
    metadata = {"module": {"name": f"holoscan-{name}", "dependencies": _entries(deps)}}
    write(root / "modules" / name / "metadata.json", json.dumps(metadata))
    write(root / "modules" / name / "CMakeLists.txt", _finds(deps))


def application(root: Path, name: str, modes: dict, deps: tuple[str, ...] = ()) -> None:
    metadata = {"application": {"name": name, "modes": modes}}
    if deps:
        metadata["application"]["dependencies"] = {"modules": _entries(deps)}
    write(root / "applications" / name / "metadata.json", json.dumps(metadata))
    write(root / "applications" / name / "CMakeLists.txt", _finds(deps))


def _entries(deps: tuple[str, ...]) -> list[dict]:
    return [{"name": dep} for dep in deps]


def _finds(deps: tuple[str, ...]) -> str:
    return "".join(f"find_package(holoscan-{dep} 1.0 REQUIRED CONFIG)\n" for dep in deps)


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


class BuildJobsTest(unittest.TestCase):
    def test_every_cpu_by_default(self):
        for unset in ({}, {"CI_BUILD_JOBS": ""}, {"CI_BUILD_JOBS": "  "}):
            with (
                self.subTest(env=unset),
                mock.patch.dict(os.environ, unset, clear=True),
                mock.patch.object(ci.os, "cpu_count", return_value=14),
            ):
                self.assertEqual(ci.build_jobs(), 14)

    def test_runner_sets_the_jobs(self):
        with mock.patch.dict(os.environ, {"CI_BUILD_JOBS": "8"}):
            self.assertEqual(ci.build_jobs(), 8)

    def test_bad_value_is_a_ci_error(self):
        for bad in ("0", "-2", "eight", "2.5"):
            with (
                self.subTest(value=bad),
                mock.patch.dict(os.environ, {"CI_BUILD_JOBS": bad}),
                self.assertRaisesRegex(ci.CiError, "CI_BUILD_JOBS"),
            ):
                ci.build_jobs()


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


class _Tree(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)

    def tearDown(self):
        self._directory.cleanup()


class GraphTest(_Tree):
    def test_modules_without_dependencies_build_in_name_order(self):
        module(self.root, "zeta")
        module(self.root, "alpha")
        self.assertEqual(list(ci.load_graph(self.root).modules), ["alpha", "zeta"])

    def test_modules_build_after_what_they_depend_on(self):
        module(self.root, "alpha", ("zeta",))
        module(self.root, "zeta", ("mid",))
        module(self.root, "mid")
        module(self.root, "beta")
        graph = ci.load_graph(self.root)
        self.assertEqual(list(graph.modules), ["beta", "mid", "zeta", "alpha"])
        self.assertEqual(graph.module_deps("alpha"), ("zeta",))

    def test_toposort(self):
        order = ci.toposort({"d": ("b", "c"), "c": ("a",), "b": ("a",), "a": ()})
        self.assertEqual(order, ["a", "b", "c", "d"])
        self.assertEqual(ci.toposort({}), [])

    def test_cycle_is_an_error_that_names_it(self):
        with self.assertRaisesRegex(ci.CiError, r"cycle: a -> b -> c -> a"):
            ci.toposort({"a": ("b",), "b": ("c",), "c": ("a",), "d": ()})
        # Only the cycle, not what merely depends on it.
        with self.assertRaisesRegex(ci.CiError, r"cycle: b -> c -> b$"):
            ci.toposort({"a": ("b",), "b": ("c",), "c": ("b",)})
        module(self.root, "one", ("two",))
        module(self.root, "two", ("one",))
        with self.assertRaisesRegex(ci.CiError, r"cycle: one -> two -> one"):
            ci.load_graph(self.root)

    def test_module_cannot_depend_on_itself(self):
        module(self.root, "alpha", ("alpha",))
        with self.assertRaisesRegex(ci.CiError, "depends on itself"):
            ci.load_graph(self.root)

    def test_unknown_dependency_is_an_error(self):
        module(self.root, "alpha", ("missing",))
        with self.assertRaisesRegex(ci.CiError, r"module.dependencies names 'missing'"):
            ci.load_graph(self.root)

    def test_unknown_application_dependency_is_an_error(self):
        module(self.root, "alpha")
        application(self.root, "demo", CI_MODE, ("missing-demo-dep",))
        with self.assertRaisesRegex(
            ci.CiError, r"application.dependencies.modules names 'missing-demo-dep'"
        ):
            ci.load_graph(self.root)

    def test_package_name_instead_of_folder_gets_a_hint(self):
        module(self.root, "point-cloud")
        application(self.root, "demo", CI_MODE, ("holoscan-point-cloud",))
        with self.assertRaisesRegex(ci.CiError, r"name the folder, 'point-cloud'"):
            ci.load_graph(self.root)

    def test_dependency_with_a_source_is_from_another_repository(self):
        module(self.root, "alpha")
        metadata = {
            "module": {
                "name": "holoscan-beta",
                "dependencies": [
                    {"name": "alpha"},
                    {"name": "holoscan-far", "source": {"git_url": "u", "ref": "0" * 40}},
                ],
            }
        }
        write(self.root / "modules" / "beta" / "metadata.json", json.dumps(metadata))
        write(self.root / "modules" / "beta" / "CMakeLists.txt")
        self.assertEqual(ci.load_graph(self.root).module_deps("beta"), ("alpha",))

    def test_malformed_dependencies_are_errors(self):
        module(self.root, "alpha")
        for dependencies, message in (
            ([{"name": "alpha"}], "application.dependencies must be an object"),
            ({"modules": "alpha"}, "application.dependencies.modules must be a list"),
            ({"modules": ["alpha"]}, "needs a name"),
        ):
            metadata = {"application": {"name": "d", "dependencies": dependencies}}
            write(self.root / "applications" / "d" / "metadata.json", json.dumps(metadata))
            write(self.root / "applications" / "d" / "CMakeLists.txt")
            with (
                self.subTest(dependencies=dependencies),
                self.assertRaisesRegex(ci.CiError, message),
            ):
                ci.load_graph(self.root)

    def test_cmake_use_must_be_declared(self):
        module(self.root, "alpha")
        module(self.root, "beta")
        application(self.root, "demo", CI_MODE, ("alpha",))
        write(
            self.root / "applications" / "demo" / "tests" / "CMakeLists.txt",
            "# find_package(holoscan-gamma) in a comment does not count\n"
            "find_package(holoscan 5.0 REQUIRED)\n"  # The SDK, not a module.
            "find_package(holoscan-cli)\n"  # Not a module of this collection.
            "find_package( HOLOSCAN-BETA 1.0 REQUIRED CONFIG)\n",
        )
        with self.assertRaisesRegex(
            ci.CiError,
            r'find holoscan-beta, which metadata.json does not declare; add \{"name": "beta"\}',
        ):
            ci.load_graph(self.root)

    def test_module_may_find_its_own_package(self):
        module(self.root, "alpha")
        write(
            self.root / "modules" / "alpha" / "cmake" / "holoscan-alpha-config.cmake.in",
            "find_dependency(holoscan-alpha)\n",
        )
        ci.load_graph(self.root)

    def test_dependencies_and_dependents_are_transitive(self):
        module(self.root, "base")
        module(self.root, "mid", ("base",))
        module(self.root, "top", ("mid",))
        module(self.root, "other")
        graph = ci.load_graph(self.root)
        self.assertEqual(graph.dependencies({"top"}), {"mid", "base"})
        self.assertEqual(graph.dependents({"base"}), {"mid", "top"})
        self.assertEqual(graph.dependents({"top"}), set())
        self.assertEqual(graph.dependencies({"other"}), set())


class SelectTest(_Tree):
    """base <- mid <- top, and other; each with a demo."""

    def setUp(self):
        super().setUp()
        module(self.root, "base")
        module(self.root, "mid", ("base",))
        module(self.root, "top", ("mid",))
        module(self.root, "other")
        application(self.root, "base_demo", CI_MODE, ("base",))
        application(self.root, "top_demo", CI_MODE, ("top",))
        application(self.root, "other_demo", CI_MODE, ("other",))
        write(self.root / "README.md")
        write(self.root / "modules" / "README.md")
        write(self.root / "modules" / "mid" / "README.md")
        write(self.root / "applications" / "top_demo" / "README.md")
        self.graph = ci.load_graph(self.root)

    def select(self, *paths):
        selection, why = ci.select(self.graph, self.root, list(paths))
        return selection, why

    def test_module_selects_its_dependents_and_their_demos(self):
        selection, why = self.select("modules/mid/src/mid.cpp", "modules/mid/tests/t.cpp")
        self.assertFalse(selection.everything)
        self.assertEqual(selection.modules, ["mid", "top"])
        self.assertEqual(selection.applications, ["top_demo"])
        # What they depend on is built, not tested.
        self.assertEqual(ci.to_build(self.graph, selection), ["base", "mid", "top"])
        self.assertIn("module mid: changed (modules/mid/src/mid.cpp)", why)
        self.assertIn("module top: depends on mid", why)
        self.assertIn("application top_demo: uses top", why)

    def test_module_at_the_bottom_selects_everything_above_it(self):
        selection, _ = self.select("modules/base/CMakeLists.txt")
        self.assertEqual(selection.modules, ["base", "mid", "top"])
        self.assertEqual(selection.applications, ["base_demo", "top_demo"])

    def test_demo_selects_itself_and_builds_its_dependencies(self):
        selection, why = self.select("applications/top_demo/src/main.cpp")
        self.assertEqual(selection.modules, [])
        self.assertEqual(selection.applications, ["top_demo"])
        self.assertEqual(ci.to_build(self.graph, selection), ["base", "mid", "top"])
        self.assertEqual(
            why, ["application top_demo: changed (applications/top_demo/src/main.cpp)"]
        )

    def test_module_and_demo_together(self):
        selection, _ = self.select("modules/other/a.cu", "applications/base_demo/metadata.json")
        self.assertEqual(selection.modules, ["other"])
        self.assertEqual(selection.applications, ["base_demo", "other_demo"])
        self.assertEqual(ci.to_build(self.graph, selection), ["base", "other"])

    def test_anything_else_selects_everything(self):
        for path in (
            ".github/workflows/ci.yml",
            ".github/scripts/ci.py",
            ".github/docker/Dockerfile",
            ".github/docker/requirements.txt",
            ".github/RUNNERS.md",  # Even documentation, under .github/.
            "CMakeLists.txt",
            "cmake/helpers.cmake",
            "requirements.txt",
            ".pre-commit-config.yaml",
            "LICENSE",
            "modules/gone/src/x.cpp",  # A module that no longer exists.
            "applications/gone/main.cpp",
            "modules/CMakeLists.txt",
            "modules/mid",
            "unknown/file",
        ):
            with self.subTest(path=path):
                selection, why = self.select("modules/mid/src/mid.cpp", path)
                self.assertTrue(selection.everything)
                self.assertEqual(selection.modules, ["base", "mid", "other", "top"])
                self.assertEqual(len(selection.applications), 3)
                self.assertEqual(why, [f"everything: {path} is not in a module's or demo's folder"])

    def test_documentation_only_selects_nothing(self):
        selection, why = self.select(
            "README.md",
            "modules/README.md",
            "modules/mid/README.md",
            "applications/top_demo/README.md",
        )
        self.assertTrue(selection.empty)
        self.assertEqual(ci.to_build(self.graph, selection), [])
        self.assertIn("documentation only, nothing to build: modules/mid/README.md", why)
        self.assertTrue(self.select()[0].empty)

    def test_documentation_with_code_selects_the_code(self):
        selection, _ = self.select("applications/top_demo/README.md", "modules/other/x.cpp")
        self.assertEqual(selection.modules, ["other"])
        self.assertEqual(selection.applications, ["other_demo"])

    def test_deleted_documentation_selects_its_folder(self):
        # A module installs its README: removing one can break its install.
        selection, _ = self.select("modules/base/README.md")
        self.assertEqual(selection.modules, ["base", "mid", "top"])

    def test_selection_round_trips_through_json(self):
        selection, _ = self.select("modules/mid/x.cpp")
        again = ci.Selection.from_json(selection.to_json(), self.graph)
        self.assertEqual(again, selection)
        full = ci.Selection.from_json(ci.everything(self.graph, True).to_json(), self.graph)
        self.assertTrue(full.everything and full.nightly)

    def test_bad_selection_is_an_error(self):
        for text, message in (
            ("{", "not what"),
            ("{}", "not what"),
            ('{"everything": false, "modules": ["nope"], "applications": []}', "no such module"),
            ('{"everything": false, "modules": [], "applications": ["x"]}', "no such application"),
        ):
            with self.subTest(text=text), self.assertRaisesRegex(ci.CiError, message):
                ci.Selection.from_json(text, self.graph)

    def test_everything_selection_takes_the_tree_as_it_is(self):
        text = '{"everything": true, "modules": [], "applications": []}'
        self.assertEqual(ci.Selection.from_json(text, self.graph), ci.everything(self.graph))


@unittest.skipUnless(shutil.which("git"), "needs git")
class ChangedFilesTest(_Tree):
    def git(self, *args):
        subprocess.run(
            ["git", "-C", str(self.root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
            check=True,
            capture_output=True,
        )

    def commit(self, path, content):
        write(self.root / path, content)
        self.git("add", "-A")
        self.git("commit", "-q", "-m", path)

    def setUp(self):
        super().setUp()
        self.git("init", "-q", "-b", "main")
        module(self.root, "alpha")
        module(self.root, "beta")
        application(self.root, "alpha_demo", CI_MODE, ("alpha",))
        write(self.root / "README.md", "x")
        self.commit("modules/alpha/README.md", "a")
        self.git("checkout", "-q", "-b", "topic")
        self.commit("modules/alpha/src/a.cpp", "1")
        self.git("checkout", "-q", "main")
        # On the base only, after the branch point: not the pull request's change.
        self.commit("modules/beta/src/b.cpp", "2")
        self.git("checkout", "-q", "topic")

    def test_diff_from_the_merge_base(self):
        self.assertEqual(ci.changed_files(self.root, "main"), ["modules/alpha/src/a.cpp"])

    def test_bad_base_is_an_error(self):
        for base in ("no-such-ref", "--output=x", ""):
            with self.subTest(base=base), self.assertRaises(ci.CiError):
                ci.changed_files(self.root, base)

    def run_select(self, *args):
        output = self.root.parent / f"{self.root.name}.out"
        self.addCleanup(output.unlink, missing_ok=True)
        with contextlib.redirect_stdout(open(os.devnull, "w")) as devnull:
            self.addCleanup(devnull.close)
            ci.select_command(self.root, *args, output)
        lines = dict(line.split("=", 1) for line in output.read_text().splitlines())
        return json.loads(lines["selection"]), lines["gpu"]

    def test_select_writes_the_selection_for_the_workflow(self):
        selection, gpu = self.run_select("main", False)
        self.assertEqual(selection["modules"], ["alpha"])
        self.assertEqual(selection["applications"], ["alpha_demo"])
        self.assertFalse(selection["everything"])
        self.assertEqual(gpu, "true")

    def test_documentation_only_needs_no_gpu(self):
        self.git("checkout", "-q", "-b", "docs", "main")
        self.commit("README.md", "changed")
        self.commit("modules/alpha/README.md", "changed")
        selection, gpu = self.run_select("main", False)
        self.assertEqual((selection["modules"], selection["applications"]), ([], []))
        self.assertEqual(gpu, "false")

    def test_no_base_unknown_base_and_nightly_select_everything(self):
        for base, nightly in ((None, False), ("no-such-ref", False), ("main", True)):
            with self.subTest(base=base, nightly=nightly):
                selection, gpu = self.run_select(base, nightly)
                self.assertTrue(selection["everything"])
                self.assertEqual(selection["nightly"], nightly)
                self.assertEqual(gpu, "true")


@unittest.skipUnless(shutil.which("cmake"), "needs cmake")
class RootCMakeTest(_Tree):
    """The root CMakeLists.txt adds the modules in dependency order, as CI builds them."""

    def setUp(self):
        super().setUp()
        shutil.copy(Path(ci.ROOT) / "CMakeLists.txt", self.root / "CMakeLists.txt")

    def stub(self, name, deps=()):
        target = name.replace("-", "_")
        metadata = {"module": {"name": f"holoscan-{name}", "dependencies": _entries(deps)}}
        write(self.root / "modules" / name / "metadata.json", json.dumps(metadata))
        write(
            self.root / "modules" / name / "CMakeLists.txt",
            f"project({target} LANGUAGES NONE)\n"
            f"add_library({target} INTERFACE)\n"
            f"add_library(holoscan::{target} ALIAS {target})\n"
            f'message(STATUS "added {name}")\n',
        )

    def configure(self):
        result = subprocess.run(
            ["cmake", "-S", str(self.root), "-B", str(self.root / "build")],
            check=False,
            capture_output=True,
            text=True,
        )
        added = [line.split()[-1] for line in result.stdout.splitlines() if "-- added " in line]
        return result.returncode, added, result.stderr

    def test_dependencies_first(self):
        self.stub("alpha", ("zeta",))
        self.stub("zeta", ("mid",))
        self.stub("mid")
        self.stub("beta")
        code, added, stderr = self.configure()
        self.assertEqual(code, 0, stderr)
        self.assertEqual(added, ["mid", "zeta", "alpha", "beta"])

    def test_cycle_fails(self):
        self.stub("alpha", ("beta",))
        self.stub("beta", ("alpha",))
        code, _, stderr = self.configure()
        self.assertNotEqual(code, 0)
        self.assertIn("cycle: alpha -> beta -> alpha", " ".join(stderr.split()))

    def test_unknown_dependency_fails(self):
        self.stub("alpha", ("missing",))
        code, _, stderr = self.configure()
        self.assertNotEqual(code, 0)
        self.assertIn("names 'missing', which is not a module", " ".join(stderr.split()))


class SdkTest(unittest.TestCase):
    """What `ci.py sdk` passes to CMake and to the ci mode, with the builds faked."""

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.builds = []
        self.build_dirs = []
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
        self.build_dirs.append((Path(build).name, Path(build)))

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

    def test_full_run_tests_every_module_in_dependency_order(self):
        module(self.root, "alpha", ("beta",))
        module(self.root, "beta")
        for name in ("alpha", "beta"):
            write(self.root / "modules" / name / "tests" / "installed_consumer" / "CMakeLists.txt")
        ci.sdk(self.root, self.root / "build", "x86_64")
        self.assertEqual(
            [Path(build).parent.name + "/" + name for name, build in self.build_dirs],
            [
                "modules/beta",
                "consumers/beta",
                "modules/alpha",
                "consumers/alpha",
            ],
        )
        self.assertEqual(os.environ["HOLOSCAN_CI_NIGHTLY"], "0")

    def test_selection_builds_dependencies_without_their_tests(self):
        module(self.root, "base")
        module(self.root, "top", ("base",))
        module(self.root, "other")
        for name in ("base", "top", "other"):
            write(self.root / "modules" / name / "tests" / "installed_consumer" / "CMakeLists.txt")
        application(self.root, "top_demo", CI_MODE, ("top",))
        application(self.root, "other_demo", CI_MODE, ("other",))
        selection = '{"everything": false, "modules": ["top"], "applications": ["top_demo"]}'
        ci.sdk(self.root, self.root / "build", "jetson", selection)
        # top and its consumer with their tests, then the demo; base only built.
        self.assertEqual(
            [name for name, _ in self.builds], ["top", "installed_consumer", "top_demo"]
        )
        commands = [command for command, _ in self.runs]
        base_configure = [c for c in commands if c[:2] == ["cmake", "-S"]]
        self.assertEqual(len(base_configure), 1)
        self.assertEqual(Path(base_configure[0][2]).name, "base")
        self.assertIn("-DBUILD_TESTING=OFF", base_configure[0])
        installs = [Path(c[2]).name for c in commands if c[:2] == ["cmake", "--install"]]
        self.assertEqual(installs, ["base", "top"])
        self.assertFalse(any("other" in " ".join(map(str, c)) for c in commands))
        self.assertEqual(os.environ["HOLOSCAN_CI_PLATFORM"], "jetson")

    def test_empty_selection_builds_nothing(self):
        module(self.root, "alpha")
        application(self.root, "demo", CI_MODE, ("alpha",))
        selection = '{"everything": false, "modules": [], "applications": []}'
        ci.sdk(self.root, self.root / "build", "x86_64", selection)
        self.assertEqual((self.builds, self.runs), ([], []))

    def test_nightly_is_exported(self):
        selection = '{"everything": true, "nightly": true, "modules": [], "applications": []}'
        ci.sdk(self.root, self.root / "build", "x86_64", selection)
        self.assertEqual(os.environ["HOLOSCAN_CI_NIGHTLY"], "1")

    def test_host_runs_only_the_selected_suites(self):
        for name in ("alpha", "beta"):
            module(self.root, name)
            write(self.root / "modules" / name / "tests" / "host" / "CMakeLists.txt")
        ci.host(
            self.root,
            self.root / "build",
            '{"everything": false, "modules": ["beta"], "applications": []}',
        )
        self.assertEqual([name for name, _ in self.builds], ["host"])
        self.builds.clear()
        ci.host(self.root, self.root / "build")
        self.assertEqual(len(self.builds), 2)


if __name__ == "__main__":
    unittest.main()
