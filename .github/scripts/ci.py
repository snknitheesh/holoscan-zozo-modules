# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0
"""Build and test every module under modules/ and every demo under applications/.

Each folder plugs in by layout alone (see modules/README.md and
applications/README.md), so adding one needs no workflow change:

  host   Build and run the SDK-free test projects (<folder>/tests/host/).
  sdk    For each module: build it, run its full CTest suite, install it, then
         build and test its installed-package consumer (tests/installed_consumer/).
         Modules build in name order, each against those installed before it.
         Then, for each demo: build it against the installed modules, run its
         tests, and run its finite `ci` mode from metadata.json, which is killed
         after modes.ci.run.timeout seconds (default 900).
  list   Print what would be built.

Every step must succeed; the first failure ends the run with a nonzero status.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP_BIN = "<holohub_app_bin>"
# Seconds a demo's ci mode may run, unless its modes.ci.run.timeout says otherwise.
CI_MODE_TIMEOUT = 900


class CiError(Exception):
    """A discovery or build failure, reported without a traceback."""


@dataclass(frozen=True)
class Project:
    kind: str  # "module" or "application"
    name: str
    path: Path
    metadata: dict

    @property
    def label(self) -> str:
        return f"{self.kind} {self.name}"


def discover(root: Path, folder: str, kind: str) -> list[Project]:
    """Every <root>/<folder>/<name>/, sorted by name. Each must have a metadata.json:
    a folder without one would otherwise never be built or tested."""
    base = root / folder
    if not base.is_dir():
        return []
    projects = []
    for path in sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")):
        metadata_file = path / "metadata.json"
        if not metadata_file.is_file():
            raise CiError(f"{path}: has no metadata.json")
        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise CiError(f"{metadata_file}: {error}") from error
        if not isinstance(metadata, dict) or kind not in metadata:
            raise CiError(f"{metadata_file}: expected a '{kind}' metadata envelope")
        if not (path / "CMakeLists.txt").is_file():
            raise CiError(f"{path}: has metadata.json but no CMakeLists.txt")
        projects.append(Project(kind, path.name, path, metadata))
    return projects


def ci_mode(app: Project) -> dict:
    """The application's `ci` mode: a finite, headless run."""
    modes = app.metadata["application"].get("modes", {})
    mode = modes.get("ci") if isinstance(modes, dict) else None
    run = mode.get("run") if isinstance(mode, dict) else None
    if not isinstance(run, dict) or not isinstance(run.get("command"), str):
        raise CiError(f"{app.path / 'metadata.json'}: needs modes.ci.run.command")
    try:
        ci_timeout(mode)
    except CiError as error:
        raise CiError(f"{app.path / 'metadata.json'}: {error}") from error
    return mode


def ci_timeout(mode: dict) -> float:
    """Seconds the ci mode may run: modes.ci.run.timeout, or CI_MODE_TIMEOUT."""
    timeout = mode["run"].get("timeout", CI_MODE_TIMEOUT)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, int | float)
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise CiError("modes.ci.run.timeout must be a positive number of seconds")
    return timeout


def ci_command(mode: dict, app: Project, app_bin: Path) -> tuple[list[str], Path]:
    """The ci mode's command and working directory, with <holohub_app_bin> resolved.

    <holohub_app_bin> is the demo's build directory. Any other workdir is relative to
    the demo's source folder.
    """
    run = mode["run"]
    command = shlex.split(run["command"].replace(APP_BIN, str(app_bin)))
    workdir = run.get("workdir", "holohub_app_bin")
    cwd = app_bin if workdir in ("holohub_app_bin", APP_BIN) else app.path / workdir
    return command, cwd


def ci_cmake_options(mode: dict) -> list[str]:
    options = mode.get("build", {}).get("cmake_options", [])
    if not isinstance(options, list) or not all(isinstance(option, str) for option in options):
        raise CiError("modes.ci.build.cmake_options must be a list of strings")
    return options


def run(command: list[str], *, cwd: Path | None = None, timeout: float | None = None) -> None:
    """Run a command; any nonzero status fails the run. With a timeout, the command
    and everything it started are killed when it expires, and the run fails."""
    print(f"+ {shlex.join(command)}", flush=True)
    started = time.monotonic()
    # With a timeout, its own process group, so that the kill reaches the command's
    # children too. Without one, it stays in ci.py's group: a signal to that group
    # (Ctrl-C, a hang-up, `timeout`), or a SIGKILL of ci.py's group, reaches all of it,
    # while a signal to ci.py's PID alone stops only the command, not what it started.
    group = timeout is not None
    process = subprocess.Popen(command, cwd=cwd, start_new_session=group)
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        stop(process, group)
        raise CiError(f"timed out after {timeout:g} s: {shlex.join(command)}") from None
    except BaseException:
        # Interrupted: Ctrl-C, or SIGTERM or SIGHUP through exit_on_signals().
        stop(process, group)
        raise
    if returncode:
        raise CiError(f"exit status {returncode}: {shlex.join(command)}")
    print(f"  ({time.monotonic() - started:.1f} s)", flush=True)


def stop(process: subprocess.Popen, group: bool, grace: float = 10) -> None:
    """SIGTERM the command (its whole group if it has one), give it `grace` seconds,
    then SIGKILL whatever is left: what ignored SIGTERM, or outlived the leader."""
    if process.returncode is not None:
        return  # Already reaped: its PID, and so its group ID, may now be another's.

    def send(sig: signal.Signals) -> None:
        with contextlib.suppress(ProcessLookupError):
            if group:
                os.killpg(process.pid, sig)
            else:
                process.send_signal(sig)

    try:
        send(signal.SIGTERM)
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=grace)
    finally:
        # Also when a second Ctrl-C interrupts the wait.
        send(signal.SIGKILL)
        process.wait()


def exit_on_signals() -> None:
    """Turn SIGTERM and SIGHUP into SystemExit, so that run() stops its command
    instead of leaving it running when ci.py is told to stop."""

    def handler(signum: int, frame: object) -> None:
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, handler)


def section(title: str) -> None:
    print(f"\n=== {title}", flush=True)


def generator() -> list[str]:
    return ["-G", "Ninja"] if shutil.which("ninja") else []


def build_jobs() -> int:
    """Parallel build jobs: CI_BUILD_JOBS when the runner sets it (a machine that also
    serves other work keeps cores and memory for it), else every CPU."""
    value = os.environ.get("CI_BUILD_JOBS", "").strip()
    if not value:
        return os.cpu_count() or 2
    if not value.isdigit() or int(value) < 1:
        raise CiError(f"CI_BUILD_JOBS must be a positive whole number, not {value!r}")
    return int(value)


def cmake_build_and_test(source: Path, build: Path, options: list[str], no_tests: str) -> None:
    run(["cmake", "-S", str(source), "-B", str(build), *generator(), *options])
    run(["cmake", "--build", str(build), "--parallel", str(build_jobs())])
    run(
        [
            "ctest",
            "--test-dir",
            str(build),
            "--output-on-failure",
            f"--no-tests={no_tests}",
            "--timeout",
            "1200",
        ]
    )


def host(root: Path, build_root: Path) -> int:
    """Build and run each folder's SDK-free tests/host/ project."""
    projects = discover(root, "modules", "module") + discover(root, "applications", "application")
    suites = [p for p in projects if (p.path / "tests" / "host" / "CMakeLists.txt").is_file()]
    print(f"Discovered {len(projects)} project(s), {len(suites)} with SDK-free tests.")
    for project in suites:
        section(f"{project.label}: SDK-free tests")
        cmake_build_and_test(
            project.path / "tests" / "host",
            build_root / "host" / project.kind / project.name,
            ["-DCMAKE_BUILD_TYPE=Release"],
            no_tests="error",
        )
    if not suites:
        print("No SDK-free test projects; nothing to run.")
    return 0


def sdk(root: Path, build_root: Path, platform: str) -> int:
    """Build, test, install and consume every module, then build and run every demo."""
    modules = discover(root, "modules", "module")
    apps = discover(root, "applications", "application")
    modes = {app.name: ci_mode(app) for app in apps}  # Fail before building anything.
    print(f"Platform {platform}: {len(modules)} module(s), {len(apps)} application(s).")
    prefix = build_root / "install"
    # Module tests and ci modes may check what holds only on one platform.
    os.environ["HOLOSCAN_CI_PLATFORM"] = platform
    release = ["-DCMAKE_BUILD_TYPE=Release"]
    for module in modules:
        build = build_root / "modules" / module.name
        section(f"{module.label}: build and full test suite")
        cmake_build_and_test(
            module.path,
            build,
            [
                *release,
                "-DBUILD_TESTING=ON",
                f"-DCMAKE_INSTALL_PREFIX={prefix}",
                # The modules installed before this one, in name order.
                f"-DCMAKE_PREFIX_PATH={prefix}",
            ],
            no_tests="error",
        )
        section(f"{module.label}: install")
        run(["cmake", "--install", str(build)])
        consumer = module.path / "tests" / "installed_consumer"
        if not (consumer / "CMakeLists.txt").is_file():
            raise CiError(f"{module.path}: needs tests/installed_consumer/CMakeLists.txt")
        section(f"{module.label}: consumer of the installed package")
        cmake_build_and_test(
            consumer,
            build_root / "consumers" / module.name,
            [*release, f"-DCMAKE_PREFIX_PATH={prefix}"],
            no_tests="error",
        )
    for app in apps:
        build = build_root / "applications" / app.name
        mode = modes[app.name]
        section(f"{app.label}: build against the installed modules and test")
        cmake_build_and_test(
            app.path,
            build,
            [
                *release,
                "-DBUILD_TESTING=ON",
                f"-DCMAKE_PREFIX_PATH={prefix}",
                *ci_cmake_options(mode),
            ],
            no_tests="ignore",
        )
        section(f"{app.label}: ci mode")
        command, cwd = ci_command(mode, app, build.resolve())
        run(command, cwd=cwd, timeout=ci_timeout(mode))
    if not modules and not apps:
        print("No modules or applications; nothing to build.")
    section(f"OK: {len(modules)} module(s), {len(apps)} application(s) on {platform}")
    return 0


def list_projects(root: Path) -> int:
    for project in discover(root, "modules", "module") + discover(
        root, "applications", "application"
    ):
        print(f"{project.kind:12} {project.name:30} {project.path.relative_to(root)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["host", "sdk", "list"])
    parser.add_argument(
        "--build-root", type=Path, default=ROOT / "build-ci", help="where builds go"
    )
    parser.add_argument(
        "--platform",
        default="x86_64",
        choices=["x86_64", "jetson"],
        help="exported to the tests and ci modes as HOLOSCAN_CI_PLATFORM",
    )
    args = parser.parse_args(argv)
    build_root = args.build_root.resolve()
    exit_on_signals()
    try:
        if args.command == "host":
            return host(ROOT, build_root)
        if args.command == "sdk":
            return sdk(ROOT, build_root, args.platform)
        return list_projects(ROOT)
    except CiError as error:
        print(f"\nci.py: FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
