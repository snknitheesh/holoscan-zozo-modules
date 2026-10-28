# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0
"""Build and test the modules under modules/ and the demos under applications/.

Each folder plugs in by layout alone (see modules/README.md and
applications/README.md), so adding one needs no workflow change:

  select  Decide what a run tests, and print why. With --changed-since <ref>, only
          the modules and demos whose folders changed since the merge base of <ref>
          and HEAD, the modules and demos that depend on them, and nothing for a
          change to documentation alone; a change anywhere else selects everything.
          Without it, everything. --output appends `selection=` and `gpu=` lines for
          GITHUB_OUTPUT.
  host    Build and run the SDK-free test projects (<folder>/tests/host/).
  sdk     For each module, in dependency order: build it, run its full CTest suite,
          install it, then build and test its installed-package consumer
          (tests/installed_consumer/). Each module builds against those installed
          before it. Then, for each demo: build it against the installed modules, run
          its tests, and run its finite `ci` mode from metadata.json, which is killed
          after modes.ci.run.timeout seconds (default 900).
  list    Print the modules in build order and the demos, with their dependencies.

host and sdk take --selection, the JSON that select printed, and then test only what
it names; sdk still builds and installs, without testing, the modules those depend on.
Every step must succeed; the first failure ends the run with a nonzero status.
"""

from __future__ import annotations

import argparse
import contextlib
import heapq
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
APP_BIN = "<holohub_app_bin>"
# Seconds a demo's ci mode may run, unless its modes.ci.run.timeout says otherwise.
CI_MODE_TIMEOUT = 900
# A CMake file's use of a module of this collection: find_package(holoscan-<name> ...).
FIND_MODULE = re.compile(
    r"\b(?:find_package|find_dependency)\s*\(\s*holoscan-([A-Za-z0-9_-]+)", re.IGNORECASE
)


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

    @property
    def metadata_file(self) -> Path:
        return self.path / "metadata.json"


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


# --- The dependency graph -----------------------------------------------------------


@dataclass(frozen=True)
class Graph:
    """Every module and demo, and the modules each one depends on, by folder name."""

    modules: dict[str, Project]  # In build order: each after the modules it depends on.
    applications: dict[str, Project]  # In name order.
    requires: dict[tuple[str, str], tuple[str, ...]]  # (kind, name) -> module folders

    def module_deps(self, name: str) -> tuple[str, ...]:
        return self.requires[("module", name)]

    def app_deps(self, name: str) -> tuple[str, ...]:
        return self.requires[("application", name)]

    def dependencies(self, modules: set[str]) -> set[str]:
        """Every module that `modules` depend on, directly or not, without themselves."""
        found: set[str] = set()
        pending = [dep for name in modules for dep in self.module_deps(name)]
        while pending:
            name = pending.pop()
            if name not in found:
                found.add(name)
                pending.extend(self.module_deps(name))
        return found - modules

    def dependents(self, modules: set[str]) -> set[str]:
        """Every module that depends on `modules`, directly or not, without themselves."""
        found = set(modules)
        for name in self.modules:  # Build order: a module comes after its dependencies.
            if found.intersection(self.module_deps(name)):
                found.add(name)
        return found - modules


def dependency_names(project: Project) -> list[str]:
    """The in-tree modules a project declares in its metadata.json, as written.

    A module lists them in module.dependencies, a demo in
    application.dependencies.modules, as the Holoscan CLI reads them. An entry with a
    `source` block is a module from another repository and is not part of the graph;
    any other entry is a module of this collection, named by its folder under
    modules/, which is also where the CLI looks for it.
    """
    envelope = project.metadata[project.kind]
    where = "module.dependencies"
    entries = envelope.get("dependencies", [])
    if project.kind == "application":
        where = "application.dependencies.modules"
        dependencies = envelope.get("dependencies", {})
        if not isinstance(dependencies, dict):
            raise CiError(f"{project.metadata_file}: application.dependencies must be an object")
        entries = dependencies.get("modules", [])
    if not isinstance(entries, list):
        raise CiError(f"{project.metadata_file}: {where} must be a list")
    names = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise CiError(f"{project.metadata_file}: every entry of {where} needs a name")
        if "source" not in entry:
            names.append(entry["name"])
    return names


def cmake_uses(project: Project) -> set[str]:
    """The holoscan-<name> packages the project's CMake files find, other than itself."""
    found = set()
    for path in sorted(project.path.rglob("*")):
        if path.name == "CMakeLists.txt" or path.name.endswith((".cmake", ".cmake.in")):
            text = path.read_text(encoding="utf-8", errors="replace")
            code = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
            found.update(match.lower() for match in FIND_MODULE.findall(code))
    found.discard(project.name)
    return found


def toposort(requires: dict[str, tuple[str, ...]]) -> list[str]:
    """The names in an order where each comes after those it requires; ties by name.
    A cycle is a CiError that names it."""
    waiting = {name: set(deps) for name, deps in requires.items()}
    ready = [name for name, deps in waiting.items() if not deps]
    heapq.heapify(ready)
    order = []
    while ready:
        name = heapq.heappop(ready)
        order.append(name)
        for other, deps in waiting.items():
            if name in deps:
                deps.discard(name)
                if not deps:
                    heapq.heappush(ready, other)
    if len(order) < len(requires):
        raise CiError(f"modules depend on each other in a cycle: {find_cycle(requires, order)}")
    return order


def find_cycle(requires: dict[str, tuple[str, ...]], done: list[str]) -> str:
    """One cycle among the modules toposort() could not order, as "a -> b -> a"."""
    left = {name: [dep for dep in requires[name] if dep not in done] for name in requires}
    name = min(n for n in left if n not in done)
    path: list[str] = []
    while name not in path:
        path.append(name)
        name = min(left[name])  # Every module left over depends on one left over.
    return " -> ".join([*path[path.index(name) :], name])


def load_graph(root: Path) -> Graph:
    """Discover every module and demo, check their declared dependencies against each
    other and against their CMake files, and order the modules by dependency."""
    modules = {p.name: p for p in discover(root, "modules", "module")}
    apps = {p.name: p for p in discover(root, "applications", "application")}
    requires: dict[tuple[str, str], tuple[str, ...]] = {}
    for project in [*modules.values(), *apps.values()]:
        where = "module.dependencies"
        if project.kind == "application":
            where = "application.dependencies.modules"
        names = dependency_names(project)
        for name in names:
            if name in modules:
                continue
            hint = ""
            if name.startswith("holoscan-") and name.removeprefix("holoscan-") in modules:
                hint = (
                    f": name the folder, {name.removeprefix('holoscan-')!r}, which is"
                    " where the Holoscan CLI looks for an in-tree module"
                )
            raise CiError(
                f"{project.metadata_file}: {where} names {name!r}, which is not a module"
                f" in modules/ and has no source{hint}"
            )
        if project.kind == "module" and project.name in names:
            raise CiError(f"{project.metadata_file}: the module depends on itself")
        undeclared = sorted((cmake_uses(project) & modules.keys()) - set(names))
        if undeclared:
            entries = ", ".join(f'{{"name": "{name}"}}' for name in undeclared)
            raise CiError(
                f"{project.path}: its CMake files find holoscan-{', holoscan-'.join(undeclared)},"
                f" which {project.metadata_file.name} does not declare; add {entries}"
                f" to {where}"
            )
        requires[(project.kind, project.name)] = tuple(sorted(set(names)))
    order = toposort({name: requires[("module", name)] for name in modules})
    return Graph({name: modules[name] for name in order}, apps, requires)


# --- Selecting what to test ---------------------------------------------------------


@dataclass
class Selection:
    """What a run tests: these modules' suites and these demos. `everything` is a run
    of every module and demo, whatever the lists say; `nightly` asks also for the
    tests that only the nightly run makes (HOLOSCAN_CI_NIGHTLY)."""

    everything: bool
    modules: list[str]
    applications: list[str]
    nightly: bool = False

    @property
    def empty(self) -> bool:
        return not self.everything and not self.modules and not self.applications

    def to_json(self) -> str:
        return json.dumps(
            {
                "everything": self.everything,
                "nightly": self.nightly,
                "modules": self.modules,
                "applications": self.applications,
            },
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, text: str, graph: Graph) -> Selection:
        try:
            data = json.loads(text)
            selection = cls(
                bool(data["everything"]),
                list(data["modules"]),
                list(data["applications"]),
                bool(data.get("nightly", False)),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise CiError(f"--selection is not what `ci.py select` prints: {error}") from error
        if selection.everything:
            return everything(graph, selection.nightly)
        for kind, names, known in (
            ("module", selection.modules, graph.modules),
            ("application", selection.applications, graph.applications),
        ):
            unknown = [name for name in names if name not in known]
            if unknown:
                raise CiError(f"--selection names no such {kind}: {', '.join(unknown)}")
        return selection


def everything(graph: Graph, nightly: bool = False) -> Selection:
    return Selection(True, list(graph.modules), list(graph.applications), nightly)


def is_documentation(root: Path, path: str) -> bool:
    """A Markdown file that still exists, outside .github/: nothing builds from one.
    A deleted one is not, since a module may install its README."""
    return path.endswith(".md") and not path.startswith(".github/") and (root / path).is_file()


def select(graph: Graph, root: Path, paths: list[str]) -> tuple[Selection, list[str]]:
    """What a change to `paths` (relative to the root) tests, and why, line by line.

    A path in modules/<m>/ selects module m, every module that depends on it, and every
    demo that uses any of those; a path in applications/<a>/ selects demo a. A
    documentation file selects nothing. Any other path, such as the workflows, the
    root CMakeLists.txt, cmake/, the CI image or a folder that is no longer a module or
    demo, selects everything: what it affects is not known.
    """
    changed: dict[tuple[str, str], str] = {}
    docs = []
    for path in paths:
        parts = PurePosixPath(path).parts
        if is_documentation(root, path):
            docs.append(path)
            continue
        if len(parts) >= 3 and parts[0] == "modules" and parts[1] in graph.modules:
            changed.setdefault(("module", parts[1]), path)
        elif len(parts) >= 3 and parts[0] == "applications" and parts[1] in graph.applications:
            changed.setdefault(("application", parts[1]), path)
        else:
            return everything(graph), [f"everything: {path} is not in a module's or demo's folder"]
    why = [f"documentation only, nothing to build: {path}" for path in docs]
    direct = {name for kind, name in changed if kind == "module"}
    modules = direct | graph.dependents(direct)
    for name in graph.modules:
        if name in direct:
            why.append(f"module {name}: changed ({changed[('module', name)]})")
        elif name in modules:
            on = sorted(direct & graph.dependencies({name}))
            why.append(f"module {name}: depends on {', '.join(on)}")
    apps = []
    for name in graph.applications:
        uses = sorted(set(graph.app_deps(name)) & modules)
        if ("application", name) in changed:
            why.append(f"application {name}: changed ({changed[('application', name)]})")
        elif uses:
            why.append(f"application {name}: uses {', '.join(uses)}")
        else:
            continue
        apps.append(name)
    order = [name for name in graph.modules if name in modules]
    return Selection(False, order, apps), why


def to_build(graph: Graph, selection: Selection) -> list[str]:
    """The modules a run builds and installs, in build order: those it tests, and those
    that they, or the demos it runs, depend on."""
    tested = set(selection.modules)
    needed = tested | {dep for app in selection.applications for dep in graph.app_deps(app)}
    needed |= graph.dependencies(needed)
    return [name for name in graph.modules if name in needed]


def changed_files(root: Path, base: str) -> list[str]:
    """The files that differ between HEAD and its merge base with `base`, as a pull
    request shows them. Both sides of a rename count."""
    if not base or base.startswith("-"):
        raise CiError(f"not a commit: {base!r}")
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "diff",
                "--name-only",
                "--no-renames",
                "-z",
                f"{base}...HEAD",
                "--",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "") or error
        raise CiError(f"cannot diff {base}...HEAD: {str(detail).strip()}") from error
    return [path for path in result.stdout.split("\0") if path]


def select_command(root: Path, base: str | None, nightly: bool, output: Path | None) -> int:
    graph = load_graph(root)
    if nightly:
        selection, why = everything(graph, True), ["everything: the nightly run"]
    elif not base:
        selection, why = everything(graph), ["everything: no base to compare with"]
    else:
        try:
            paths = changed_files(root, base)
        except CiError as error:
            # Fail safe: what cannot be compared is tested in full.
            print(f"::warning::{error}; testing everything")
            selection, why = everything(graph), [f"everything: {error}"]
        else:
            print(f"{len(paths)} file(s) changed since the merge base with {base}.")
            selection, why = select(graph, root, paths)
    section("Selection")
    for line in why or ["nothing: no file changed"]:
        print(f"  {line}")
    built = [name for name in to_build(graph, selection) if name not in selection.modules]
    for name in built:
        print(f"  module {name}: built and installed for what depends on it, not tested")
    if selection.empty:
        summary = "Nothing to build or test on the GPUs"
    else:
        summary = (
            f"Testing {'everything: ' if selection.everything else ''}"
            f"module(s) {', '.join(selection.modules) or 'none'};"
            f" demo(s) {', '.join(selection.applications) or 'none'}"
        )
        if built:
            summary += f"; building without tests {', '.join(built)}"
    print(f"::notice title=CI selection::{summary}")
    if output is not None:
        with output.open("a", encoding="utf-8") as stream:
            stream.write(f"selection={selection.to_json()}\n")
            stream.write(f"gpu={'false' if selection.empty else 'true'}\n")
    return 0


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


def selected(graph: Graph, selection_json: str | None) -> Selection:
    """The run's selection: --selection, or everything without it."""
    if selection_json is None:
        return everything(graph)
    return Selection.from_json(selection_json, graph)


def host(root: Path, build_root: Path, selection_json: str | None = None) -> int:
    """Build and run the SDK-free tests/host/ project of each selected folder."""
    graph = load_graph(root)
    selection = selected(graph, selection_json)
    projects = [graph.modules[name] for name in selection.modules] + [
        graph.applications[name] for name in selection.applications
    ]
    suites = [p for p in projects if (p.path / "tests" / "host" / "CMakeLists.txt").is_file()]
    print(f"Selected {len(projects)} project(s), {len(suites)} with SDK-free tests.")
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


def sdk(root: Path, build_root: Path, platform: str, selection_json: str | None = None) -> int:
    """Build, test, install and consume the selected modules, building and installing
    what they and the selected demos depend on first; then build and run the demos."""
    graph = load_graph(root)
    # Fail on any demo's ci mode before building anything.
    modes = {name: ci_mode(app) for name, app in graph.applications.items()}
    selection = selected(graph, selection_json)
    tested = set(selection.modules)
    modules = [graph.modules[name] for name in to_build(graph, selection)]
    apps = [graph.applications[name] for name in selection.applications]
    print(
        f"Platform {platform}: {len(tested)} module(s) to test, of {len(graph.modules)};"
        f" {len(modules) - len(tested)} more to build for them;"
        f" {len(apps)} application(s), of {len(graph.applications)}."
    )
    prefix = build_root / "install"
    # Module tests and ci modes may check what holds only on one platform, and leave
    # long runs or downloaded models to the nightly run.
    os.environ["HOLOSCAN_CI_PLATFORM"] = platform
    os.environ["HOLOSCAN_CI_NIGHTLY"] = "1" if selection.nightly else "0"
    release = ["-DCMAKE_BUILD_TYPE=Release"]
    for module in modules:
        build = build_root / "modules" / module.name
        options = [
            *release,
            f"-DCMAKE_INSTALL_PREFIX={prefix}",
            # The modules installed before this one: those it depends on.
            f"-DCMAKE_PREFIX_PATH={prefix}",
        ]
        if module.name not in tested:
            # Only a dependency of what is tested: its own tests are not this run's.
            section(f"{module.label}: build, for what depends on it")
            run(
                [
                    "cmake",
                    "-S",
                    str(module.path),
                    "-B",
                    str(build),
                    *generator(),
                    *options,
                    "-DBUILD_TESTING=OFF",
                ]
            )
            run(["cmake", "--build", str(build), "--parallel", str(build_jobs())])
            section(f"{module.label}: install")
            run(["cmake", "--install", str(build)])
            continue
        section(f"{module.label}: build and full test suite")
        cmake_build_and_test(module.path, build, [*options, "-DBUILD_TESTING=ON"], "error")
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
        print("No modules or applications selected; nothing to build.")
    section(
        f"OK: {len(tested)} module(s) tested, {len(modules) - len(tested)} more built,"
        f" {len(apps)} application(s) on {platform}"
    )
    return 0


def list_projects(root: Path) -> int:
    """The modules in build order, then the demos, each with what it depends on."""
    graph = load_graph(root)
    for project in [*graph.modules.values(), *graph.applications.values()]:
        deps = ", ".join(graph.requires[(project.kind, project.name)]) or "-"
        print(
            f"{project.kind:12} {project.name:30} {project.path.relative_to(root)!s:40}"
            f" depends on: {deps}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["select", "host", "sdk", "list"])
    parser.add_argument(
        "--build-root", type=Path, default=ROOT / "build-ci", help="where builds go"
    )
    parser.add_argument(
        "--platform",
        default="x86_64",
        choices=["x86_64", "jetson"],
        help="exported to the tests and ci modes as HOLOSCAN_CI_PLATFORM",
    )
    parser.add_argument(
        "--changed-since",
        metavar="REF",
        help="select: test only what changed since the merge base of REF and HEAD",
    )
    parser.add_argument(
        "--nightly",
        action="store_true",
        help="select: test everything, with HOLOSCAN_CI_NIGHTLY=1 for the tests",
    )
    parser.add_argument(
        "--output", type=Path, help="select: append selection= and gpu= lines to this file"
    )
    parser.add_argument(
        "--selection",
        metavar="JSON",
        help="host, sdk: test only what this selection, printed by select, names",
    )
    args = parser.parse_args(argv)
    build_root = args.build_root.resolve()
    exit_on_signals()
    try:
        if args.command == "select":
            return select_command(ROOT, args.changed_since, args.nightly, args.output)
        if args.command == "host":
            return host(ROOT, build_root, args.selection)
        if args.command == "sdk":
            return sdk(ROOT, build_root, args.platform, args.selection)
        return list_projects(ROOT)
    except CiError as error:
        print(f"\nci.py: FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
