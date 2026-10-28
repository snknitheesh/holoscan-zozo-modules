# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0
"""Build and check one module's Debian package for a release; see .github/RELEASING.md.

  tag      Check a release tag, <module>-v<major>.<minor>.<patch>, against the module's
           metadata.json, and print module=, version=, package= lines for GITHUB_OUTPUT.
  package  Build the module with its Debian package configuration (PKG_<package>=ON, for
           every major GPU architecture), run CPack, and check the .deb's name, package,
           version and architecture. Runs in the CI image as the runner's user.
  consume  Install a .deb with apt, then build and run the module's
           tests/installed_consumer/ against the installed package, with nothing else on
           CMAKE_PREFIX_PATH. Runs as root in a fresh CI container, and writes outside the
           repository only.

Every step must succeed; the first failure ends the run with a nonzero status.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from ci import ROOT, CiError, Project, discover, exit_on_signals, generator, run, section

TAG = re.compile(
    r"(?P<module>[a-z0-9][a-z0-9-]*)-v(?P<version>(?:0|[1-9]\d*)(?:\.(?:0|[1-9]\d*)){2})"
)


def find_module(root: Path, name: str) -> Project:
    for module in discover(root, "modules", "module"):
        if module.name == name:
            return module
    raise CiError(f"no module modules/{name}/")


def package_name(module: Project) -> str:
    """The Debian package: binary_packages.debian, else the module's name."""
    metadata = module.metadata["module"]
    return (metadata.get("binary_packages") or {}).get("debian") or metadata["name"]


def check_module(module: Project, version: str) -> None:
    metadata = module.metadata["module"]
    if metadata.get("name") != f"holoscan-{module.name}":
        raise CiError(
            f"{module.path / 'metadata.json'}: module.name must be holoscan-{module.name}"
        )
    if metadata.get("version") != version:
        raise CiError(
            f"{module.path / 'metadata.json'}: module.version is {metadata.get('version')!r}, "
            f"but the tag is for {version}; bump it (and the CMake project VERSION) first"
        )


def parse_tag(root: Path, tag: str) -> tuple[Project, str]:
    match = TAG.fullmatch(tag)
    if not match:
        raise CiError(f"tag {tag!r} is not <module>-v<major>.<minor>.<patch>, e.g. apriltag-v1.0.0")
    module = find_module(root, match["module"])
    check_module(module, match["version"])
    return module, match["version"]


def deb_architecture() -> str:
    return subprocess.run(
        ["dpkg", "--print-architecture"], check=True, capture_output=True, text=True
    ).stdout.strip()


def deb_fields(deb: Path) -> dict[str, str]:
    output = subprocess.run(
        ["dpkg-deb", "--field", str(deb), "Package", "Version", "Architecture"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return dict(line.split(": ", 1) for line in output.splitlines() if ": " in line)


def package(root: Path, name: str, version: str, out: Path, build_root: Path) -> Path:
    module = find_module(root, name)
    check_module(module, version)
    deb_name = package_name(module)
    option = "PKG_" + re.sub(r"[^A-Za-z0-9_]", "_", deb_name)
    build = build_root / "package" / module.name
    staging = build_root / "cpack" / module.name
    for folder in (build, staging):
        shutil.rmtree(folder, ignore_errors=True)
    section(f"module {module.name}: build with its Debian package configuration")
    run(
        [
            "cmake",
            "-S",
            str(module.path),
            "-B",
            str(build),
            *generator(),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_TESTING=OFF",
            f"-D{option}=ON",
        ]
    )
    run(["cmake", "--build", str(build), "--parallel", str(os.cpu_count() or 2)])
    config = build / "pkg" / f"CPackConfig-{deb_name}.cmake"
    if not config.is_file():
        raise CiError(f"{option}=ON wrote no {config}; see modules/README.md")
    section(f"module {module.name}: package")
    run(["cpack", "--config", str(config), "-G", "DEB", "-B", str(staging)])
    debs = sorted(staging.glob("*.deb"))
    expected = f"{deb_name}_{version}_{deb_architecture()}.deb"
    if [deb.name for deb in debs] != [expected]:
        raise CiError(f"expected CPack to write {expected} alone, got {[d.name for d in debs]}")
    fields = deb_fields(debs[0])
    wanted = {"Package": deb_name, "Version": version, "Architecture": deb_architecture()}
    if fields != wanted:
        raise CiError(f"{expected}: control fields {fields}, expected {wanted}")
    out.mkdir(parents=True, exist_ok=True)
    target = out / expected
    shutil.copy2(debs[0], target)
    print(f"Wrote {target}", flush=True)
    return target


def consume(root: Path, name: str, deb: Path, build_root: Path) -> int:
    module = find_module(root, name)
    consumer = module.path / "tests" / "installed_consumer"
    if not (consumer / "CMakeLists.txt").is_file():
        raise CiError(f"{module.path}: needs tests/installed_consumer/CMakeLists.txt")
    os.environ["DEBIAN_FRONTEND"] = "noninteractive"
    section(f"install {deb.name}")
    run(["apt-get", "update"])
    # A path, so that apt installs the file rather than a package of that name.
    run(["apt-get", "install", "--yes", "--no-install-recommends", str(deb.resolve())])
    run(["dpkg", "--status", package_name(module)])
    section(f"module {module.name}: consumer of the installed package")
    build = build_root / "consumer" / module.name
    shutil.rmtree(build, ignore_errors=True)
    run(
        ["cmake", "-S", str(consumer), "-B", str(build), *generator(), "-DCMAKE_BUILD_TYPE=Release"]
    )
    run(["cmake", "--build", str(build), "--parallel", str(os.cpu_count() or 2)])
    run(["ctest", "--test-dir", str(build), "--output-on-failure", "--no-tests=error"])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    tag = commands.add_parser("tag", help="check a release tag; print its module and version")
    tag.add_argument("tag")
    build = commands.add_parser("package", help="build and check the module's .deb")
    build.add_argument("module")
    build.add_argument("--version", required=True)
    build.add_argument("--out", type=Path, required=True, help="where the .deb goes")
    build.add_argument("--build-root", type=Path, default=ROOT / "build-release")
    use = commands.add_parser("consume", help="install a .deb and build the module's consumer")
    use.add_argument("module")
    use.add_argument("--deb", type=Path, required=True)
    use.add_argument("--build-root", type=Path, default=Path("/tmp/release-consumer"))
    args = parser.parse_args(argv)
    exit_on_signals()
    try:
        if args.command == "tag":
            module, version = parse_tag(ROOT, args.tag)
            print(f"module={module.name}\nversion={version}\npackage={package_name(module)}")
            return 0
        if args.command == "package":
            package(ROOT, args.module, args.version, args.out.resolve(), args.build_root.resolve())
            return 0
        return consume(ROOT, args.module, args.deb, args.build_root.resolve())
    except CiError as error:
        print(f"\nrelease.py: FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
