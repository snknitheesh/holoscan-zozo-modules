# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0
"""Run the CI container on a self-hosted GPU runner, and leave nothing behind.

  run    Run a command in the CI image, in a container labeled as a CI container
         and removed however the run ends: `--rm` and `--init`, plus `docker rm -f`
         on exit and on SIGINT, SIGTERM and SIGHUP (which is how the runner cancels a
         step). Output also goes to a log file. Start it with `exec`, so that the
         runner's signals reach it rather than the step's shell.
  sweep  Remove every container with the CI label (a runner runs one job at a time,
         so any of them is left over), every stopped container made from this
         repository's image or its build steps (a failed legacy build leaves those),
         this repository's untagged and per-run images, and trim the daemon's BuildKit
         cache when the disk runs low.

Only containers with the CI label, stopped containers and images with this
repository's org.opencontainers.image.source label are ever removed: a running
container from the image without the CI label is someone's, and is left alone. The
runner job hook, .github/runner/job-hook.sh, removes the same containers, for when a
job dies before its own cleanup runs.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABEL = "io.github.snknitheesh.holoscan-zozo-modules.ci"
RUN_LABEL = f"{LABEL}.run"
SOURCE_LABEL = (
    "org.opencontainers.image.source=https://github.com/snknitheesh/holoscan-zozo-modules"
)
RUN_TAG_PREFIX = "holoscan-modules-ci:run-"


def log(message: str) -> None:
    print(message, flush=True)


def docker(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    result = subprocess.run(["docker", *args], check=False, text=True, capture_output=True)
    if check and result.returncode:
        raise SystemExit(
            f"::error::docker {shlex.join(args)} failed ({result.returncode}): "
            f"{result.stderr.strip()}"
        )
    return result


def run_id() -> str:
    """This job's identity: run, attempt and job, from the Actions environment."""
    parts = [os.environ.get("GITHUB_RUN_ID", "local"), os.environ.get("GITHUB_RUN_ATTEMPT", "1")]
    job = os.environ.get("CI_JOB_KEY", "")
    return "-".join([*parts, job] if job else parts)


def stale_targets(tags: list[str], image: str) -> list[str]:
    """What to remove for one of this repository's images: its per-run tags, or the
    image itself when it has no tag. Nothing when anything else has tagged it."""
    run_tags = [tag for tag in tags if tag.startswith(RUN_TAG_PREFIX)]
    if len(run_tags) != len(tags):
        return []
    return run_tags or [image]


def remove_containers() -> None:
    def listed(*filters: str) -> list[str]:
        args = [arg for value in filters for arg in ("--filter", value)]
        return docker("ps", "--all", "--quiet", *args, check=True).stdout.split()

    ids = listed(f"label={LABEL}")
    # Several status filters match any of them.
    stopped = listed(f"label={SOURCE_LABEL}", "status=created", "status=exited", "status=dead")
    ids = list(dict.fromkeys(ids + stopped))
    if not ids:
        log("No CI or build containers left over.")
        return
    log(f"Removing left-over CI or build container(s): {' '.join(ids)}")
    docker("rm", "--force", *ids, check=True)


def remove_images() -> None:
    listed = docker(
        "image", "ls", "--quiet", "--no-trunc", "--filter", f"label={SOURCE_LABEL}", check=True
    ).stdout.split()
    for image in dict.fromkeys(listed):
        inspected = docker("image", "inspect", "--format", "{{json .RepoTags}}", image)
        if inspected.returncode:
            continue  # Already gone with an earlier tag.
        for target in stale_targets(json.loads(inspected.stdout or "null") or [], image):
            removed = docker("rmi", target)
            if removed.returncode == 0:
                log(f"Removed stale image {target}")
            else:
                log(f"Kept {target}: {removed.stderr.strip()}")


def limit_build_cache() -> None:
    """Trim the daemon's BuildKit cache, least recently used first, but only when the
    disk holding Docker's data has less than CI_BUILD_CACHE_MIN_FREE_GB (default 100)
    gigabytes free. The cache is the daemon's, shared with everything else built on
    the machine, so it is never cut by a fixed size; and `docker builder prune` drops
    the records a later build of this image reuses even with plenty of disk free, so
    it is not run at all then."""
    min_free = int(os.environ.get("CI_BUILD_CACHE_MIN_FREE_GB", "100"))
    root = docker("info", "--format", "{{.DockerRootDir}}", check=True).stdout.strip()
    try:
        free = shutil.disk_usage(root).free // 10**9
    except OSError as error:
        print(f"::warning::cannot read the free space of {root}: {error}", flush=True)
        return
    if free >= min_free:
        log(f"Build cache left alone: {free} GB free on {root} (trimmed below {min_free} GB).")
        return
    if docker("buildx", "version").returncode:
        print("::warning::Disk is low, but there is no Docker Buildx to trim with.", flush=True)
        return
    pruned = docker("builder", "prune", "--force", "--min-free-space", f"{min_free}GB")
    if pruned.returncode:
        print(f"::warning::docker builder prune failed: {pruned.stderr.strip()}", flush=True)
    else:
        summary = pruned.stdout.strip().splitlines()[-1:] or ["nothing"]
        log(f"Disk low ({free} GB free on {root}): trimmed the build cache, {summary[0]}")


def sweep(_args: argparse.Namespace) -> int:
    remove_containers()
    remove_images()
    limit_build_cache()
    return 0


def run(args: argparse.Namespace) -> int:
    name = args.name
    workspace = os.environ.get("GITHUB_WORKSPACE", str(ROOT))
    command = [
        "docker",
        "run",
        "--rm",
        "--init",
        "--pull",
        "never",
        "--name",
        name,
        "--label",
        f"{LABEL}=1",
        "--label",
        f"{RUN_LABEL}={run_id()}",
        *shlex.split(args.docker_options),
        "-e",
        "NVIDIA_DRIVER_CAPABILITIES=all",
        "-e",
        "HOME=/tmp",
        "-e",
        "PYTHONUNBUFFERED=1",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-v",
        f"{workspace}:{workspace}",
        "-w",
        workspace,
        args.image,
        *args.command,
    ]
    received: list[int] = []
    children: list[subprocess.Popen] = []

    def remove() -> None:
        subprocess.run(["docker", "rm", "--force", name], check=False, capture_output=True)

    def on_signal(number: int, _frame) -> None:
        received.append(number)
        log(f"\nci_container.py: {signal.Signals(number).name}, removing container {name}")
        remove()
        for child in children:  # The docker CLI, whose container is gone now.
            child.kill()

    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, on_signal)
    log(f"+ {shlex.join(command)}")
    try:
        with open(args.log, "wb") as log_file:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            children.append(process)
            assert process.stdout is not None
            for line in iter(process.stdout.readline, b""):
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
                log_file.write(line)
            status = process.wait()
    finally:
        remove()
    if received:
        return 128 + received[0]
    # A docker CLI killed by a signal: report it the way a shell does (128 + signal).
    return 128 - status if status < 0 else status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("sweep", help="remove left-over CI containers and images")
    execute = actions.add_parser("run", help="run a command in the CI container")
    execute.add_argument("--image", required=True)
    execute.add_argument("--name", required=True)
    execute.add_argument("--log", required=True, type=Path)
    execute.add_argument("--docker-options", default="", help="extra `docker run` options")
    execute.add_argument("command", nargs=argparse.REMAINDER, help="-- then the command")
    args = parser.parse_args(argv)
    if args.action == "sweep":
        return sweep(args)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("run: no command given")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
