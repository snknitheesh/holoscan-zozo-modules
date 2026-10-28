# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ci_container.py, with a fake `docker` instead of Docker."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import ci_container

SCRIPT = Path(ci_container.__file__)

# Records each call; `docker run` prints a line and then runs until it is killed,
# like a test that hangs.
FAKE_DOCKER = """#!/bin/sh
echo "$*" >> "$FAKE_DOCKER_CALLS"
if [ "$1" = run ]; then
  echo started
  exec sleep 60
fi
exit 0
"""


class StaleTargetsTest(unittest.TestCase):
    def test_untagged_image_is_removed(self):
        self.assertEqual(ci_container.stale_targets([], "sha256:1"), ["sha256:1"])

    def test_per_run_tags_are_removed(self):
        tags = ["holoscan-modules-ci:run-1-1-x86", "holoscan-modules-ci:run-2-1-x86"]
        self.assertEqual(ci_container.stale_targets(tags, "sha256:1"), tags)

    def test_image_tagged_by_anyone_else_is_kept(self):
        for tags in (["holoscan-modules-ci:local"], ["mine:dev", "holoscan-modules-ci:run-1"]):
            self.assertEqual(ci_container.stale_targets(tags, "sha256:1"), [])


class BuildCacheTest(unittest.TestCase):
    """The shared BuildKit cache is pruned only when the disk is low."""

    def prune_calls(self, free_gb: int) -> list:
        calls = []

        def fake_docker(*args, check=False):
            calls.append(args)
            stdout = "/var/lib/docker" if args[0] == "info" else "Total: 0B"
            return subprocess.CompletedProcess(args, 0, stdout, "")

        usage = mock.Mock(free=free_gb * 10**9)
        with (
            mock.patch.object(ci_container, "docker", fake_docker),
            mock.patch.object(ci_container.shutil, "disk_usage", return_value=usage),
            mock.patch.dict(os.environ, {"CI_BUILD_CACHE_MIN_FREE_GB": "100"}),
        ):
            ci_container.limit_build_cache()
        return [call for call in calls if call[:2] == ("builder", "prune")]

    def test_enough_free_disk_leaves_the_cache_alone(self):
        self.assertEqual(self.prune_calls(free_gb=129), [])

    def test_low_disk_trims_to_the_threshold(self):
        self.assertEqual(
            self.prune_calls(free_gb=40),
            [("builder", "prune", "--force", "--min-free-space", "100GB")],
        )


class RunnerOptionsTest(unittest.TestCase):
    def test_nothing_without_the_variables(self):
        self.assertEqual(ci_container.runner_options({}), [])
        self.assertEqual(
            ci_container.runner_options({"CI_DOCKER_OPTIONS": "", "CI_BUILD_JOBS": " "}), []
        )

    def test_docker_options_are_split_like_a_shell(self):
        env = {"CI_DOCKER_OPTIONS": "--memory 22g --cpuset-cpus 2-13 --label 'a b=1'"}
        self.assertEqual(
            ci_container.runner_options(env),
            ["--memory", "22g", "--cpuset-cpus", "2-13", "--label", "a b=1"],
        )

    def test_build_jobs_are_passed_into_the_container(self):
        self.assertEqual(
            ci_container.runner_options({"CI_BUILD_JOBS": "8"}), ["-e", "CI_BUILD_JOBS"]
        )


class RunTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        fake = root / "bin" / "docker"
        fake.parent.mkdir()
        fake.write_text(FAKE_DOCKER)
        fake.chmod(0o755)
        self.calls = root / "calls"
        self.log = root / "ci.log"
        self.env = {
            **os.environ,
            "PATH": f"{fake.parent}{os.pathsep}{os.environ['PATH']}",
            "FAKE_DOCKER_CALLS": str(self.calls),
            "GITHUB_RUN_ID": "7",
            "GITHUB_RUN_ATTEMPT": "2",
            "CI_JOB_KEY": "x86",
        }

    def tearDown(self):
        self.directory.cleanup()

    def start(self, *options):
        process = subprocess.Popen(
            [
                sys.executable,
                str(SCRIPT),
                "run",
                "--image",
                "image@sha256:1",
                "--name",
                "ci-7",
                "--log",
                str(self.log),
                "--docker-options",
                "--gpus all",
                *options,
                "--",
                "python3",
                "ci.py",
            ],
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.addCleanup(process.stdout.close)
        self.addCleanup(process.kill)
        self.assertEqual(process.stdout.readline().split()[:3], [b"+", b"docker", b"run"])
        self.assertEqual(process.stdout.readline().strip(), b"started")
        return process

    def calls_made(self):
        return self.calls.read_text().splitlines()

    def test_labels_init_and_rm_on_every_container(self):
        process = self.start()
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=20)
        command = self.calls_made()[0].split()
        for option in ("--rm", "--init", "--gpus"):
            self.assertIn(option, command)
        self.assertIn(f"{ci_container.LABEL}=1", command)
        self.assertIn(f"{ci_container.RUN_LABEL}=7-2-x86", command)
        self.assertEqual(command[-3:], ["image@sha256:1", "python3", "ci.py"])
        self.assertIn("started", self.log.read_text())

    def test_runs_as_the_runner_user_with_a_writable_workspace(self):
        process = self.start()
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=20)
        command = self.calls_made()[0].split()
        self.assertEqual(command[command.index("--user") + 1], f"{os.getuid()}:{os.getgid()}")
        self.assertFalse(command[command.index("-v") + 1].endswith(":ro"))

    def test_root_gets_a_read_only_workspace(self):
        process = self.start("--root")
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=20)
        command = self.calls_made()[0].split()
        self.assertEqual(command[command.index("--user") + 1], "0:0")
        self.assertTrue(command[command.index("-v") + 1].endswith(":ro"))

    def test_runner_options_reach_docker_run(self):
        self.env.update(CI_DOCKER_OPTIONS="--memory 22g --cpuset-cpus 2-13", CI_BUILD_JOBS="8")
        process = self.start()
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=20)
        command = self.calls_made()[0].split()
        # After the workflow's options, so that the runner's limits apply to every job.
        gpus = command.index("--gpus")
        self.assertEqual(
            command[gpus + 2 : gpus + 8],
            ["--memory", "22g", "--cpuset-cpus", "2-13", "-e", "CI_BUILD_JOBS"],
        )

    def test_signals_remove_the_container(self):
        for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            with self.subTest(signal=signal.Signals(number).name):
                self.calls.unlink(missing_ok=True)
                process = self.start()
                started = time.monotonic()
                process.send_signal(number)
                self.assertEqual(process.wait(timeout=20), 128 + number)
                self.assertLess(time.monotonic() - started, 15)
                self.assertIn("rm --force ci-7", self.calls_made())

    def test_killed_docker_cli_still_removes_the_container(self):
        process = self.start()
        cli = subprocess.run(
            ["pgrep", "-P", str(process.pid)], capture_output=True, text=True, check=True
        ).stdout.split()
        os.kill(int(cli[0]), signal.SIGKILL)
        self.assertEqual(process.wait(timeout=20), 128 + signal.SIGKILL)
        self.assertIn("rm --force ci-7", self.calls_made())


if __name__ == "__main__":
    unittest.main()
