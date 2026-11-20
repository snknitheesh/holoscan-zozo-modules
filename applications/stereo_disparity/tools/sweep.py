# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0
"""Measure every matcher setting the module's README compares, with the demo.

Runs the built demo once per backend, setting and input, one backend at a time, and
prints a CSV row for each: the operator's mean and longest time per frame, and its
accuracy against the synthetic scene's truth or a recording's camera depth. A setting
the platform refuses (such as one or two passes on Jetson Thor) gets a row saying so.

  python3 tools/sweep.py build/stereo_disparity/stereo_disparity --backends cuda,ofa,pva

The inputs are the synthetic scene at 1280x720 (MONO8 in GPU memory) and, when
HOLOSCAN_TEST_DATA is set and --camera-depth-scale given, the recordings in its
stereo-camera/ folder whose names end in --clips' suffixes (the local recordings
with the projector on, by default).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import pathlib
import re
import subprocess
import sys

FIELDS = (
    "backend",
    "confidence",
    "max_disparity",
    "diagonals",
    "passes",
    "downscale",
    "input",
    "compute_ms",
    "compute_max_ms",
    "valid",
    "truth_within_1px",
    "truth_median_px",
    "truth_mae_px",
    "coverage",
    "median_rel_err",
    "p90_rel_err",
    "within_5pct",
    "note",
)


def settings(backend: str) -> list[tuple[int, bool, int, int]]:
    """(max_disparity, diagonals, passes, downscale) for every setting `backend` reads."""
    if backend == "cuda":
        return [(d, g, 3, 1) for d, g in itertools.product((64, 128, 256), (True, False))]
    diagonals = (True, False) if backend == "ofa" else (True,)  # pva reads no diagonals
    return list(itertools.product((128, 256), diagonals, (1, 2, 3), (1, 2, 4)))


def measure(demo: str, backend: str, setting: tuple, source: str, args) -> dict:
    """One run's row."""
    max_disparity, diagonals, passes, downscale = setting
    options = {
        "--backend": backend,
        "--max-disparity": max_disparity,
        "--diagonals": "true" if diagonals else "false",
        "--num-passes": passes,
        "--downscale": downscale,
        "--display": "false",
        "--period-ms": args.period_ms,
        "--count": args.count,
        "--confidence": "true" if args.confidence else "false",
    }
    if source == "synthetic":
        options["--color"] = "false"
    else:
        options.update(
            {
                "--source": "recording",
                "--clip": source,
                "--camera-depth-scale": args.camera_depth_scale,
            }
        )
    command = [demo, *(str(part) for pair in options.items() for part in pair)]
    row = dict.fromkeys(FIELDS, "")
    row.update(
        backend=backend,
        confidence=str(args.confidence).lower(),
        max_disparity=max_disparity,
        diagonals=str(diagonals).lower(),
        passes=passes,
        downscale=downscale,
        input=source,
    )
    done = subprocess.run(
        command, capture_output=True, text=True, timeout=args.timeout, check=False
    )
    output = done.stdout + done.stderr
    line = next((text for text in output.splitlines() if " backend=" in text), "")
    timing = re.search(r"compute_ms\(mean/max\)=([0-9.]+)/([0-9.]+)", line)
    if done.returncode != 0 or timing is None:
        error = next((text for text in output.splitlines() if text.startswith("error")), "")
        row["note"] = error or f"exit status {done.returncode}"
        return row

    def value(pattern: str) -> str:
        found = re.search(pattern + r"=([0-9.]+)", line)
        return found.group(1) if found else ""

    row.update(
        compute_ms=timing.group(1),
        compute_max_ms=timing.group(2),
        valid=value(r"truth: valid") or value(r" valid"),
        truth_within_1px=value("within_1px"),
        truth_median_px=value("median_px"),
        truth_mae_px=value("mae_px"),
        coverage=value("coverage"),
        median_rel_err=value("median_rel_err"),
        p90_rel_err=value("p90_rel_err"),
        within_5pct=value("within_5pct"),
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("demo", help="the built stereo_disparity binary")
    parser.add_argument("--backends", default="cuda", help="comma-separated: cuda, ofa, pva")
    parser.add_argument("--count", type=int, default=30, help="frames per run")
    parser.add_argument("--period-ms", type=int, default=100, help="frame period")
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds per run")
    parser.add_argument(
        "--confidence", action="store_true", help="also output the confidence map (cuda, pva)"
    )
    parser.add_argument(
        "--clips",
        default="720p_emitter_on,480p_emitter_on",
        help="comma-separated endings of the recordings' folder names",
    )
    parser.add_argument(
        "--camera-depth-scale",
        type=float,
        help="meters per unit of the recordings' depth images; without it, no recordings",
    )
    args = parser.parse_args()
    sources = ["synthetic"]
    data = os.environ.get("HOLOSCAN_TEST_DATA")
    if data and args.camera_depth_scale:
        folders = sorted(path.name for path in (pathlib.Path(data) / "stereo-camera").iterdir())
        for suffix in args.clips.split(","):
            sources += [name for name in folders if name.endswith(suffix)]
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS)
    writer.writeheader()
    for backend in args.backends.split(","):
        for setting, source in itertools.product(settings(backend), sources):
            writer.writerow(measure(args.demo, backend, setting, source, args))
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
