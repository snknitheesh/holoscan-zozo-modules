#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

"""Prepare a Fast-FoundationStereo ONNX model for holoscan-foundation-stereo's FoundationStereoOp.

The models and the upstream code are NVIDIA's, under NVIDIA's licenses, and are
not part of holoscan-foundation-stereo or its repository. This script writes a
single-file ONNX model into the output directory in one of two ways:

- fetch (the default): downloads NVIDIA's C-Fast-FoundationStereo ONNX export
  for 480x736 inputs from Hugging Face at a pinned revision, checks its SHA-256
  digest and its inputs and output, and writes
  c_fast_foundationstereo_480x736.onnx. No PyTorch is needed. The model is
  under the NVIDIA Open Model Agreement.
- export: runs the upstream repository's scripts/make_single_onnx.py, at a
  pinned commit, with a separately installed Python that has PyTorch and the
  upstream requirements, to export a checkpoint for another input size or
  iteration count: the pinned C-Fast-FoundationStereo checkpoint (downloaded
  and checked), or a research checkpoint you downloaded yourself
  (--checkpoint). The upstream code is under NVIDIA's non-commercial research
  license; see tools/README.md.

Either way the model takes `left_image` and `right_image`, float32
[1, 3, H, W] ImageNet-normalized RGB, and gives `disparity`, float32
[1, 1, H, W]. The precision is chosen when TensorRT builds the engine
(FoundationStereoOp's `fp16`, on by default), not here.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HF_REVISION = "53a84e28a368f37e699776250215fcb450d6a109"
HF_BASE = f"https://huggingface.co/nvidia/c-fast-foundationstereo/resolve/{HF_REVISION}"
FETCHED_ONNX = (
    f"{HF_BASE}/onnx/c_fast_foundationstereo_bp2_dynbatch_480x736.onnx",
    "794647dc718c9564d5ccb56d15527b70a9267e8ddfbd5ecb1e784b323fbd60ed",
)
FETCHED_NAME = "c_fast_foundationstereo_480x736"
FETCHED_SIZE = (480, 736)
CHECKPOINT = (
    f"{HF_BASE}/model_best_bp2_serialize.pth",
    "7aee85948373da62b0503c2542507129a3e7cab9d97d10e6790d89512a7db214",
)
UPSTREAM_URL = "https://github.com/NVlabs/Fast-FoundationStereo.git"
UPSTREAM_COMMIT = "476f4249561f7c79ca707326954f9255643412a6"
DOWNLOAD_TIMEOUT_SECONDS = 300
INPUT_NAMES = ("left_image", "right_image")
# Run with the export's Python in the upstream checkout: copies a checkpoint,
# adding the `normalize` setting that the pinned upstream code reads and that
# the C-Fast-FoundationStereo checkpoint predates, with the value upstream
# assumes when it is missing (True; see its scripts/make_plugin_onnx.py).
COMPLETE_CHECKPOINT = """
import sys
import torch
from omegaconf import open_dict
sys.path.insert(0, ".")
model = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
with open_dict(model.args):
    if "normalize" not in model.args:
        print("prepare_model: the checkpoint has no normalize setting; using True", flush=True)
        model.args.normalize = True
torch.save(model, sys.argv[2])
"""
OUTPUT_NAME = "disparity"


class PreparationError(RuntimeError):
    """A user-facing model preparation failure."""


@dataclass(frozen=True)
class ModelInfo:
    """The graph inputs and outputs of an ONNX model: name to dimensions."""

    inputs: dict[str, list[int | str]]
    outputs: dict[str, list[int | str]]


# A minimal protobuf reader, enough to find the inputs and outputs of an ONNX
# ModelProto without the onnx package.


def _varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if offset >= len(data) or shift > 63:
            raise PreparationError("malformed ONNX file: bad varint")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7


def _fields(data: bytes) -> list[tuple[int, int, int | bytes]]:
    """Return (field number, wire type, value) for each field of a message."""
    fields = []
    offset = 0
    while offset < len(data):
        key, offset = _varint(data, offset)
        number, wire_type = key >> 3, key & 7
        if wire_type == 0:
            value, offset = _varint(data, offset)
        elif wire_type == 1:
            value, offset = data[offset : offset + 8], offset + 8
        elif wire_type == 2:
            length, offset = _varint(data, offset)
            if offset + length > len(data):
                raise PreparationError("malformed ONNX file: truncated field")
            value, offset = data[offset : offset + length], offset + length
        elif wire_type == 5:
            value, offset = data[offset : offset + 4], offset + 4
        else:
            raise PreparationError(f"malformed ONNX file: wire type {wire_type}")
        fields.append((number, wire_type, value))
    if offset != len(data):
        raise PreparationError("malformed ONNX file: truncated message")
    return fields


def _messages(data: bytes, number: int) -> list[bytes]:
    return [v for n, w, v in _fields(data) if n == number and w == 2 and isinstance(v, bytes)]


def _string(data: bytes, number: int) -> str:
    values = _messages(data, number)
    return values[0].decode("utf-8") if values else ""


def _shape(value_info: bytes) -> list[int | str]:
    """The dimensions of a tensor ValueInfoProto; a symbolic one by its name, or '?'."""
    dims: list[int | str] = []
    for type_proto in _messages(value_info, 2):
        for tensor_type in _messages(type_proto, 1):
            for shape in _messages(tensor_type, 2):
                for dim in _messages(shape, 1):
                    values = [v for n, w, v in _fields(dim) if n == 1 and w == 0]
                    if values:
                        dims.append(int(values[0]))
                    else:
                        dims.append(_string(dim, 2) or "?")
    return dims


def read_model_info(path: Path) -> ModelInfo:
    """Read the graph inputs and outputs of an ONNX model."""
    try:
        data = path.read_bytes()
    except OSError as error:
        raise PreparationError(f"{path}: cannot read: {error}") from error
    graphs = _messages(data, 7)
    if len(graphs) != 1:
        raise PreparationError(f"{path}: not an ONNX model")
    graph = graphs[0]
    initializers = {_string(tensor, 8) for tensor in _messages(graph, 5)}
    inputs = {
        _string(v, 1): _shape(v) for v in _messages(graph, 11) if _string(v, 1) not in initializers
    }
    outputs = {_string(v, 1): _shape(v) for v in _messages(graph, 12)}
    return ModelInfo(inputs, outputs)


def check_model(info: ModelInfo, height: int, width: int) -> None:
    """Reject a model the operator cannot run at height x width."""
    if tuple(info.inputs) != INPUT_NAMES:
        raise PreparationError(
            f"the model's inputs are {list(info.inputs)}, not {list(INPUT_NAMES)}; "
            "it is not a single-file Fast-FoundationStereo export"
        )
    for name, dims in info.inputs.items():
        # A symbolic batch dimension is built for 1.
        batch_ok = dims[:1] == [1] or (len(dims) > 0 and isinstance(dims[0], str))
        if len(dims) != 4 or dims[1:] != [3, height, width] or not batch_ok:
            raise PreparationError(f"input {name} is {dims}, not [1, 3, {height}, {width}]")
    if list(info.outputs) != [OUTPUT_NAME] or len(info.outputs[OUTPUT_NAME]) != 4:
        raise PreparationError(f"the model's outputs are {info.outputs}, not one 4-D {OUTPUT_NAME}")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, sha256: str, destination: Path) -> None:
    """Download url to destination, which appears only when its digest matches."""
    partial = destination.with_name(destination.name + ".partial")
    print(f"prepare_model: downloading {url}", flush=True)
    try:
        with (
            urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response,
            partial.open("wb") as file,
        ):
            shutil.copyfileobj(response, file)
    except OSError as error:
        partial.unlink(missing_ok=True)
        raise PreparationError(f"cannot download {url}: {error}") from error
    actual = sha256_of(partial)
    if actual != sha256:
        partial.unlink(missing_ok=True)
        raise PreparationError(f"{url}: SHA-256 {actual} does not match the pinned {sha256}")
    os.replace(partial, destination)


def remove_stale_engines(output: Path, stem: str) -> None:
    """Remove TensorRT engines cached for an earlier model of the same name."""
    for stale in output.glob(f"{stem}.*engine*"):
        stale.unlink()


def fetch(arguments: argparse.Namespace) -> Path:
    output: Path = arguments.output
    onnx = output / f"{FETCHED_NAME}.onnx"
    url, sha256 = FETCHED_ONNX
    if onnx.is_file() and not arguments.force:
        if sha256_of(onnx) != sha256:
            raise PreparationError(f"{onnx} exists and is not the pinned model; pass --force")
        print(f"prepare_model: {onnx} exists; pass --force to fetch it again", flush=True)
        return onnx
    with tempfile.TemporaryDirectory(dir=output, prefix=".prepare-") as temporary:
        downloaded = Path(temporary) / onnx.name
        download(url, sha256, downloaded)
        check_model(read_model_info(downloaded), *FETCHED_SIZE)
        remove_stale_engines(output, FETCHED_NAME)
        os.replace(downloaded, onnx)
    print(f"prepare_model: wrote {onnx}", flush=True)
    return onnx


def upstream_checkout(arguments: argparse.Namespace, workdir: Path) -> Path:
    """The upstream repository at the pinned commit: --repo, or a fresh clone."""
    if arguments.repo:
        repo: Path = arguments.repo
        try:
            head = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise PreparationError(f"{repo}: not a git checkout: {error}") from error
        if head != UPSTREAM_COMMIT and not arguments.any_commit:
            raise PreparationError(
                f"{repo} is at {head}, not the pinned {UPSTREAM_COMMIT}; check that commit "
                "out, or pass --any-commit to export with it anyway"
            )
        return repo
    repo = workdir / "Fast-FoundationStereo"
    print(f"prepare_model: cloning {UPSTREAM_URL} at {UPSTREAM_COMMIT}", flush=True)
    try:
        subprocess.run(["git", "clone", "--quiet", UPSTREAM_URL, str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "--quiet", UPSTREAM_COMMIT], check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise PreparationError(f"cannot clone {UPSTREAM_URL}: {error}") from error
    return repo


def run(command: list[str], cwd: Path, what: str) -> None:
    try:
        result = subprocess.run(command, cwd=cwd, check=False)
    except OSError as error:
        raise PreparationError(f"cannot run {command[0]}: {error}") from error
    if result.returncode != 0:
        raise PreparationError(f"{what} failed with exit status {result.returncode}")


def export(arguments: argparse.Namespace) -> Path:
    output: Path = arguments.output
    height, width = arguments.height, arguments.width
    stem = arguments.name or (
        f"{'fast_foundationstereo' if arguments.checkpoint else 'c_fast_foundationstereo'}"
        f"_{height}x{width}_it{arguments.valid_iters}"
    )
    onnx = output / f"{stem}.onnx"
    if onnx.is_file() and not arguments.force:
        check_model(read_model_info(onnx), height, width)
        print(f"prepare_model: {onnx} exists; pass --force to export it again", flush=True)
        return onnx
    python = shutil.which(arguments.python)
    if python is None:
        raise PreparationError(
            f"the Python '{arguments.python}' was not found; create a virtual environment with "
            "PyTorch and the upstream requirements (see tools/README.md) and pass its python "
            "with --python"
        )
    with tempfile.TemporaryDirectory(dir=output, prefix=".prepare-") as temporary:
        workdir = Path(temporary)
        repo = upstream_checkout(arguments, workdir)
        if arguments.checkpoint:
            checkpoint: Path = arguments.checkpoint
            if not checkpoint.is_file():
                raise PreparationError(f"{checkpoint}: no such file")
            print(
                f"prepare_model: exporting {checkpoint} (SHA-256 {sha256_of(checkpoint)})",
                flush=True,
            )
        else:
            checkpoint = workdir / "model_best_bp2_serialize.pth"
            download(*CHECKPOINT, checkpoint)
        completed = workdir / "checkpoint.pth"
        run(
            [python, "-c", COMPLETE_CHECKPOINT, str(checkpoint.resolve()), str(completed)],
            repo,
            "reading the checkpoint",
        )
        command = [
            python,
            str(repo / "scripts" / "make_single_onnx.py"),
            "--model_dir",
            str(completed),
            "--save_path",
            str(workdir / "export"),
            "--height",
            str(height),
            "--width",
            str(width),
            "--valid_iters",
            str(arguments.valid_iters),
            "--max_disp",
            str(arguments.max_disp),
            "--onnx_name",
            stem,
        ]
        print("prepare_model: running " + " ".join(command), flush=True)
        run(command, repo, "the export")
        exported = workdir / "export" / f"{stem}.onnx"
        if not exported.is_file():
            raise PreparationError(f"the export did not write {exported.name}")
        check_model(read_model_info(exported), height, width)
        remove_stale_engines(output, stem)
        os.replace(exported, onnx)
    print(f"prepare_model: wrote {onnx}", flush=True)
    return onnx


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch NVIDIA's C-Fast-FoundationStereo ONNX model (480x736), or export a "
            "Fast-FoundationStereo checkpoint to ONNX with the upstream code."
        )
    )
    parser.add_argument("--output", type=Path, required=True, help="destination directory")
    parser.add_argument(
        "--source",
        choices=("fetch", "export"),
        default="fetch",
        help="fetch the pinned ONNX model (default), or export a checkpoint",
    )
    parser.add_argument("--force", action="store_true", help="replace an existing model")
    export_options = parser.add_argument_group("export")
    export_options.add_argument(
        "--python",
        default="python3",
        help="a Python with PyTorch and the upstream requirements (default: python3 on PATH)",
    )
    export_options.add_argument(
        "--checkpoint",
        type=Path,
        help="a research checkpoint (model_best_bp2_serialize.pth) to export instead of the "
        "pinned C-Fast-FoundationStereo one",
    )
    export_options.add_argument(
        "--repo", type=Path, help="an upstream checkout at the pinned commit, instead of cloning"
    )
    export_options.add_argument(
        "--any-commit", action="store_true", help="accept --repo at another commit"
    )
    export_options.add_argument("--height", type=int, default=480, help="input height (480)")
    export_options.add_argument("--width", type=int, default=640, help="input width (640)")
    export_options.add_argument(
        "--valid-iters", type=int, default=8, help="refinement iterations (8)"
    )
    export_options.add_argument(
        "--max-disp", type=int, default=192, help="largest disparity searched, in pixels (192)"
    )
    export_options.add_argument("--name", help="file name of the model, without .onnx")
    return parser.parse_args(arguments)


def prepare(arguments: argparse.Namespace) -> Path:
    """Prepare the model; return its ONNX path."""
    if arguments.source == "export":
        for option in ("height", "width"):
            value = getattr(arguments, option)
            if value < 32 or value > 4096 or value % 32 != 0:
                raise PreparationError(f"--{option} must be a multiple of 32 in 32..4096")
        if not 1 <= arguments.valid_iters <= 32:
            raise PreparationError("--valid-iters must be 1..32")
        if not 16 <= arguments.max_disp <= 1024:
            raise PreparationError("--max-disp must be 16..1024")
    elif arguments.checkpoint or arguments.repo:
        raise PreparationError("--checkpoint and --repo apply to --source export")
    arguments.output.mkdir(parents=True, exist_ok=True)
    return export(arguments) if arguments.source == "export" else fetch(arguments)


def main(arguments: list[str] | None = None) -> int:
    try:
        prepare(parse_arguments(arguments))
    except PreparationError as error:
        print(f"prepare_model: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
