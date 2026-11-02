#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

"""Prepare a Depth Anything V2 ONNX model for holoscan-mono-depth's MonoDepthOp.

The weights are not part of holoscan-mono-depth or its repository. This script
downloads a pinned checkpoint from Hugging Face (its config.json and
model.safetensors at a fixed revision, each checked against its SHA-256 digest),
exports it to ONNX at a fixed input size with PyTorch and Hugging Face
Transformers, checks the export, and writes into the output directory:

- <model>_<W>x<H>.onnx: input float32 [1, 3, H, W], RGB normalized with the
  ImageNet mean and standard deviation; output float32 [1, H, W]. For the
  relative models the output is relative inverse depth (larger is nearer, up to
  an unknown scale and shift); for the metric models it is depth in meters.

The default, `small`, is Depth Anything V2 Small, under Apache-2.0. `base` and
`large` are under CC-BY-NC-4.0, for non-commercial use only, and need
--accept-non-commercial-license. The metric models are fine-tuned from Small.

PyTorch and Transformers are not needed to run this script, only to export: it
runs the export with the Python interpreter given by --python (by default its
own), which must have torch, transformers and safetensors installed; for example
a separate virtual environment. The ONNX file is FP32; MonoDepthOp builds its
TensorRT engine with FP16 kernels by default.
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

HUGGING_FACE_URL = "https://huggingface.co/{repository}/resolve/{revision}/{file}"
DOWNLOAD_TIMEOUT_SECONDS = 300
PATCH_SIZE = 14
MIN_SIDE = 14
MAX_SIDE = 2044


@dataclass(frozen=True)
class Checkpoint:
    """A pinned Hugging Face checkpoint."""

    repository: str
    revision: str
    config_sha256: str
    weights_sha256: str
    license: str
    commercial: bool
    metric: bool


CHECKPOINTS = {
    "small": Checkpoint(
        "depth-anything/Depth-Anything-V2-Small-hf",
        "5426e4f0f36572d16453bbda7a8389317b1bef99",
        "c56698d3643dde1f83ea2212759e6b31a22b8f827246a36dd007ee8a22b3ff75",
        "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70",
        "Apache-2.0",
        commercial=True,
        metric=False,
    ),
    "base": Checkpoint(
        "depth-anything/Depth-Anything-V2-Base-hf",
        "b1958afc87fb45a9e3746cb387596094de553ed8",
        "1f616484953fbbddcc14da69ab81ef834fd58c11be04fa2f2eecf5696b2f7389",
        "9996d15c334d264cfcc29f4318d7d3789a006ecc444782c2fb82f7eaa1acb8bf",
        "CC-BY-NC-4.0",
        commercial=False,
        metric=False,
    ),
    "large": Checkpoint(
        "depth-anything/Depth-Anything-V2-Large-hf",
        "7581137eff8d4e94f6e796d3baea0e9fa79b22d2",
        "0ceca59ffa2350fb7c19e4565e2e5d28e2d491ec296ba91ab8c779474f9a7458",
        "4e01e34ed5549b529b70b92d53226bc370f03041977b390d3dde45d47f516cf9",
        "CC-BY-NC-4.0",
        commercial=False,
        metric=False,
    ),
    "metric-indoor-small": Checkpoint(
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        "8078d68a9c75a972131914f6afd0c1723be0da7f",
        "0b1d9fc591693d16b864249a9cfcf5264c8c00be999c1643fa1c64dc105d55f8",
        "e990eb82fbf11b05b7813261196a2b841bdcf5a05f64396724a8987fa90504a3",
        "Apache-2.0 (as the original Metric-Hypersim-Small checkpoint)",
        commercial=True,
        metric=True,
    ),
    "metric-outdoor-small": Checkpoint(
        "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
        "fd2c22027eaf20374204f14099b8341e1925ad39",
        "3417ccb9df6fafe7e03c46583db7185d15e8ad4f5bbbdd28295bb06cfb132b48",
        "ad065c77a7421ca55159a1f0db9433397a607690f2d76bb8a6fc54b1be7a3124",
        "Apache-2.0 (as the original Metric-VKITTI-Small checkpoint)",
        commercial=True,
        metric=True,
    ),
}

# Run by the --python interpreter: argv is the checkpoint folder, the ONNX
# path, the input width and height. It imports only what the export needs.
EXPORT_SCRIPT = r"""
import sys

import torch
from transformers import DepthAnythingForDepthEstimation

folder, onnx, width, height = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
model = DepthAnythingForDepthEstimation.from_pretrained(folder, local_files_only=True).eval()


class Depth(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model(pixel_values=image).predicted_depth


example = torch.zeros(1, 3, height, width)
with torch.no_grad():
    torch.onnx.export(
        Depth(model), (example,), onnx, input_names=["image"], output_names=["depth"],
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
"""


class PreparationError(RuntimeError):
    """A user-facing model preparation failure."""


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


def _shape(value_info: bytes) -> list[int]:
    """The dimensions of a tensor ValueInfoProto; -1 for a symbolic one."""
    dims = []
    for type_proto in _messages(value_info, 2):
        for tensor_type in _messages(type_proto, 1):
            for shape in _messages(tensor_type, 2):
                for dim in _messages(shape, 1):
                    values = [v for n, w, v in _fields(dim) if n == 1 and w == 0]
                    dims.append(int(values[0]) if values else -1)
    return dims


def read_shapes(path: Path) -> tuple[list[int], list[int]]:
    """The shapes of an ONNX model's one graph input and one graph output."""
    try:
        data = path.read_bytes()
    except OSError as error:
        raise PreparationError(f"{path}: cannot read: {error}") from error
    graphs = _messages(data, 7)
    if len(graphs) != 1:
        raise PreparationError(f"{path}: not an ONNX model")
    graph = graphs[0]
    initializers = {_string(tensor, 8) for tensor in _messages(graph, 5)}
    inputs = [v for v in _messages(graph, 11) if _string(v, 1) not in initializers]
    outputs = _messages(graph, 12)
    if len(inputs) != 1 or len(outputs) != 1:
        raise PreparationError(f"{path}: expected one input and one output")
    return _shape(inputs[0]), _shape(outputs[0])


def check_model(path: Path, width: int, height: int) -> None:
    """Reject a model MonoDepthOp cannot run at width x height."""
    input_shape, output_shape = read_shapes(path)
    if input_shape != [1, 3, height, width]:
        raise PreparationError(f"the model takes {input_shape}, not [1, 3, {height}, {width}]")
    if output_shape not in ([1, height, width], [1, 1, height, width]):
        raise PreparationError(f"the model gives {output_shape}, not [1, {height}, {width}]")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, sha256: str, destination: Path) -> None:
    """Download url to destination, which appears only when its digest matches."""
    partial = destination.with_name(destination.name + ".partial")
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


def fetch_checkpoint(checkpoint: Checkpoint, folder: Path, cache: Path | None) -> None:
    """Put the checkpoint's config.json and model.safetensors in folder.

    A file already in `cache` with the pinned digest is used without a
    download; a downloaded one is kept there.
    """
    for name, sha256 in (
        ("config.json", checkpoint.config_sha256),
        ("model.safetensors", checkpoint.weights_sha256),
    ):
        destination = folder / name
        cached = cache / checkpoint.repository.replace("/", "--") / name if cache else None
        if cached is not None and cached.is_file() and sha256_of(cached) == sha256:
            shutil.copyfile(cached, destination)
            continue
        url = HUGGING_FACE_URL.format(
            repository=checkpoint.repository, revision=checkpoint.revision, file=name
        )
        print(f"prepare_model: downloading {url}", flush=True)
        download(url, sha256, destination)
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(destination, cached)


def export(python: str, folder: Path, onnx: Path, width: int, height: int) -> None:
    """Export the checkpoint in folder to onnx with the given interpreter."""
    command = [python, "-c", EXPORT_SCRIPT, str(folder), str(onnx), str(width), str(height)]
    print(f"prepare_model: exporting {width}x{height} with {python}", flush=True)
    try:
        result = subprocess.run(command, check=False)
    except OSError as error:
        raise PreparationError(f"cannot run {python}: {error}") from error
    if result.returncode != 0:
        raise PreparationError(
            f"the export failed with exit status {result.returncode}; the interpreter "
            "needs torch, transformers and safetensors (pass another one with --python)"
        )
    if not onnx.is_file():
        raise PreparationError("the export wrote no ONNX file")


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a pinned Depth Anything V2 checkpoint and export it to ONNX at a fixed "
            "input size, with PyTorch and Transformers."
        )
    )
    parser.add_argument("--output", type=Path, required=True, help="destination directory")
    parser.add_argument(
        "--model",
        choices=list(CHECKPOINTS),
        default="small",
        help=(
            "checkpoint (default: small, Apache-2.0); base and large are CC-BY-NC-4.0, "
            "non-commercial only; the metric models give depth in meters"
        ),
    )
    parser.add_argument(
        "--width", type=int, default=518, help="model input width, a multiple of 14 (default: 518)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=518,
        help="model input height, a multiple of 14 (default: 518)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="the interpreter with torch and transformers that runs the export (default: this one)",
    )
    parser.add_argument(
        "--cache", type=Path, help="keep downloaded checkpoints here, and reuse them"
    )
    parser.add_argument(
        "--accept-non-commercial-license",
        action="store_true",
        help="needed for base and large, whose CC-BY-NC-4.0 license forbids commercial use",
    )
    parser.add_argument("--force", action="store_true", help="export again over an existing model")
    return parser.parse_args(arguments)


def model_path(arguments: argparse.Namespace) -> Path:
    return (
        arguments.output
        / f"depth_anything_v2_{arguments.model}_{arguments.width}x{arguments.height}.onnx"
    )


def prepare(arguments: argparse.Namespace) -> Path:
    """Prepare the model; return the ONNX path."""
    for name in ("width", "height"):
        side = getattr(arguments, name)
        if side < MIN_SIDE or side > MAX_SIDE or side % PATCH_SIZE != 0:
            raise PreparationError(
                f"--{name} must be a multiple of {PATCH_SIZE} in {MIN_SIDE}..{MAX_SIDE}"
            )
    checkpoint = CHECKPOINTS[arguments.model]
    if not checkpoint.commercial and not arguments.accept_non_commercial_license:
        raise PreparationError(
            f"{arguments.model} is licensed under {checkpoint.license}, for non-commercial use "
            "only; pass --accept-non-commercial-license to use it, or use small (Apache-2.0)"
        )
    output: Path = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    onnx = model_path(arguments)
    if onnx.is_file() and not arguments.force:
        check_model(onnx, arguments.width, arguments.height)
        print(f"prepare_model: {onnx} exists; pass --force to export it again", flush=True)
        return onnx
    with tempfile.TemporaryDirectory(dir=output, prefix=".prepare-") as temporary:
        folder = Path(temporary)
        fetch_checkpoint(checkpoint, folder, arguments.cache)
        exported = folder / "model.onnx"
        export(arguments.python, folder, exported, arguments.width, arguments.height)
        check_model(exported, arguments.width, arguments.height)
        # A TensorRT engine cached for an earlier export of the same name
        # would otherwise be loaded for this one.
        for stale in output.glob(f"{onnx.stem}.*.engine.*"):
            stale.unlink()
        os.replace(exported, onnx)
    kind = "depth in meters" if checkpoint.metric else "relative inverse depth"
    print(
        f"prepare_model: wrote {onnx} ({checkpoint.repository}, {checkpoint.license}; "
        f"output: {kind})",
        flush=True,
    )
    return onnx


def main(arguments: list[str] | None = None) -> int:
    try:
        prepare(parse_arguments(arguments))
    except PreparationError as error:
        print(f"prepare_model: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
