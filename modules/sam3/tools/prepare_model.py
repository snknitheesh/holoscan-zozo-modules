#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

"""Prepare SAM 3 for holoscan-sam3's Sam3SegmentationOp.

The weights are not part of holoscan-sam3 or its repository. They are Meta's,
under the SAM License, and on Hugging Face their repository is gated: request
access at https://huggingface.co/facebook/sam3, accept the license there, and
give this script your own access token (--token, HF_TOKEN, or the token that
`hf auth login` saved). The script also needs --accept-sam-license, to record
that you accept the license for what it writes.

It downloads a pinned revision of the checkpoint (every file checked against
its SHA-256 digest), splits the model in two and exports both parts to ONNX
with PyTorch and Hugging Face Transformers, computes the text embeddings of a
list of prompts, checks the export, and writes into the output directory:

- sam3_encoder.onnx (and sam3_encoder.onnx.data, its weights): SAM 3's vision
  backbone and feature pyramid, run once per image. Input float32
  [1, 3, 1008, 1008], RGB normalized to [-1, 1]; outputs fpn0, fpn1 and fpn2,
  float32 [1, 256, 288, 288], [1, 256, 144, 144] and [1, 256, 72, 72].
- sam3_decoder.onnx: SAM 3's prompt (geometry) encoder, its DETR encoder and
  decoder, its scoring and its mask head, run once per prompt on the encoder's
  outputs. Inputs fpn0..fpn2, text_features [1, 32, 256], text_mask [1, 32],
  boxes [1, 4, 4] (center x, center y, width, height in fractions of the
  image), box_labels [1, 4] and box_valid [1, 4]; outputs masks
  [1, 200, 288, 288] (logits), boxes_xyxy [1, 200, 4] (fractions of the
  image) and scores [1, 200] (the class probability times the presence
  probability).
- sam3_prompts.bin: the prompt table, the text embeddings of the prompts
  (holoscan/sam3/prompts.hpp describes the format). Every run adds its
  prompts to the table; "visual", the text of prompts made of boxes alone, is
  always there.
- SAM_LICENSE: the SAM License, which applies to all of the above.

SAM 3's text encoder is not exported: it runs once per prompt, here. To use
new phrases, run the script again with --prompts and --prompts-only, which
only adds to the table.

PyTorch and Transformers are not needed to run this script, only to export: it
runs the export with the Python interpreter given by --python (by default its
own), which must have torch, torchvision, transformers, onnx and onnxscript
installed; for example a separate virtual environment. The ONNX files are
FP32; Sam3SegmentationOp builds its TensorRT engines with FP16 kernels by
default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HUGGING_FACE_URL = "https://huggingface.co/{repository}/resolve/{revision}/{file}"
DOWNLOAD_TIMEOUT_SECONDS = 300
TABLE_MAGIC = b"SAM3PRM1"
VISUAL_PROMPT = "visual"
MAX_PROMPT_BYTES = 256
ENCODER = "sam3_encoder.onnx"
DECODER = "sam3_decoder.onnx"
TABLE = "sam3_prompts.bin"
LICENSE_COPY = "SAM_LICENSE"


@dataclass(frozen=True)
class Checkpoint:
    """A pinned Hugging Face checkpoint: its files and their SHA-256 digests."""

    repository: str
    revision: str
    files: dict[str, str]
    license: str


CHECKPOINT = Checkpoint(
    "facebook/sam3",
    "3c879f39826c281e95690f02c7821c4de09afae7",
    {
        "config.json": "4616385e4b21f2e5e22c875b65679185cbccfa95de42542b9166f7dc3d57160f",
        "model.safetensors": "6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a",
        "processor_config.json": "6420cf2671fa9309ea95bc0144a8b9861666d1c5f43c8db09e410dacda974fce",
        "tokenizer.json": "6d9109cc838977f3ca94a379eec36aecc7c807e1785cd729660ca2fc0171fb35",
        "tokenizer_config.json": "39670ad98457fe8f14ca59f6bb74591e9fc850974380a63993e5b8ffc865baa2",
        "special_tokens_map.json": (
            "2cdb3b8331a60c92fc1e55a13e9fd61fd2293c5a51275fdcccd62b780052530e"
        ),
        "vocab.json": "5047b556ce86ccaf6aa22b3ffccfc52d391ea4accdab9c2f2407da5b742d4363",
        "merges.txt": "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a",
        "LICENSE": "bec48f70bd37bf8280a9d1ebf01642d26086f6122ba735baa08fe03c5a6e7448",
    },
    "SAM License (Meta, November 19, 2025)",
)

# The export's fixed shapes: SAM 3 takes 1008 x 1008 images (its position
# embeddings are for that size), has 200 queries and 32 text tokens; the
# decoder is exported with four box slots.
IMAGE_SIZE = 1008
FEATURES = [[1, 256, 288, 288], [1, 256, 144, 144], [1, 256, 72, 72]]
TOKENS = 32
DIM = 256
BOXES = 4
QUERIES = 200
MASK_SIZE = 288
ENCODER_SHAPES = ([[1, 3, IMAGE_SIZE, IMAGE_SIZE]], FEATURES)
DECODER_SHAPES = (
    [*FEATURES, [1, TOKENS, DIM], [1, TOKENS], [1, BOXES, 4], [1, BOXES], [1, BOXES]],
    [[1, QUERIES, MASK_SIZE, MASK_SIZE], [1, QUERIES, 4], [1, QUERIES]],
)

# The prompts every table gets, besides --prompts: the demo's synthetic
# shapes, and a few everyday things for a camera.
DEFAULT_PROMPTS = (
    "circle",
    "triangle",
    "blue rectangle",
    "person",
    "face",
    "hand",
    "cup",
    "bottle",
    "phone",
    "keyboard",
    "chair",
    "car",
)

# Run by the --python interpreter: argv is the checkpoint folder, the output
# folder, "all" or "prompts", and a JSON file with the prompts. It imports
# only what the export needs.
EXPORT_SCRIPT = r"""
import json
import struct
import sys
from pathlib import Path

import torch
from transformers import Sam3Model, Sam3Processor
from transformers.models.sam3.modeling_sam3 import box_cxcywh_to_xyxy, inverse_sigmoid

folder, out, what, prompts_file = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]
model = Sam3Model.from_pretrained(folder, local_files_only=True, attn_implementation="eager")
model = model.eval()
processor = Sam3Processor.from_pretrained(folder, local_files_only=True)
BOXES = 4


class Encoder(torch.nn.Module):
    # SAM 3's vision backbone and its feature pyramid; the pyramid's coarsest
    # level is not used by the image detector, and is left out.
    def __init__(self, model):
        super().__init__()
        self.vision = model.vision_encoder

    def forward(self, image):
        features = self.vision(image).fpn_hidden_states
        return features[0], features[1], features[2]


class Decoder(torch.nn.Module):
    # Sam3Model.forward from the encoder's features on, for one prompt. The
    # geometry encoder always runs, as in SAM 3's own predictor: with every
    # box slot invalid it contributes its CLS token alone.
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, fpn0, fpn1, fpn2, text_features, text_mask, boxes, box_labels, box_valid):
        m = self.model
        features = (fpn0, fpn1, fpn2)
        position = tuple(
            m.vision_encoder.neck.position_encoding(f.shape, f.device, f.dtype) for f in features
        )
        geometry = m.geometry_encoder(
            box_embeddings=boxes,
            box_mask=box_valid > 0.5,
            box_labels=(box_labels > 0.5).long(),
            img_feats=features,
            img_pos_embeds=position,
        )
        prompt = torch.cat([text_features, geometry.last_hidden_state], 1)
        prompt_mask = torch.cat([text_mask > 0.5, geometry.attention_mask], 1)
        encoded = m.detr_encoder(
            vision_features=[fpn2],
            text_features=prompt,
            vision_pos_embeds=[position[2]],
            text_mask=prompt_mask,
        )
        decoded = m.detr_decoder(
            vision_features=encoded.last_hidden_state,
            text_features=encoded.text_features,
            vision_pos_encoding=encoded.pos_embeds_flattened,
            text_mask=prompt_mask,
            spatial_shapes=encoded.spatial_shapes,
        )
        queries = decoded.intermediate_hidden_states[-1:]
        offsets = m.detr_decoder.box_head(queries)
        reference = inverse_sigmoid(decoded.reference_boxes[-1:])
        boxes_xyxy = box_cxcywh_to_xyxy((reference + offsets).sigmoid())[-1]
        logits = m.dot_product_scoring(
            decoder_hidden_states=queries,
            text_features=encoded.text_features,
            text_mask=prompt_mask,
        ).squeeze(-1)[-1]
        scores = logits.sigmoid() * decoded.presence_logits[-1].sigmoid()
        masks = m.mask_decoder(
            decoder_queries=queries[-1],
            backbone_features=list(features),
            encoder_hidden_states=encoded.last_hidden_state,
            prompt_features=prompt,
            prompt_mask=prompt_mask,
        ).pred_masks
        return masks, boxes_xyxy, scores


def embed(text):
    inputs = processor(text=text, return_tensors="pt")
    features = model.get_text_features(
        input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
    ).pooler_output
    return inputs.attention_mask[0].float(), features[0].float()


with torch.no_grad():
    prompts = json.loads(Path(prompts_file).read_text())
    entries = [(text, *embed(text)) for text in prompts]
    tokens, dim = entries[0][2].shape
    with (out / "text_embeddings.bin").open("wb") as table:
        table.write(b"SAM3PRM1" + struct.pack("<III", len(entries), tokens, dim))
        for text, mask, features in entries:
            name = text.encode("utf-8")
            table.write(struct.pack("<I", len(name)) + name)
            table.write(mask.numpy().astype("<f4").tobytes())
            table.write(features.numpy().astype("<f4").tobytes())
    if what == "all":
        image = torch.zeros(1, 3, 1008, 1008)
        encoder = Encoder(model).eval()
        features = encoder(image)
        torch.onnx.export(
            encoder, (image,), str(out / "sam3_encoder.onnx"), input_names=["image"],
            output_names=["fpn0", "fpn1", "fpn2"], opset_version=18, dynamo=True,
            external_data=True,
        )
        text_mask, text_features = entries[0][1], entries[0][2]
        example = (
            *features, text_features[None], text_mask[None], torch.zeros(1, BOXES, 4),
            torch.zeros(1, BOXES), torch.zeros(1, BOXES),
        )
        torch.onnx.export(
            Decoder(model).eval(), example, str(out / "sam3_decoder.onnx"),
            input_names=[
                "fpn0", "fpn1", "fpn2", "text_features", "text_mask", "boxes", "box_labels",
                "box_valid",
            ],
            output_names=["masks", "boxes_xyxy", "scores"], opset_version=18, dynamo=True,
            external_data=False,
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


def read_shapes(path: Path) -> tuple[list[list[int]], list[list[int]]]:
    """The shapes of an ONNX model's graph inputs and outputs, in order."""
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
    return [_shape(v) for v in inputs], [_shape(v) for v in _messages(graph, 12)]


def check_model(path: Path, expected: tuple[list[list[int]], list[list[int]]]) -> None:
    """Reject a model whose inputs or outputs Sam3SegmentationOp cannot use."""
    inputs, outputs = read_shapes(path)
    if inputs != expected[0]:
        raise PreparationError(f"{path.name} takes {inputs}, not {expected[0]}")
    if outputs != expected[1]:
        raise PreparationError(f"{path.name} gives {outputs}, not {expected[1]}")


# The prompt table (holoscan/sam3/prompts.hpp).


def read_table(path: Path) -> tuple[int, int, dict[str, bytes]]:
    """Tokens, dim and each text's mask and features, as raw bytes."""
    try:
        data = path.read_bytes()
    except OSError as error:
        raise PreparationError(f"{path}: cannot read: {error}") from error
    if len(data) < 20 or data[:8] != TABLE_MAGIC:
        raise PreparationError(f"{path}: not a prompt table")
    count, tokens, dim = struct.unpack_from("<III", data, 8)
    entry_bytes = 4 * (tokens + tokens * dim)
    offset = 20
    entries: dict[str, bytes] = {}
    for _ in range(count):
        if offset + 4 > len(data):
            raise PreparationError(f"{path}: truncated")
        (length,) = struct.unpack_from("<I", data, offset)
        offset += 4
        end = offset + length + entry_bytes
        if length == 0 or length > MAX_PROMPT_BYTES or end > len(data):
            raise PreparationError(f"{path}: a damaged entry")
        text = data[offset : offset + length].decode("utf-8")
        if text in entries:
            raise PreparationError(f"{path}: '{text}' twice")
        entries[text] = data[offset + length : end]
        offset = end
    if offset != len(data):
        raise PreparationError(f"{path}: data after its last entry")
    return tokens, dim, entries


def write_table(path: Path, tokens: int, dim: int, entries: dict[str, bytes]) -> None:
    """Write a prompt table, replacing the file atomically."""
    parts = [TABLE_MAGIC, struct.pack("<III", len(entries), tokens, dim)]
    for text, embedding in entries.items():
        name = text.encode("utf-8")
        parts += [struct.pack("<I", len(name)), name, embedding]
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(b"".join(parts))
    os.replace(partial, path)


def merge_tables(new: Path, existing: Path) -> list[str]:
    """Add the entries of table `new` to table `existing` (or make it); return its texts."""
    tokens, dim, entries = read_table(new)
    if (tokens, dim) != (TOKENS, DIM):
        raise PreparationError(f"the text embeddings are {tokens} x {dim}, not {TOKENS} x {DIM}")
    if existing.is_file():
        old_tokens, old_dim, old = read_table(existing)
        if (old_tokens, old_dim) == (tokens, dim):
            entries = {**old, **entries}
    write_table(existing, tokens, dim, entries)
    return list(entries)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def find_token(explicit: str | None) -> str | None:
    """The Hugging Face access token: given, from HF_TOKEN, or as `hf auth login` saved it."""
    if explicit:
        return explicit
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    try:
        saved = (home / "token").read_text().strip()
    except OSError:
        return None
    return saved or None


def download(url: str, sha256: str, destination: Path, token: str | None) -> None:
    """Download url to destination, which appears only when its digest matches."""
    partial = destination.with_name(destination.name + ".partial")
    request = urllib.request.Request(url)
    if token and url.startswith("https:"):
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with (
            urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response,
            partial.open("wb") as file,
        ):
            shutil.copyfileobj(response, file)
    except urllib.error.HTTPError as error:
        partial.unlink(missing_ok=True)
        if error.code in (401, 403):
            raise PreparationError(
                f"{url}: access denied ({error.code}). facebook/sam3 is gated: request access at "
                "https://huggingface.co/facebook/sam3, accept the SAM License there, and give "
                "your access token with --token or HF_TOKEN"
            ) from error
        raise PreparationError(f"cannot download {url}: {error}") from error
    except OSError as error:
        partial.unlink(missing_ok=True)
        raise PreparationError(f"cannot download {url}: {error}") from error
    actual = sha256_of(partial)
    if actual != sha256:
        partial.unlink(missing_ok=True)
        raise PreparationError(f"{url}: SHA-256 {actual} does not match the pinned {sha256}")
    os.replace(partial, destination)


def fetch_checkpoint(
    checkpoint: Checkpoint, folder: Path, cache: Path | None, token: str | None
) -> None:
    """Put the checkpoint's files in folder.

    A file already in `cache` with the pinned digest is used without a
    download; a downloaded one is kept there.
    """
    for name, sha256 in checkpoint.files.items():
        destination = folder / name
        cached = cache / checkpoint.repository.replace("/", "--") / name if cache else None
        if cached is not None and cached.is_file() and sha256_of(cached) == sha256:
            shutil.copyfile(cached, destination)
            continue
        if token is None:
            raise PreparationError(
                "facebook/sam3 is gated: request access at https://huggingface.co/facebook/sam3, "
                "accept the SAM License there, and give your access token with --token or "
                "HF_TOKEN (or log in with `hf auth login`)"
            )
        url = HUGGING_FACE_URL.format(
            repository=checkpoint.repository, revision=checkpoint.revision, file=name
        )
        print(f"prepare_model: downloading {url}", flush=True)
        download(url, sha256, destination, token)
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(destination, cached)


def export(python: str, folder: Path, out: Path, what: str, prompts: list[str]) -> None:
    """Run the export script with the given interpreter."""
    prompts_file = out / "prompts.json"
    prompts_file.write_text(json.dumps(prompts))
    command = [python, "-c", EXPORT_SCRIPT, str(folder), str(out), what, str(prompts_file)]
    print(
        f"prepare_model: {'exporting SAM 3 and ' if what == 'all' else ''}embedding "
        f"{len(prompts)} prompts with {python}",
        flush=True,
    )
    try:
        result = subprocess.run(command, check=False)
    except OSError as error:
        raise PreparationError(f"cannot run {python}: {error}") from error
    if result.returncode != 0:
        raise PreparationError(
            f"the export failed with exit status {result.returncode}; the interpreter needs "
            "torch, torchvision, transformers, onnx and onnxscript (pass another one with --python)"
        )
    expected = ["text_embeddings.bin"] + ([ENCODER, DECODER] if what == "all" else [])
    for name in expected:
        if not (out / name).is_file():
            raise PreparationError(f"the export wrote no {name}")


def parse_prompts(arguments: argparse.Namespace) -> list[str]:
    """The prompts to embed: the defaults, --prompts and --prompts-file, in order, once each."""
    prompts = [VISUAL_PROMPT, *DEFAULT_PROMPTS]
    if arguments.prompts:
        prompts += arguments.prompts.split(",")
    if arguments.prompts_file:
        try:
            prompts += arguments.prompts_file.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise PreparationError(f"cannot read {arguments.prompts_file}: {error}") from error
    result: list[str] = []
    for prompt in (p.strip() for p in prompts):
        if not prompt or prompt in result:
            continue
        if len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
            raise PreparationError(f"prompt '{prompt[:40]}...' is over {MAX_PROMPT_BYTES} bytes")
        result.append(prompt)
    return result


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Meta's SAM 3 at a pinned revision, export its encoder and decoder to ONNX, "
            "and embed text prompts, with PyTorch and Transformers. The weights are gated and "
            "under the SAM License."
        )
    )
    parser.add_argument("--output", type=Path, required=True, help="destination directory")
    parser.add_argument(
        "--accept-sam-license",
        action="store_true",
        help="required: you have read and accept the SAM License (see the module's NOTICE)",
    )
    parser.add_argument(
        "--token", help="Hugging Face access token (default: HF_TOKEN, or the saved login)"
    )
    parser.add_argument(
        "--prompts",
        help="comma-separated prompts to embed, besides the defaults (e.g. 'apple,red mug')",
    )
    parser.add_argument("--prompts-file", type=Path, help="a file with one prompt per line")
    parser.add_argument(
        "--prompts-only",
        action="store_true",
        help="only add the prompts to an existing export's table",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="the interpreter with torch and transformers that runs the export (default: this one)",
    )
    parser.add_argument(
        "--cache", type=Path, help="keep downloaded checkpoint files here, and reuse them"
    )
    parser.add_argument("--force", action="store_true", help="export again over an existing model")
    return parser.parse_args(arguments)


def prepare(arguments: argparse.Namespace) -> Path:
    """Prepare the model; return the output directory."""
    if not arguments.accept_sam_license:
        raise PreparationError(
            "SAM 3's weights are under the SAM License (https://huggingface.co/facebook/sam3/blob/"
            f"{CHECKPOINT.revision}/LICENSE); read it, and pass --accept-sam-license to accept it"
        )
    prompts = parse_prompts(arguments)
    output: Path = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    exported = all((output / name).is_file() for name in (ENCODER, DECODER, TABLE))
    if arguments.prompts_only and not exported:
        raise PreparationError(f"--prompts-only needs an export in {output}; run without it first")
    models = not arguments.prompts_only and (arguments.force or not exported)
    if exported and not models:
        check_model(output / ENCODER, ENCODER_SHAPES)
        check_model(output / DECODER, DECODER_SHAPES)
        _, _, entries = read_table(output / TABLE)
        missing = [p for p in prompts if p not in entries]
        if not missing:
            print(f"prepare_model: {output} is up to date; pass --force to export again")
            return output
        prompts = missing
    token = find_token(arguments.token)
    with tempfile.TemporaryDirectory(dir=output, prefix=".prepare-") as temporary:
        work = Path(temporary)
        folder = work / "checkpoint"
        out = work / "out"
        folder.mkdir()
        out.mkdir()
        fetch_checkpoint(CHECKPOINT, folder, arguments.cache, token)
        export(arguments.python, folder, out, "all" if models else "prompts", prompts)
        if models:
            check_model(out / ENCODER, ENCODER_SHAPES)
            check_model(out / DECODER, DECODER_SHAPES)
            # TensorRT engines cached for an earlier export would otherwise be
            # loaded for this one, and an old table may not fit it.
            for stem in (Path(ENCODER).stem, Path(DECODER).stem):
                for stale in output.glob(f"{stem}.*.engine.*"):
                    stale.unlink()
            (output / TABLE).unlink(missing_ok=True)
            for produced in sorted(out.glob("sam3_*.onnx*")):
                os.replace(produced, output / produced.name)
        texts = merge_tables(out / "text_embeddings.bin", output / TABLE)
        shutil.copyfile(folder / "LICENSE", output / LICENSE_COPY)
    print(
        f"prepare_model: wrote {output} ({CHECKPOINT.repository}@{CHECKPOINT.revision[:7]}, "
        f"{CHECKPOINT.license}); {len(texts)} prompts in {TABLE}",
        flush=True,
    )
    return output


def main(arguments: list[str] | None = None) -> int:
    try:
        prepare(parse_arguments(arguments))
    except PreparationError as error:
        print(f"prepare_model: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
