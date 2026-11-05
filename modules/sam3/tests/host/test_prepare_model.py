# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

"""Tests of tools/prepare_model.py without the SDK, a GPU, PyTorch or the network.

A fake interpreter stands in for the export and writes fixture ONNX models and
a prompt table, downloads come from local file:// URLs, and the fixtures are
encoded here as minimal ONNX protobuf.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import json
import os
import stat
import struct
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import prepare_model

TOKENS = prepare_model.TOKENS
DIM = prepare_model.DIM


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _field(number: int, payload: bytes | int) -> bytes:
    if isinstance(payload, int):
        return _varint(number << 3) + _varint(payload)
    return _varint((number << 3) | 2) + _varint(len(payload)) + payload


def _value_info(name: str, dims: list[int | str]) -> bytes:
    shape = b"".join(
        _field(1, _field(2, d.encode()) if isinstance(d, str) else _field(1, d)) for d in dims
    )
    tensor_type = _field(1, 1) + _field(2, shape)
    return _field(1, name.encode()) + _field(2, _field(1, tensor_type))


def onnx_model(inputs: list[list[int | str]], outputs: list[list[int | str]]) -> bytes:
    """A ModelProto with the given graph inputs and outputs."""
    graph = b"".join(_field(11, _value_info(f"in{i}", d)) for i, d in enumerate(inputs))
    # An initializer listed among the inputs, as older exporters do.
    graph += _field(5, _field(8, b"const") + _field(9, b"\0" * 4))
    graph += _field(11, _value_info("const", [1]))
    graph += b"".join(_field(12, _value_info(f"out{i}", d)) for i, d in enumerate(outputs))
    return _field(1, 8) + _field(7, graph)


ENCODER = onnx_model(*prepare_model.ENCODER_SHAPES)
DECODER = onnx_model(*prepare_model.DECODER_SHAPES)


def fake_python(directory: Path, encoder: bytes = ENCODER, exit_status: int = 0) -> Path:
    """An interpreter that, given the export script, writes the fixtures where it is asked to.

    Each prompt's embedding is its index in the run, repeated; every call is logged.
    """
    (directory / "encoder.onnx").write_bytes(encoder)
    (directory / "decoder.onnx").write_bytes(DECODER)
    script = directory / "python"
    script.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import json, pathlib, shutil, struct, sys
            assert sys.argv[1] == "-c" and "torch.onnx.export" in sys.argv[2], sys.argv[:2]
            folder, out, what, prompts_file = sys.argv[3:7]
            folder, out = pathlib.Path(folder), pathlib.Path(out)
            assert (folder / "model.safetensors").read_bytes() == b"model.safetensors"
            prompts = json.loads(pathlib.Path(prompts_file).read_text())
            with open({str(directory / "calls.log")!r}, "a") as log:
                log.write(what + " " + json.dumps(prompts) + "\\n")
            with (out / "text_embeddings.bin").open("wb") as table:
                table.write(b"SAM3PRM1" + struct.pack("<III", len(prompts), {TOKENS}, {DIM}))
                for index, text in enumerate(prompts):
                    name = text.encode()
                    table.write(struct.pack("<I", len(name)) + name)
                    table.write(struct.pack("<f", index) * ({TOKENS} + {TOKENS} * {DIM}))
            if what == "all":
                shutil.copyfile({str(directory / "encoder.onnx")!r}, out / "sam3_encoder.onnx")
                (out / "sam3_encoder.onnx.data").write_bytes(b"weights")
                shutil.copyfile({str(directory / "decoder.onnx")!r}, out / "sam3_decoder.onnx")
            sys.exit({exit_status})
            """
        )
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


class PrepareModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name)
        self.output = self.root / "data"
        self.pinned = prepare_model.CHECKPOINT
        # The checkpoint served from a local folder, each file holding its name.
        self.served = self.root / "served"
        folder = self.served / self.pinned.repository / self.pinned.revision
        folder.mkdir(parents=True)
        files = {}
        for name in self.pinned.files:
            (folder / name).write_bytes(name.encode())
            files[name] = hashlib.sha256(name.encode()).hexdigest()
        self.checkpoint = dataclasses.replace(self.pinned, files=files)
        url = self.served.as_uri() + "/{repository}/{revision}/{file}"
        for patch in (
            mock.patch.object(prepare_model, "HUGGING_FACE_URL", url),
            mock.patch.object(prepare_model, "CHECKPOINT", self.checkpoint),
            mock.patch.dict(os.environ, {"HF_TOKEN": "test-token", "HF_HOME": str(self.root)}),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        self.python = fake_python(self.root)

    def run_main(self, *arguments: str, accept: bool = True) -> tuple[int, str]:
        stderr = io.StringIO()
        flags = ["--accept-sam-license"] if accept else []
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            status = prepare_model.main(
                ["--output", str(self.output), "--python", str(self.python), *flags, *arguments]
            )
        return status, stderr.getvalue()

    def calls(self) -> list[tuple[str, list[str]]]:
        log = self.root / "calls.log"
        if not log.exists():
            return []
        return [
            (line.split(" ", 1)[0], json.loads(line.split(" ", 1)[1]))
            for line in log.read_text().splitlines()
        ]

    def test_reads_model_inputs_and_outputs(self) -> None:
        path = self.root / "model.onnx"
        path.write_bytes(onnx_model([[1, 3, 8, 8], ["batch", 4]], [[1, 2], [5]]))
        self.assertEqual(prepare_model.read_shapes(path), ([[1, 3, 8, 8], [-1, 4]], [[1, 2], [5]]))

    def test_rejects_malformed_files(self) -> None:
        for content in (b"", b"\xff\xff\xff", ENCODER[:-3]):
            path = self.root / "bad.onnx"
            path.write_bytes(content)
            with self.assertRaises(prepare_model.PreparationError):
                prepare_model.read_shapes(path)

    def test_check_model(self) -> None:
        path = self.root / "model.onnx"
        path.write_bytes(DECODER)
        prepare_model.check_model(path, prepare_model.DECODER_SHAPES)
        with self.assertRaisesRegex(prepare_model.PreparationError, r"takes .*, not "):
            prepare_model.check_model(path, prepare_model.ENCODER_SHAPES)
        inputs, outputs = prepare_model.DECODER_SHAPES
        path.write_bytes(onnx_model(inputs, outputs[:2]))
        with self.assertRaisesRegex(prepare_model.PreparationError, r"gives .*, not "):
            prepare_model.check_model(path, prepare_model.DECODER_SHAPES)

    def test_table_round_trip_and_merge(self) -> None:
        entry = struct.pack("<f", 1.0) * (TOKENS + TOKENS * DIM)
        first = self.root / "first.bin"
        prepare_model.write_table(first, TOKENS, DIM, {"visual": entry, "apple": entry})
        self.assertEqual(
            prepare_model.read_table(first), (TOKENS, DIM, {"visual": entry, "apple": entry})
        )
        merged = self.root / "merged.bin"
        prepare_model.write_table(merged, TOKENS, DIM, {"pear": entry})
        self.assertEqual(prepare_model.merge_tables(first, merged), ["pear", "visual", "apple"])
        for damaged in (b"", b"SAM3PRM2" + first.read_bytes()[8:], first.read_bytes()[:-1]):
            (self.root / "damaged.bin").write_bytes(damaged)
            with self.assertRaises(prepare_model.PreparationError):
                prepare_model.read_table(self.root / "damaged.bin")

    def test_exports_and_embeds(self) -> None:
        self.output.mkdir()
        stale = self.output / "sam3_encoder.GPU.12.0.170.trt.10.16.1.11.cfg0.engine.fp16"
        stale.write_bytes(b"old engine")
        status, error = self.run_main("--prompts", "apple, red mug,apple")
        self.assertEqual(status, 0, error)
        # One export, with "visual", the defaults and the new prompts, once each.
        expected = ["visual", *prepare_model.DEFAULT_PROMPTS, "apple", "red mug"]
        self.assertEqual(self.calls(), [("all", expected)])
        self.assertFalse(stale.exists())
        self.assertEqual(
            sorted(p.name for p in self.output.iterdir()),
            [
                "SAM_LICENSE",
                "sam3_decoder.onnx",
                "sam3_encoder.onnx",
                "sam3_encoder.onnx.data",
                "sam3_prompts.bin",
            ],
        )
        self.assertEqual((self.output / "SAM_LICENSE").read_bytes(), b"LICENSE")
        tokens, dim, entries = prepare_model.read_table(self.output / "sam3_prompts.bin")
        self.assertEqual((tokens, dim, list(entries)), (TOKENS, DIM, expected))

    def test_adds_prompts_without_exporting_again(self) -> None:
        self.assertEqual(self.run_main()[0], 0)
        status, error = self.run_main("--prompts", "pear", "--prompts-only")
        self.assertEqual(status, 0, error)
        self.assertEqual(self.calls()[1], ("prompts", ["pear"]))
        _, _, entries = prepare_model.read_table(self.output / "sam3_prompts.bin")
        self.assertEqual(list(entries)[-1], "pear")
        self.assertIn("circle", entries)
        # Nothing to do: no download, no export.
        (self.served / self.pinned.repository / self.pinned.revision / "LICENSE").unlink()
        self.assertEqual(self.run_main("--prompts", "pear")[0], 0)
        self.assertEqual(len(self.calls()), 2)
        # Without --prompts-only, a missing prompt is embedded; the models stay.
        prompts_file = self.root / "prompts.txt"
        prompts_file.write_text("kiwi\n\nplum\n")
        (self.served / self.pinned.repository / self.pinned.revision / "LICENSE").write_bytes(
            b"LICENSE"
        )
        self.assertEqual(self.run_main("--prompts-file", str(prompts_file))[0], 0)
        self.assertEqual(self.calls()[2], ("prompts", ["kiwi", "plum"]))

    def test_force_exports_again(self) -> None:
        self.assertEqual(self.run_main()[0], 0)
        self.assertEqual(self.run_main("--force")[0], 0)
        self.assertEqual([what for what, _ in self.calls()], ["all", "all"])

    def test_prompts_only_needs_an_export(self) -> None:
        status, error = self.run_main("--prompts-only", "--prompts", "pear")
        self.assertEqual(status, 1)
        self.assertIn("needs an export", error)

    def test_needs_the_license_accepted(self) -> None:
        status, error = self.run_main(accept=False)
        self.assertEqual(status, 1)
        self.assertIn("SAM License", error)
        self.assertIn("--accept-sam-license", error)
        self.assertEqual(self.calls(), [])

    def test_needs_a_token(self) -> None:
        with mock.patch.dict(os.environ, {"HF_TOKEN": ""}):
            status, error = self.run_main()
        self.assertEqual(status, 1)
        self.assertIn("gated", error)
        self.assertIn("--token", error)

    def test_finds_the_token(self) -> None:
        self.assertEqual(prepare_model.find_token("given"), "given")
        self.assertEqual(prepare_model.find_token(None), "test-token")
        (self.root / "token").write_text("saved\n")
        with mock.patch.dict(os.environ, {"HF_TOKEN": ""}):
            self.assertEqual(prepare_model.find_token(None), "saved")

    def test_rejects_a_wrong_digest(self) -> None:
        files = dict(self.checkpoint.files, **{"model.safetensors": "0" * 64})
        wrong = dataclasses.replace(self.checkpoint, files=files)
        with mock.patch.object(prepare_model, "CHECKPOINT", wrong):
            status, error = self.run_main()
        self.assertEqual(status, 1)
        self.assertIn("does not match the pinned", error)
        self.assertFalse(any(self.output.glob("*.onnx")))

    def test_cache_is_filled_and_reused(self) -> None:
        cache = self.root / "cache"
        self.assertEqual(self.run_main("--cache", str(cache))[0], 0)
        self.assertEqual(
            (cache / "facebook--sam3" / "model.safetensors").read_bytes(), b"model.safetensors"
        )
        # Without the server or a token, the cache serves the export.
        (self.served / self.pinned.repository / self.pinned.revision / "config.json").unlink()
        with mock.patch.dict(os.environ, {"HF_TOKEN": ""}):
            status, error = self.run_main("--cache", str(cache), "--force")
        self.assertEqual(status, 0, error)

    def test_reports_a_failed_export(self) -> None:
        self.python = fake_python(self.root, exit_status=3)
        status, error = self.run_main()
        self.assertEqual(status, 1)
        self.assertIn("exit status 3", error)
        self.assertIn("torch, torchvision, transformers", error)

    def test_rejects_an_export_of_another_shape(self) -> None:
        _, outputs = prepare_model.ENCODER_SHAPES
        self.python = fake_python(self.root, encoder=onnx_model([[1, 3, 672, 672]], outputs))
        status, error = self.run_main()
        self.assertEqual(status, 1)
        self.assertIn("sam3_encoder.onnx takes [[1, 3, 672, 672]]", error)
        self.assertFalse(any(self.output.glob("*.onnx")))

    def test_rejects_a_long_prompt(self) -> None:
        status, error = self.run_main("--prompts", "x" * 300)
        self.assertEqual(status, 1)
        self.assertIn("over 256 bytes", error)

    def test_missing_interpreter(self) -> None:
        self.python = self.root / "missing" / "python"
        status, error = self.run_main()
        self.assertEqual(status, 1)
        self.assertIn("cannot run", error)

    def test_checkpoint_is_pinned(self) -> None:
        self.assertRegex(self.pinned.revision, r"^[0-9a-f]{40}$")
        self.assertIn("model.safetensors", self.pinned.files)
        self.assertIn("LICENSE", self.pinned.files)
        for name, digest in self.pinned.files.items():
            with self.subTest(name):
                self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertIn("SAM License", self.pinned.license)


if __name__ == "__main__":
    unittest.main()
