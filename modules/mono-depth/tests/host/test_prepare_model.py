# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

"""Tests of tools/prepare_model.py without the SDK, a GPU, PyTorch or the network.

A fake interpreter stands in for the export and writes fixture ONNX models,
downloads come from local file:// URLs, and the fixtures are encoded here as
minimal ONNX protobuf.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import stat
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import prepare_model


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


def onnx_model(input_dims: list[int | str], output_dims: list[int | str]) -> bytes:
    """A ModelProto with one graph input and one graph output."""
    graph = _field(11, _value_info("image", input_dims))
    # An initializer listed among the inputs, as older exporters do.
    graph += _field(5, _field(8, b"const") + _field(9, b"\0" * 4))
    graph += _field(11, _value_info("const", [1]))
    graph += _field(12, _value_info("depth", output_dims))
    return _field(1, 8) + _field(7, graph)


def fake_python(directory: Path, model: bytes, exit_status: int = 0) -> Path:
    """An interpreter that, given the export script, writes `model` where it is asked to."""
    fixture = directory / "fixture.onnx"
    fixture.write_bytes(model)
    script = directory / "python"
    script.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import pathlib, shutil, sys
            assert sys.argv[1] == "-c" and "torch.onnx.export" in sys.argv[2], sys.argv[:2]
            folder, onnx, width, height = sys.argv[3:7]
            assert (pathlib.Path(folder) / "config.json").read_bytes() == b"config"
            assert (pathlib.Path(folder) / "model.safetensors").read_bytes() == b"weights"
            pathlib.Path({str(directory / "size.txt")!r}).write_text(width + "x" + height)
            shutil.copyfile({str(fixture)!r}, onnx)
            sys.exit({exit_status})
            """
        )
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


class PrepareModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.output = self.root / "data"
        self.pinned = dict(prepare_model.CHECKPOINTS)
        # Every checkpoint served from a local folder, with fixture digests.
        served = self.root / "served"
        self.checkpoints = {}
        for name, checkpoint in prepare_model.CHECKPOINTS.items():
            folder = served / checkpoint.repository / checkpoint.revision
            folder.mkdir(parents=True)
            (folder / "config.json").write_bytes(b"config")
            (folder / "model.safetensors").write_bytes(b"weights")
            self.checkpoints[name] = dataclasses.replace(
                checkpoint,
                config_sha256=hashlib.sha256(b"config").hexdigest(),
                weights_sha256=hashlib.sha256(b"weights").hexdigest(),
            )
        self.served = served
        url = served.as_uri() + "/{repository}/{revision}/{file}"
        for patch in (
            mock.patch.object(prepare_model, "HUGGING_FACE_URL", url),
            mock.patch.dict(prepare_model.CHECKPOINTS, self.checkpoints),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(self._temporary.cleanup)

    def run_main(self, *arguments: str) -> tuple[int, str]:
        stderr = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            status = prepare_model.main(["--output", str(self.output), *arguments])
        return status, stderr.getvalue()

    def test_reads_model_inputs_and_outputs(self) -> None:
        path = self.root / "model.onnx"
        path.write_bytes(onnx_model([1, 3, 294, 518], [1, 294, 518]))
        self.assertEqual(prepare_model.read_shapes(path), ([1, 3, 294, 518], [1, 294, 518]))
        path.write_bytes(onnx_model(["batch", 3, "height", 518], [1, 1, 294, 518]))
        self.assertEqual(prepare_model.read_shapes(path)[0], [-1, 3, -1, 518])

    def test_rejects_malformed_files(self) -> None:
        for content in (b"", b"\xff\xff\xff", onnx_model([1, 3, 518, 518], [1, 518, 518])[:-3]):
            path = self.root / "bad.onnx"
            path.write_bytes(content)
            with self.assertRaises(prepare_model.PreparationError):
                prepare_model.read_shapes(path)

    def test_check_model(self) -> None:
        path = self.root / "model.onnx"
        path.write_bytes(onnx_model([1, 3, 294, 518], [1, 1, 294, 518]))
        prepare_model.check_model(path, 518, 294)
        with self.assertRaisesRegex(prepare_model.PreparationError, r"not \[1, 3, 518, 518\]"):
            prepare_model.check_model(path, 518, 518)
        path.write_bytes(onnx_model([1, 3, 294, 518], [1, 147, 259]))
        with self.assertRaisesRegex(prepare_model.PreparationError, r"not \[1, 294, 518\]"):
            prepare_model.check_model(path, 518, 294)

    def test_downloads_checks_and_exports_small(self) -> None:
        python = fake_python(self.root, onnx_model([1, 3, 294, 518], [1, 294, 518]))
        self.output.mkdir()
        stale = self.output / "depth_anything_v2_small_518x294.GPU.12.0.trt.10.16.engine.fp16"
        stale.write_bytes(b"old engine")
        status, error = self.run_main("--width", "518", "--height", "294", "--python", str(python))
        self.assertEqual(status, 0, error)
        self.assertEqual((self.root / "size.txt").read_text(), "518x294")
        # The engine cached for the previous export is gone, and nothing is
        # left of the work directory.
        self.assertFalse(stale.exists())
        self.assertEqual(
            [p.name for p in self.output.iterdir()], ["depth_anything_v2_small_518x294.onnx"]
        )

    def test_existing_model_is_kept_unless_forced(self) -> None:
        python = fake_python(self.root, onnx_model([1, 3, 518, 518], [1, 518, 518]))
        self.assertEqual(self.run_main("--python", str(python))[0], 0)
        onnx = self.output / "depth_anything_v2_small_518x518.onnx"
        onnx.write_bytes(onnx_model([1, 3, 518, 518], [1, 1, 518, 518]))
        self.assertEqual(self.run_main("--python", str(python))[0], 0)
        self.assertEqual(prepare_model.read_shapes(onnx)[1], [1, 1, 518, 518])
        self.assertEqual(self.run_main("--python", str(python), "--force")[0], 0)
        self.assertEqual(prepare_model.read_shapes(onnx)[1], [1, 518, 518])

    def test_rejects_a_wrong_digest(self) -> None:
        python = fake_python(self.root, onnx_model([1, 3, 518, 518], [1, 518, 518]))
        wrong = dataclasses.replace(self.checkpoints["small"], weights_sha256="0" * 64)
        with mock.patch.dict(prepare_model.CHECKPOINTS, {"small": wrong}):
            status, error = self.run_main("--python", str(python))
        self.assertEqual(status, 1)
        self.assertIn("does not match the pinned", error)
        self.assertFalse(any(self.output.glob("*.onnx")))

    def test_cache_is_filled_and_reused(self) -> None:
        python = fake_python(self.root, onnx_model([1, 3, 518, 518], [1, 518, 518]))
        cache = self.root / "cache"
        self.assertEqual(self.run_main("--python", str(python), "--cache", str(cache))[0], 0)
        cached = cache / "depth-anything--Depth-Anything-V2-Small-hf" / "model.safetensors"
        self.assertEqual(cached.read_bytes(), b"weights")
        # Without the server, the cache serves the export.
        repository = self.checkpoints["small"].repository
        (self.served / repository / self.checkpoints["small"].revision / "config.json").unlink()
        status, error = self.run_main("--python", str(python), "--cache", str(cache), "--force")
        self.assertEqual(status, 0, error)

    def test_non_commercial_models_need_their_license_accepted(self) -> None:
        python = fake_python(self.root, onnx_model([1, 3, 518, 518], [1, 518, 518]))
        for model in ("base", "large"):
            with self.subTest(model):
                status, error = self.run_main("--model", model, "--python", str(python))
                self.assertEqual(status, 1)
                self.assertIn("CC-BY-NC-4.0", error)
                self.assertIn("--accept-non-commercial-license", error)
                status, error = self.run_main(
                    "--model", model, "--python", str(python), "--accept-non-commercial-license"
                )
                self.assertEqual(status, 0, error)
                self.assertTrue((self.output / f"depth_anything_v2_{model}_518x518.onnx").is_file())

    def test_checkpoints_are_pinned(self) -> None:
        for name, checkpoint in self.pinned.items():
            with self.subTest(name):
                self.assertRegex(checkpoint.revision, r"^[0-9a-f]{40}$")
                self.assertRegex(checkpoint.config_sha256, r"^[0-9a-f]{64}$")
                self.assertRegex(checkpoint.weights_sha256, r"^[0-9a-f]{64}$")
                self.assertEqual(checkpoint.commercial, "CC-BY-NC" not in checkpoint.license)
        self.assertEqual(self.pinned["small"].license, "Apache-2.0")
        self.assertFalse(self.pinned["small"].metric)
        self.assertTrue(self.pinned["metric-indoor-small"].metric)
        self.assertFalse(self.pinned["base"].commercial or self.pinned["large"].commercial)

    def test_reports_a_failed_export(self) -> None:
        python = fake_python(self.root, onnx_model([1, 3, 518, 518], [1, 518, 518]), exit_status=3)
        status, error = self.run_main("--python", str(python))
        self.assertEqual(status, 1)
        self.assertIn("exit status 3", error)
        self.assertIn("torch, transformers", error)

    def test_rejects_an_export_of_another_size(self) -> None:
        python = fake_python(self.root, onnx_model([1, 3, 364, 364], [1, 364, 364]))
        status, error = self.run_main("--python", str(python))
        self.assertEqual(status, 1)
        self.assertIn("not [1, 3, 518, 518]", error)
        self.assertFalse(any(self.output.glob("*.onnx")))

    def test_missing_interpreter(self) -> None:
        status, error = self.run_main("--python", str(self.root / "missing" / "python"))
        self.assertEqual(status, 1)
        self.assertIn("cannot run", error)

    def test_rejects_bad_input_sizes(self) -> None:
        for option, size in (("--width", "500"), ("--height", "0"), ("--width", "2058")):
            status, error = self.run_main(option, size)
            self.assertEqual(status, 1)
            self.assertIn("multiple of 14", error)


if __name__ == "__main__":
    unittest.main()
