# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

"""Tests of tools/prepare_model.py without the SDK, a GPU, PyTorch or the network.

Downloads come from local file:// URLs, a fake Python stands in for the
upstream exporter, and the fixtures are encoded here as minimal ONNX protobuf.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import shutil
import stat
import subprocess
import sys
import tempfile
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


def onnx_model(
    height: int = 480,
    width: int = 736,
    batch: int | str = "batch_size",
    inputs: tuple[str, ...] = ("left_image", "right_image"),
    output: str = "disparity",
) -> bytes:
    """A ModelProto with the graph inputs and output of a stereo export."""
    graph = b"".join(_field(11, _value_info(n, [batch, 3, height, width])) for n in inputs)
    # An initializer, which some exporters also list among the inputs.
    graph += _field(5, _field(8, b"weight") + _field(9, b"\0" * 4))
    graph += _field(11, _value_info("weight", [1]))
    graph += _field(12, _value_info(output, [batch, 1, "h", "w"]))
    return _field(1, 8) + _field(7, graph)


FAKE_PYTHON = """\
#!{python}
# Stands in for a Python with PyTorch: copies the checkpoint for the -c step,
# and writes a fixture model for the upstream export script.
import shutil, sys
from pathlib import Path
arguments = sys.argv[1:]
if arguments[0] == "-c":
    shutil.copyfile(arguments[2], arguments[3])
    sys.exit(0)
options = dict(zip(arguments[1::2], arguments[2::2]))
save = Path(options["--save_path"])
save.mkdir(parents=True, exist_ok=True)
Path(save / (options["--onnx_name"] + ".onnx")).write_bytes(Path("{fixture}").read_bytes())
(save / "arguments.txt").write_text(" ".join(arguments))
sys.exit({status})
"""


class PrepareModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self._temporary.name)
        self.output = self.directory / "out"

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def run_tool(self, *arguments: str) -> tuple[int, str]:
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors), contextlib.redirect_stdout(io.StringIO()):
            status = prepare_model.main(["--output", str(self.output), *arguments])
        return status, errors.getvalue()

    def served(self, name: str, data: bytes) -> tuple[str, str]:
        path = self.directory / name
        path.write_bytes(data)
        return path.as_uri(), hashlib.sha256(data).hexdigest()

    # The model's inputs and output.

    def test_reads_and_accepts_an_export(self) -> None:
        path = self.directory / "model.onnx"
        path.write_bytes(onnx_model())
        info = prepare_model.read_model_info(path)
        self.assertEqual(list(info.inputs), ["left_image", "right_image"])
        self.assertEqual(info.inputs["left_image"], ["batch_size", 3, 480, 736])
        self.assertEqual(info.outputs, {"disparity": ["batch_size", 1, "h", "w"]})
        prepare_model.check_model(info, 480, 736)
        path.write_bytes(onnx_model(batch=1))
        prepare_model.check_model(prepare_model.read_model_info(path), 480, 736)

    def test_rejects_other_models(self) -> None:
        cases = {
            "size": (onnx_model(height=448), "not \\[1, 3, 480, 736\\]"),
            "batch": (onnx_model(batch=2), "not \\[1, 3, 480, 736\\]"),
            "inputs": (onnx_model(inputs=("images",)), "not a single-file"),
            "order": (onnx_model(inputs=("right_image", "left_image")), "not a single-file"),
            "output": (onnx_model(output="output0"), "not one 4-D disparity"),
        }
        for name, (data, message) in cases.items():
            with self.subTest(name):
                path = self.directory / f"{name}.onnx"
                path.write_bytes(data)
                with self.assertRaisesRegex(prepare_model.PreparationError, message):
                    prepare_model.check_model(prepare_model.read_model_info(path), 480, 736)
        path = self.directory / "garbage.onnx"
        path.write_bytes(b"\xff\xff\xff")
        with self.assertRaises(prepare_model.PreparationError):
            prepare_model.read_model_info(path)

    # Fetching.

    def test_fetches_the_pinned_model(self) -> None:
        served = self.served("served.onnx", onnx_model())
        self.output.mkdir()
        stale = self.output / "c_fast_foundationstereo_480x736.onnx_b1_fp16.engine.RTX"
        stale.write_bytes(b"old")
        with mock.patch.object(prepare_model, "FETCHED_ONNX", served):
            status, errors = self.run_tool()
            self.assertEqual(status, 0, errors)
            onnx = self.output / "c_fast_foundationstereo_480x736.onnx"
            self.assertEqual(onnx.read_bytes(), onnx_model())
            self.assertFalse(stale.exists())
            # Present and pinned: kept.
            self.assertEqual(self.run_tool()[0], 0)
            # Present and not pinned: refused unless forced.
            onnx.write_bytes(b"other")
            status, errors = self.run_tool()
            self.assertEqual(status, 1)
            self.assertIn("is not the pinned model", errors)
            self.assertEqual(self.run_tool("--force")[0], 0)
            self.assertEqual(onnx.read_bytes(), onnx_model())

    def test_refuses_a_download_with_another_digest(self) -> None:
        url, _ = self.served("served.onnx", onnx_model())
        with mock.patch.object(prepare_model, "FETCHED_ONNX", (url, "0" * 64)):
            status, errors = self.run_tool()
        self.assertEqual(status, 1)
        self.assertIn("does not match the pinned", errors)
        self.assertEqual([p.name for p in self.output.iterdir() if not p.name.startswith(".")], [])

    def test_refuses_a_download_of_another_model(self) -> None:
        served = self.served("served.onnx", onnx_model(height=448))
        with mock.patch.object(prepare_model, "FETCHED_ONNX", served):
            status, errors = self.run_tool()
        self.assertEqual(status, 1)
        self.assertIn("not [1, 3, 480, 736]", errors)
        self.assertFalse((self.output / "c_fast_foundationstereo_480x736.onnx").exists())

    # Exporting.

    def fake_python(self, model: bytes, status: int = 0) -> Path:
        fixture = self.directory / "fixture.onnx"
        fixture.write_bytes(model)
        python = self.directory / "fake-python"
        python.write_text(FAKE_PYTHON.format(python=sys.executable, fixture=fixture, status=status))
        python.chmod(python.stat().st_mode | stat.S_IXUSR)
        return python

    def upstream(self) -> Path:
        repo = self.directory / "upstream"
        (repo / "scripts").mkdir(parents=True)
        (repo / "scripts" / "make_single_onnx.py").write_text("")
        git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run([*git, "add", "."], check=True)
        subprocess.run([*git, "commit", "-q", "-m", "fixture"], check=True)
        return repo

    @unittest.skipUnless(shutil.which("git"), "needs git")
    def test_exports_a_checkpoint(self) -> None:
        python = self.fake_python(onnx_model(height=480, width=640))
        checkpoint = self.directory / "model.pth"
        checkpoint.write_bytes(b"weights")
        repo = self.upstream()
        arguments = ["--source", "export", "--python", str(python), "--repo", str(repo)]
        # The repository must be at the pinned commit.
        status, errors = self.run_tool(*arguments, "--checkpoint", str(checkpoint))
        self.assertEqual(status, 1)
        self.assertIn("not the pinned", errors)
        status, errors = self.run_tool(
            *arguments, "--any-commit", "--checkpoint", str(checkpoint), "--valid-iters", "4"
        )
        self.assertEqual(status, 0, errors)
        onnx = self.output / "fast_foundationstereo_480x640_it4.onnx"
        self.assertEqual(onnx.read_bytes(), onnx_model(height=480, width=640))
        # Without --checkpoint, the pinned one is downloaded.
        served = self.served("served.pth", b"pinned weights")
        with mock.patch.object(prepare_model, "CHECKPOINT", served):
            status, errors = self.run_tool(*arguments, "--any-commit")
        self.assertEqual(status, 0, errors)
        self.assertTrue((self.output / "c_fast_foundationstereo_480x640_it8.onnx").is_file())

    @unittest.skipUnless(shutil.which("git"), "needs git")
    def test_reports_a_failed_or_wrong_export(self) -> None:
        checkpoint = self.directory / "model.pth"
        checkpoint.write_bytes(b"weights")
        repo = self.upstream()
        common = ["--source", "export", "--repo", str(repo), "--any-commit"]
        common += ["--checkpoint", str(checkpoint)]
        status, errors = self.run_tool(
            *common, "--python", str(self.fake_python(onnx_model(height=480, width=640), 3))
        )
        self.assertEqual(status, 1)
        self.assertIn("the export failed with exit status 3", errors)
        status, errors = self.run_tool(
            *common, "--python", str(self.fake_python(onnx_model(height=480, width=736)))
        )
        self.assertEqual(status, 1)
        self.assertIn("not [1, 3, 480, 640]", errors)
        status, errors = self.run_tool(*common, "--python", "no-such-python-here")
        self.assertEqual(status, 1)
        self.assertIn("was not found", errors)

    def test_rejects_unusable_arguments(self) -> None:
        for arguments, message in (
            (["--source", "export", "--height", "500"], "--height must be a multiple of 32"),
            (["--source", "export", "--width", "0"], "--width must be a multiple of 32"),
            (["--source", "export", "--valid-iters", "0"], "--valid-iters must be 1..32"),
            (["--source", "export", "--max-disp", "8"], "--max-disp must be 16..1024"),
            (["--checkpoint", "x.pth"], "apply to --source export"),
        ):
            with self.subTest(arguments):
                status, errors = self.run_tool(*arguments)
                self.assertEqual(status, 1)
                self.assertIn(message, errors)


if __name__ == "__main__":
    unittest.main()
