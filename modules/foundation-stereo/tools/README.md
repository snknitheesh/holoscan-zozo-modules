# prepare_model.py

Writes a single-file Fast-FoundationStereo ONNX model for `FoundationStereoOp`. Installed as
`share/holoscan-foundation-stereo/prepare_model.py`; CMake's `holoscan-foundation-stereo_PREPARE_MODEL`
names it.

> **The models are NVIDIA's, under NVIDIA's licenses, not this repository's Apache-2.0.** The
> default model is under the NVIDIA Open Model Agreement; the upstream code and its research
> checkpoints are under a non-commercial, research-only license. See [NOTICE](../NOTICE). The script
> never bundles or redistributes weights: it downloads or exports them on your machine.

Every model it writes takes `left_image` and `right_image`, float32 `[1, 3, H, W]` RGB normalized
with ImageNet's mean and deviation, and gives `disparity`, float32 `[1, 1, H, W]` in the model's
pixels. The precision is chosen when TensorRT builds the engine, not here: `FoundationStereoOp`
builds FP16 by default (`fp16 = false` for FP32). FP8 and INT8 quantization are out of scope.

## Fetch (default): C-Fast-FoundationStereo, 480x736

```bash
python3 modules/foundation-stereo/tools/prepare_model.py --output data/foundation_stereo
```

Downloads NVIDIA's own ONNX export of C-Fast-FoundationStereo for 480x736 inputs from Hugging Face
([nvidia/c-fast-foundationstereo](https://huggingface.co/nvidia/c-fast-foundationstereo) at revision
`53a84e28a368f37e699776250215fcb450d6a109`, 103 MB), checks its SHA-256
(`794647dc718c9564d5ccb56d15527b70a9267e8ddfbd5ecb1e784b323fbd60ed`) and its inputs and output, and
writes `data/foundation_stereo/c_fast_foundationstereo_480x736.onnx`, where the demo and the tests
look for it. It needs only Python 3.12 and the network: no PyTorch. Its batch dimension is
symbolic; `FoundationStereoOp` builds the engine for one pair.

## Export: another size, iteration count or checkpoint

The upstream repository's `scripts/make_single_onnx.py` exports a checkpoint for any input size that
is a multiple of 32. It needs PyTorch with CUDA and a GPU, so do it in a virtual environment (or a
container) of its own, never in the CI image. With
[uv](https://docs.astral.sh/uv/) (plain `python3 -m venv` and `pip` work as well):

```bash
uv venv -p 3.12 ~/ffs-export
# PyTorch 2.8 with CUDA 12.8 runs on Ampere through Blackwell; pick the build for your GPU.
uv pip install -p ~/ffs-export torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install -p ~/ffs-export timm einops omegaconf scipy numpy pyyaml onnx onnxscript \
  opencv-python-headless scikit-image imageio
python3 modules/foundation-stereo/tools/prepare_model.py --output data/foundation_stereo \
  --source export --python ~/ffs-export/bin/python --height 480 --width 640
```

The script clones [NVlabs/Fast-FoundationStereo](https://github.com/NVlabs/Fast-FoundationStereo)
at the pinned commit `476f4249561f7c79ca707326954f9255643412a6` into a temporary folder (or uses
`--repo DIR`, which must be at that commit unless `--any-commit`), downloads the pinned
C-Fast-FoundationStereo checkpoint (`model_best_bp2_serialize.pth`, SHA-256
`7aee85948373da62b0503c2542507129a3e7cab9d97d10e6790d89512a7db214`) unless `--checkpoint` names
another, and runs the export with `--valid-iters` (8) refinement iterations and `--max-disp` (192)
pixels of search. It writes `c_fast_foundationstereo_<H>x<W>_it<N>.onnx`, or
`fast_foundationstereo_...` for a `--checkpoint`, or `--name`.

The C-Fast-FoundationStereo checkpoint predates the `normalize` setting the pinned upstream code
reads; the script adds it with the value upstream assumes when it is missing (`True`), on a copy.
Exporting it at 480x736 gives the same results as the fetched ONNX file on the recorded RealSense
clip.

**Research checkpoints** (`23-36-37`, `20-26-39`, `20-30-48`) are linked from the upstream README,
on Google Drive, which offers no stable download URL; download one yourself and pass its
`model_best_bp2_serialize.pth` with `--checkpoint`. The script prints its SHA-256 for your records.
These checkpoints, and the upstream code that every export runs, are licensed for non-commercial
research use only.

The export imports and runs the upstream code and unpickles the checkpoint, which can run arbitrary
code: use checkpoints from sources you trust, in an environment of their own.

## Engines

`FoundationStereoOp` (and `prepare_engine()`) builds the TensorRT engine of an `.onnx` model
beside it on first use, named after the model, the GPU, the TensorRT version and the precision,
and loads it from there afterwards. Building the 480x736 model takes about 2 minutes on an
RTX 5090 and about 46 minutes on a Jetson AGX Orin. Preparing a model again under the same name deletes the engines
cached for the old one.

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--output DIR` | required | Where the model goes |
| `--source fetch\|export` | `fetch` | Download the pinned ONNX model, or export a checkpoint |
| `--force` | off | Replace an existing model |
| `--python PATH` | `python3` | Export: a Python with PyTorch and the upstream requirements |
| `--checkpoint FILE` | the pinned C-Fast-FoundationStereo one | Export: another checkpoint |
| `--repo DIR`, `--any-commit` | a fresh clone | Export: an upstream checkout to use |
| `--height`, `--width` | `480`, `640` | Export: input size, multiples of 32 |
| `--valid-iters` | `8` | Export: refinement iterations; fewer is faster and less accurate |
| `--max-disp` | `192` | Export: largest disparity searched, in pixels at the model's size |
| `--name` | from the size | Export: file name, without `.onnx` |

The tests in [`tests/host/test_prepare_model.py`](../tests/host/test_prepare_model.py) cover both
paths without PyTorch or the network.
