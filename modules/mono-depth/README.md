# holoscan-mono-depth

A Holoscan SDK 5 Module for monocular depth estimation on the GPU. `MonoDepthOp` takes an image
and emits its depth at the image's own size, as a GPU image: relative inverse depth always, metric
depth when it can, and a colorized depth for HoloViz on request. Inside, a CUDA kernel resizes the
image into the model's fixed input and normalizes it, the SDK's TensorRT inference library
(HoloInfer) runs the model, and a second kernel resamples the model's output back onto the image.
It is made for [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) Small. The
[Mono Depth demo](../../applications/mono_depth/) runs it end to end.

> **The depth is relative, not metric.** Depth Anything V2's relative models give *relative
> inverse depth*: larger is nearer, but in units of an unknown scale and shift, which change from
> image to image. It ranks and shapes the scene; it does not measure it. To get meters, use a
> metric checkpoint, or give the operator a scale and shift fitted to your camera and scenes
> against a reference (a depth or stereo camera, known distances), which then hold only there. See
> [Metric depth](#metric-depth).
>
> **The model is not part of this module.** [`tools/prepare_model.py`](tools/prepare_model.py)
> downloads a pinned checkpoint and exports it to ONNX with PyTorch and Hugging Face Transformers,
> which you install separately. The default, Depth Anything V2 **Small**, is **Apache-2.0**. The
> tool can also export **Base** and **Large**, which are **CC-BY-NC-4.0: non-commercial use
> only**, and only when told so. See [NOTICE](NOTICE). The tests need no model: they use a
> stand-in that the module writes itself.

| | |
| --- | --- |
| CMake package | `find_package(holoscan-mono-depth)` |
| Target | `holoscan::mono_depth` |
| Headers | `<holoscan/mono_depth/...>` |
| C++ namespace | `holoscan::mono_depth` |
| Debian package | `holoscan-mono-depth` |
| Model tool | `tools/prepare_model.py`, installed as `share/holoscan-mono-depth/prepare_model.py` |
| Platforms | x86_64; aarch64 (Jetson Thor and Orin, JetPack 7.2) |

## Operator

### `MonoDepthOp`

| Port | Direction | Type |
| --- | --- | --- |
| `image` | in | `holoscan::schema::ImageT`: RGB8, BGR8, RGBA8 or BGRA8, uint8 `[H, W, C]`, pixels packed within each row; GPU memory on device 0, or host or pinned memory |
| `depth` | out | `holoscan::schema::ImageT`: MONO32F, float32 `[H, W]`, inverse depth: relative for a relative model, `1 / meters` for a metric one, 0 where there is no valid depth; GPU memory |
| `metric_depth` | out | MONO32F, float32 `[H, W]`, meters, 0 where there is no valid depth; GPU memory. Declared only for a metric model or with `metric_from_relative` |
| `colorized` | out | RGBA8 `[H, W, 4]`: the inverse depth in the Turbo colormap, far dark blue to near dark red, opaque; GPU memory. Declared only with `colorize` |

Connect every output the configuration declares. The outputs carry the image's capture time and
frame id, and go out in the order of the table: should one fail to go out, those before it have
been delivered alone and the frame fails. Pair them by capture time, with an `OnAligned`
contract. `metric_depth` feeds holoscan-point-cloud's `DepthToPointCloudOp` as it is. Images may
change size from frame to frame, up to `max_width` x `max_height`; images in host memory are copied
to the GPU through a staging buffer, which the operator says once. The frames share the model's
buffers, and the runtime may give each one another CUDA stream, so each frame's stream waits on a
CUDA event that the previous frame recorded.

| `MonoDepthConfig` | Default | Meaning |
| --- | --- | --- |
| `model` | none | An `.onnx` export, or a TensorRT engine built for this GPU |
| `input_width`, `input_height` | `518`, `518` | The model's fixed input size; it must have been exported for it |
| `resize_mode` | `kLetterbox` | `kLetterbox` keeps the aspect ratio and pads with the mean color; `kStretch` fills the input |
| `normalization` | ImageNet's | Per RGB channel, `(value / 255 - mean) / std` |
| `model_output` | `kRelativeInverseDepth` | Or `kMetricDepth`, for a model that gives meters |
| `metric_from_relative` | none | `{scale, shift}`: meters from a relative model as `1 / z = scale * d + shift` |
| `colorize` | `false` | Declare the `colorized` output |
| `color_range` | none | `{low, high}`: the inverse depths at the colormap's ends; each frame's smallest and largest when unset |
| `max_width`, `max_height` | `1920`, `1080` | The largest image, which sizes the output buffers |
| `fp16` | `true` | Build the engine with FP16 kernels allowed |
| `cuda_graphs` | `true` | Let HoloInfer replay the inference as a CUDA graph |
| `build_workspace_bytes` | 1 GiB | TensorRT's workspace limit while it builds the engine |

Given an ONNX file, HoloInfer builds a TensorRT engine and caches it beside the file, named after
the GPU, the TensorRT version and the settings (FP16 or FP32 among them), so later runs load it in
seconds. Building Depth Anything V2 Small at 518x518 takes about 15 s on an RTX 5090 and several
minutes on a Jetson Orin. The cache is not keyed on the file's contents: after exporting a model
again to the same name, delete its `<name>.*.engine.*` files (`prepare_model.py` does). `start()`
fails, naming the problem, when the model does not take `[1, 3, input_height, input_width]` or give
one float32 `[1, H', W']` or `[1, 1, H', W']` output.

**Call `holoscan::mono_depth::prepare_engine(config)` before running the graph.** It builds and
caches the engine of an `.onnx` model. Some TensorRT releases (10.13 among them) change the SIGINT
and SIGTERM handlers while they build an engine, and the SDK's managed signal handling accepts that
only before a run starts. `start()` then loads the cached engine; without `prepare_engine()` it
builds the engine itself. TensorRT also starts threads of its own, after which the SDK no longer
takes over SIGINT and SIGTERM by itself (`RUNTIME_MANAGED_SIGNAL_DOMAIN_REQUIRED`): create the
process's `holoscan::ManagedShutdownSignals` at the start of `main()`, while it is the only thread,
and keep it until the run ends, as below.

The operator uses HoloInfer directly rather than the SDK's `InferenceOp`, which emits its outputs
with its own activation time and no frame id: the resampling needs each image's fit into the
input, and every output must keep the image's capture time.

## What it computes

The model takes a fixed `Wi x Hi` input. With `kLetterbox`, an image of `W x H` pixels is scaled by
`r = min(Wi / W, Hi / H)` to `round(W * r) x round(H * r)` (rounded half to even), centered, and
the rest is padded with the mean color, which is 0 after normalization; where the padding is odd,
the extra pixel goes on the right or the bottom. With `kStretch` it is scaled to `Wi x Hi`. Each
input pixel is a bilinear sample of the image with pixel centers aligned and edges clamped, as
OpenCV's `INTER_LINEAR` computes it, in RGB order, divided by 255 and normalized; alpha is
ignored. The input is float32 `[1, 3, Hi, Wi]` planes. (Depth Anything's own preprocessing resizes
with bicubic sampling to a variable size; a fixed size suits a TensorRT engine, and bilinear
sampling a real-time path. [Accuracy](#accuracy) has what the model achieves through it.)

The model's output, `Wo x Ho` (usually `Wi x Hi`), covers the input proportionally. Each image
pixel `(x, y)` reads it bilinearly at the output point under the image pixel's center, clamped to
the output pixels that cover the image, never the padding. The relative output is the depth `d` as
it is; a metric model's output `z` gives `1 / z`. Values that are not finite, and reciprocals of
values that are not positive, give 0.

The colorization maps each inverse depth to the nearest of the 256 Turbo entries between the
range's ends, so that near is warm and far is cool; the per-frame range is computed on the GPU, and
the colors are never copied to the host.

The per-pixel arithmetic is in [`pixel_math.hpp`](include/holoscan/mono_depth/pixel_math.hpp),
shared by the kernels and the host references in
[`reference.hpp`](include/holoscan/mono_depth/reference.hpp). The kernels are compiled without
fused multiply-adds, so the tests compare them with the references bit for bit.

## Metric depth

There are two ways to get meters, both of which enable the `metric_depth` output:

- **A metric checkpoint**, `model_output = kMetricDepth`: `prepare_model.py --model
  metric-indoor-small` (trained on Hypersim, up to 20 m) or `metric-outdoor-small` (Virtual KITTI 2,
  up to 80 m). They give meters directly, as well as their training scenes resemble yours.
- **A scale and shift for a relative model**, `metric_from_relative = {scale, shift}`, with
  `1 / z = scale * d + shift`. Fit them by least squares between `d` and `1 / z` at pixels of known
  depth `z`; the module's real-model test does exactly that against a RealSense's depth and prints
  the fit. They hold only for scenes like those they were fitted on: a relative model's scale and
  shift drift from image to image.

## Use it in your graph

```cmake
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core sensor_io)
find_package(holoscan-mono-depth 1.0 REQUIRED CONFIG)
target_link_libraries(my_app PRIVATE holoscan::mono_depth)
```

```cpp
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/mono_depth/mono_depth_op.hpp>

namespace md = holoscan::mono_depth;

// First thing in main(), while it is the only thread.
auto signals = holoscan::ManagedShutdownSignals::create();

const md::MonoDepthConfig config{.model = "/data/depth_anything_v2_small_518x518.onnx",
                                 .colorize = true};
auto depth = graph.op<md::MonoDepthOp>("depth", config);
graph.add_flow(camera->image, depth->image);           // Output<holoscan::schema::ImageT>
graph.add_flow(depth->depth, consumer->depth);         // Input<holoscan::schema::ImageT>
graph.add_flow(depth->colorized, holoviz->image_layer_input(1));

// The outputs are in GPU memory: bind their device when compiling.
holoscan::CompileOptions options;
for (const char* port : {"depth", "colorized"}) {
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "depth", .output_port = port, .device = holoscan::DeviceId{0}});
}
const auto plan = holoscan::compile(graph, std::move(options));
md::prepare_engine(config);   // Before the run: see above.
const int status = holoscan::run(plan);   // Adopts `signals`.
```

### Lossless or real time

`add_flow()` without options gives a blocking edge: every image's depth is estimated and
delivered, in order, at the queue depth the input declares (two). A blocking edge waits only
briefly for room, though. When a consumer is still busy with earlier frames, `emit()` returns
`kBackpressured` with nothing delivered, and the operator cannot send that frame's outputs again,
so the run fails. A lossless graph therefore paces its source to what its consumers keep up with,
as the demo's test modes do with one frame in flight.

For a live source, such as a camera, connect the operator's input and every output as a depth-1
drop-oldest edge instead. A full queue then drops its oldest sample rather than refusing the new
one: the operator never waits for, and is never refused by, a slower consumer, a camera faster than
the operator makes it take the newest frame, and a slow consumer finds the newest depth waiting
when it is done.

```cpp
const holoscan::ConnectionOptions newest{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                         .queue_depth = 1U};
graph.add_flow(camera->image, depth->image, newest);
graph.add_flow(depth->depth, consumer->depth, newest);
```

A consumer that pairs the depth with its image by capture time takes the image over a deeper
drop-oldest edge, so that the image is still queued when its depth arrives; the demo keeps four
frames, and shows the HoloViz wiring.

The kernels need no Holoscan graph: `<holoscan/mono_depth/kernels.hpp>` has `launch_preprocess()`,
`launch_postprocess()`, `launch_depth_range()` and `launch_colorize()`, and
`<holoscan/mono_depth/image_fit.hpp>` the geometry, which needs no CUDA either. For tests,
`<holoscan/mono_depth/stub_model.hpp>` writes the stand-in model, an ONNX network with the tensors
of a Depth Anything export whose output is a fixed weighted sum of each pixel's normalized color,
and `<holoscan/mono_depth/synthetic_scene.hpp>` renders a scene with known depth and checks a
predicted depth's ordering against it.

## Prepare the model

`prepare_model.py` downloads a checkpoint's `config.json` and `model.safetensors` from Hugging
Face at a pinned revision, checks both against their pinned SHA-256 digests, exports the model to
ONNX at a fixed input size with PyTorch and Transformers (opset 17, FP32), checks the export's
input and output, and writes `depth_anything_v2_<model>_<W>x<H>.onnx`. It needs only the standard
library itself: the export runs in the interpreter given by `--python`, which must have `torch`,
`transformers` and `safetensors`. Install them in their own virtual environment and review what
they pull in; the CPU build of PyTorch is enough. Tested with `torch` 2.15.0 and `transformers`
5.22.1:

```bash
python3 -m venv ~/depth-export-venv
~/depth-export-venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
~/depth-export-venv/bin/pip install transformers safetensors
python3 modules/mono-depth/tools/prepare_model.py --output data/mono_depth \
  --python ~/depth-export-venv/bin/python
```

| `--model` | Checkpoint | License | Output |
| --- | --- | --- | --- |
| `small` (default) | `depth-anything/Depth-Anything-V2-Small-hf`, 24.8 M parameters | Apache-2.0 | Relative inverse depth |
| `base` | `depth-anything/Depth-Anything-V2-Base-hf`, 97.5 M | **CC-BY-NC-4.0, non-commercial only**; needs `--accept-non-commercial-license` | Relative inverse depth |
| `large` | `depth-anything/Depth-Anything-V2-Large-hf`, 335.3 M | **CC-BY-NC-4.0, non-commercial only**; needs `--accept-non-commercial-license` | Relative inverse depth |
| `metric-indoor-small` | `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` | No tag on this conversion; the authors' original is Apache-2.0 | Meters, up to 20 |
| `metric-outdoor-small` | `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf` | No tag on this conversion; the authors' original is Apache-2.0 | Meters, up to 80 |

`--width` and `--height` set the input size, multiples of 14 (Depth Anything's patch size); 518x518
is what the models were trained at. A smaller or wider input is faster: for 16:9 frames,
`--width 518 --height 294` keeps the aspect ratio with 43 % fewer pixels. `--cache DIR` keeps the
downloads for later exports, and `--force` exports again over an existing model. From the
installed module, run `<prefix>/share/holoscan-mono-depth/prepare_model.py`; CMake projects find
its path in `holoscan-mono-depth_PREPARE_MODEL`.

## Security

Treat a model file as trusted input. The operator hands the file it is given to TensorRT, which
parses an ONNX model or deserializes an engine; load only models and engines that you exported or
built yourself, or otherwise trust. The export loads the checkpoint with Transformers from
`safetensors` files only, whose digests are pinned.

## Requirements

- Holoscan SDK 5.0 (Core, Sensor I/O and Inference, which brings HoloInfer and TensorRT 10;
  Testing for the tests), CUDA 13, CMake 4.0 or later and a C++20 compiler.
- For the model only: Python 3.12 or later and, in a separate environment, `torch`,
  `transformers` and `safetensors`.

## Build, test and install

From the repository root, in the repository's CI image or with the requirements above:

```bash
cmake -S modules/mono-depth -B build/mono-depth -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build/mono-depth
ctest --test-dir build/mono-depth --output-on-failure
cmake --install build/mono-depth --prefix "$PWD/install"
```

| CMake option | Default | |
| --- | --- | --- |
| `BUILD_TESTING` | `OFF` | Build the tests, when the module is the top-level project |
| `HOLOSCAN_MONO_DEPTH_BUILD_TESTING` | `OFF` | Build the tests when another project includes the module (`add_subdirectory()`, `FetchContent`); its `BUILD_TESTING` and `include(CTest)` do not |
| `HOLOSCAN_MONO_DEPTH_MODEL` | `data/mono_depth/depth_anything_v2_small_518x518.onnx` | The exported model the real-model test uses |
| `CMAKE_CUDA_ARCHITECTURES` | `native` (`all-major` for the package) | GPUs to compile for |

| Test | Needs | Checks |
| --- | --- | --- |
| `mono_depth_kernels` | GPU | The kernels against the host references, bit for bit: the preprocessing of random images in every encoding, letterboxed and stretched, wide and tall, with padded rows, down- and upscaled; the postprocessing of outputs at, below and above the input's size into inverse and metric depth; the range with non-finite values; the colorization; and what the values mean: the normalization, the padding, the channel order, the reciprocals and the colormap's ends |
| `mono_depth_operators` | GPU | `MonoDepthOp` through the SDK's operator test harness, with the stub model and TensorRT, every output value against the host reference of the whole pipeline: every encoding, from GPU, host and pinned memory, wide, tall and small images, letterboxed and stretched into a wide input, changing image sizes, the colorized output with a per-frame and a fixed range, metric depth from an affine and from a metric model, an FP16 engine, provenance on every output, unusable images, settings and models |
| `mono_depth_real_model` | GPU, the exported model; skipped without it | Depth Anything V2 Small on the synthetic scene, FP16 and FP32, letterboxed and stretched: Spearman's rank correlation with the true depth at least 0.85, and the near panel, the sphere and the far wall in that order. With `HOLOSCAN_TEST_DATA`, also real RealSense D585 frames against the camera's depth (see [Accuracy](#accuracy)) |
| `mono_depth_host` | Nothing | The image fit for wide, tall, thin and stretched images and outputs of other sizes; the references against direct evaluations; the stub model file; the synthetic scene's geometry and truth; Spearman's correlation with ties; the ordering check; the colormap |
| `mono_depth_prepare_model` | Python | `prepare_model.py` without PyTorch or the network: pinned revisions and digests, a wrong digest, the cache, the non-commercial licenses, a failed or missing exporter, an export of another size, and reuse of prepared files |

`mono_depth_host` and `mono_depth_prepare_model` are also a project of their own in
[`tests/host/`](tests/host/), which the hosted CI job builds and runs.
[`tests/installed_consumer/`](tests/installed_consumer/) is a separate project that uses the
installed package: a small graph that runs `MonoDepthOp` with the stub model on synthetic frames and
checks every frame's depth against the reference, and a run of the installed `prepare_model.py`.

## Debian package

Inside the repository's CI image, with the module's folder as the Holoscan CLI project root:

```bash
holoscan --project-root modules/mono-depth package holoscan-mono-depth --language cpp \
  --pkg-generator DEB --local
```

This builds for every major GPU architecture and writes `holoscan-mono-depth_1.0.0_<arch>.deb`. It
installs to `/usr` and depends on `holoscan (>= 5.0.0)`. It holds no model.

## Accuracy

Depth Anything V2 Small at 518x518, letterboxed, through `MonoDepthOp`:

| Input | Engine | Spearman's correlation with the true inverse depth |
| --- | --- | --- |
| The synthetic scene, 1280x720, 300 frames | FP16 | 0.9185 to 0.9200, regions always in order |
| The same | FP32 | 0.9185 to 0.9200 |
| The same, stretched | FP16 | 0.916 to 0.919 |
| RealSense D585 RGB, 1280x720, office ceiling, 4 frames | FP16 | 0.924 to 0.935 |

On the RealSense frames, the camera's depth is reprojected into the RGB camera, and one scale and
shift per frame (least squares in inverse depth) maps the prediction onto it. What remains is an
absolute relative error of 0.24 to 0.30 and 40 to 60 % of pixels within 25 % of the camera's depth:
the ordering is right, but a relative model's depth is not a measurement, and this camera's very
wide RGB lens (about 125 degrees across) is far from what the model was trained on. The FP16 and
FP32 engines give the same correlations to four places, so FP16 is the default.

## Performance

Depth Anything V2 Small, 518x518 input, measured with SDK 5.0.0, CUDA 13.2 and TensorRT 10.16 in
the repository's CI image:

| | RTX 5090 (x86_64) | Jetson AGX Orin (JetPack 7.2) |
| --- | --- | --- |
| TensorRT inference alone, FP16 (`trtexec`, CUDA graph, median) | 0.87 ms | 8.9 ms |
| The same, FP32 | 2.75 ms | 40.4 ms |
| Capture to depth on the host, 1280x720 frames, FP16, mean (median, p99) | 1.4 ms (1.4, 2.5) | 13.2 ms (10.9, 48) |
| The same, FP32 | 3.3 ms (3.2, 3.6) | 45.9 ms (43.0, 77) |
| The same, FP16, 640x480 frames | 1.1 ms | 14.5 ms (12.9, 39) |
| TensorRT execution memory, FP16 (FP32) | 38 MiB (106 MiB) | 41 MiB (150 MiB) |
| The demo's whole process on the GPU (`nvidia-smi`), FP16, 1280x720 | 772 MiB | not measured |

Both GPUs were shared with other work while measured: an inference server on the RTX 5090, and
another container's GPU tests on the Orin, which the Orin's tail latencies show. The process memory
includes the CUDA context and the TensorRT libraries; the engine's weights are about 50 MiB in FP16.
On Jetson Thor, CI runs the stub model only, as no model is downloaded there; the real model's
latency on Thor is not measured. For 16:9 frames, a 518x294 export keeps the same correlation on the
synthetic scene (0.920) and takes 1.1 ms instead of 1.4 ms on the RTX 5090.

## Licenses

Apache-2.0. The models are not covered by it; see the notes at the top and [NOTICE](NOTICE). The
Turbo colormap's table is Google's, Apache-2.0. HoloInfer and the rest of the SDK are Apache-2.0;
TensorRT and CUDA are under NVIDIA's own terms.
