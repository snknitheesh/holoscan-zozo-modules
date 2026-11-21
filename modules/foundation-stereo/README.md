# holoscan-foundation-stereo

A Holoscan SDK 5 Module for learned stereo depth on the GPU. `FoundationStereoOp` takes a rectified
left and right image and the pair's stereo camera info, and emits the left image's disparity, and
optionally its depth in meters with the depth's camera info, as GPU images, estimated by NVIDIA's
[Fast-FoundationStereo](https://github.com/NVlabs/Fast-FoundationStereo) through TensorRT. Inside, a
CUDA kernel resizes each image to the model's fixed input size and normalizes it, the SDK's TensorRT
inference library (HoloInfer) runs the model, and a second kernel resizes the disparity back to the
image and rescales it to the image's pixels. The camera info is
[holoscan-messages](../messages/)' `StereoCameraInfo`, taken as every module of this collection
takes it, and the depth with its `CameraInfo` goes straight into
[holoscan-point-cloud](../point-cloud/)'s `DepthToPointCloudOp`. It has the ports of the stereo
disparity module's `StereoDisparityOp` but one (`left`, `right`, `camera_info`; `disparity`, `depth`,
`depth_camera_info`, MONO32F, 0 where invalid), so it can take that operator's place in a graph that
does not use its `confidence` output or a downscale. The
[Foundation Stereo demo](../../applications/foundation_stereo/) runs it end to end.

> **The models are NVIDIA's, under NVIDIA's licenses, and are not part of this module.**
> [`tools/prepare_model.py`](tools/prepare_model.py) downloads or exports them on your machine; see
> [its README](tools/README.md) and [NOTICE](NOTICE).
>
> - The default model, **C-Fast-FoundationStereo** (NVIDIA's ONNX export from
>   [Hugging Face](https://huggingface.co/nvidia/c-fast-foundationstereo)), is governed by the
>   [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/),
>   which permits commercial use; its model card also calls it "for research and evaluation purposes
>   only".
> - The **Fast-FoundationStereo code and research checkpoints** on GitHub are under NVIDIA's
>   **non-commercial, research-only** license (the repository's `LICENSE.txt`: "the Work and any
>   derivative works thereof only may be used or intended for use non-commercially"). The script's
>   export path runs that code.
>
> This module is Apache-2.0 and contains no weights and no upstream code. The terms that apply to
> using and distributing a model, and anything built with it, are yours to review. The tests need no
> model: they use a stand-in that the module writes itself.

| | |
| --- | --- |
| CMake package | `find_package(holoscan-foundation-stereo)` |
| Target | `holoscan::foundation_stereo` |
| Headers | `<holoscan/foundation_stereo/...>` |
| C++ namespace | `holoscan::foundation_stereo` |
| Debian package | `holoscan-foundation-stereo` |
| Model tool | `tools/prepare_model.py`, installed as `share/holoscan-foundation-stereo/prepare_model.py` |
| Platforms | x86_64; aarch64 (Jetson Thor and Orin, JetPack 7.2) |

## `FoundationStereoOp`

| Port | Direction | Type |
| --- | --- | --- |
| `left`, `right` | in | `holoscan::schema::ImageT`: MONO8, RGB8, BGR8, RGBA8 or BGRA8, uint8 `[H, W]` or `[H, W, C]`, pixels packed within rows; GPU memory on device 0, or pinned or pageable host memory; both the same size, rectified |
| `camera_info` | in, mandatory | `holoscan::messages::schema::StereoCameraInfoT`: the pair's calibration, both cameras `kRectified` and of the images' size; it gives `fx`, the baseline and `doffs` |
| `disparity` | out | `holoscan::schema::ImageT`: MONO32F `[H, W]` on the GPU, the left image's disparity in its pixels (left column minus right column), 0 where invalid |
| `depth` (with `depth`) | out | `holoscan::schema::ImageT`: MONO32F `[H, W]` on the GPU, meters along the optical axis, `fx * baseline / (disparity + doffs)`, 0 where invalid |
| `depth_camera_info` (with `depth_camera_info`) | out | `holoscan::messages::schema::CameraInfoT`: the camera of `depth` and `disparity`, the rectified left camera at the images' size, with each frame's capture time, sent before the depth |

The left and right images pair by exact capture time; the outputs carry the left image's capture
time, frame id and header. The camera info comes as `messages::camera_info_contract()` takes it:
with `kLatest` (the default) the newest one received, from a `CameraInfoSourceOp` that reads the
pair's ROS camera_info files at its own rate, and frames before the first one are skipped; with
`kEveryFrame`, one with every pair, as the rectify module's `StereoRectifyOp` sends it. A camera
info that is not rectified, or not of the images' size, fails the run. The model gives no
confidence, so there is no confidence output. A disparity is invalid (0) where the model's is not positive, and, with
`invalidate_out_of_view`, where its match would lie left of the right image, which the right camera
cannot have seen (the model guesses there). Images in host memory are copied to the GPU through a
staging buffer, which is logged once. Images may change size from frame to frame, up to
`max_width x max_height`.

| `FoundationStereoConfig` | Default | Meaning |
| --- | --- | --- |
| `model` | none | A single-file Fast-FoundationStereo `.onnx` model, or a TensorRT engine built for this GPU |
| `fp16` | `true` | Build the engine with FP16 kernels allowed |
| `cuda_graphs` | `true` | Let HoloInfer replay the inference as a CUDA graph |
| `build_workspace_bytes` | 4 GiB | TensorRT's workspace limit while it builds the engine (1 GiB is too little for the 480x736 model on Jetson Orin) |
| `camera_info_input` | `kLatest`, 10 s | How `camera_info` is taken: `timing` (`kLatest` or `kEveryFrame`) and, for `kLatest`, the oldest camera info a frame takes (`max_age`) |
| `depth` | `false` | Declare the `depth` output |
| `depth_camera_info` | `false` | Declare the `depth_camera_info` output; needs `depth` |
| `invalidate_out_of_view` | `true` | Zero pixels matched left of the right image |
| `max_width`, `max_height` | 1920, 1080 | Largest image the outputs are sized for |
| `stats` | none | A `FoundationStereoStats` to count frames, GPU time per frame (the first frame, which also captures the CUDA graph, apart) and the drop in the GPU's free memory while the model loaded |

The settings are fixed when the operator is constructed: it has no Holoscan parameters to change
afterwards, and the module has no Python bindings.

The model's fixed input size is read from it. Given an ONNX file, HoloInfer builds a TensorRT engine
and caches it beside the file, named after the file's name, the GPU, the TensorRT version and the
precision, but not after its content; so the operator records the content's digest beside the model
(`<model>.digest`) and, when a model is replaced under the same name, deletes the engines cached for
the old one before HoloInfer looks. The first build of the 480x736 model takes about 2 minutes on an
RTX 5090 and about 46 minutes on a Jetson Orin; build it once ahead of time, for example with the
demo or `prepare_engine()`. An interrupt (Ctrl-C) during a build takes effect only once the build is
done. `start()` fails, naming the problem and the GPU's free memory, when the model cannot be loaded,
or does not take two `[1, 3, H, W]` inputs and give one `[1, 1, H, W]` float32 disparity.

**Call `holoscan::foundation_stereo::prepare_engine(config)` before running the graph.** It builds
and caches the engine of an `.onnx` model. Some TensorRT releases change the SIGINT and SIGTERM
handlers while they build an engine, and the SDK's managed signal handling accepts that only before
a run starts. TensorRT also starts threads of its own, after which the SDK no longer takes over
SIGINT and SIGTERM by itself: create the process's `holoscan::ManagedShutdownSignals` at the start
of `main()`, while it is the only thread, as below.

The operator uses HoloInfer directly rather than the SDK's `InferenceOp`, which emits its outputs
with its own activation time and no frame id: the outputs must keep the left image's capture time,
and the resize back needs the image's size.

## The model and its input size

The default is NVIDIA's **C-Fast-FoundationStereo** ONNX export for **480x736** inputs (rows x
columns): Fast-FoundationStereo is the real-time, distilled version of FoundationStereo (14.6 M
parameters), and this is the only checkpoint with a published ONNX/TensorRT path that needs no
PyTorch, and the only one under terms that permit commercial use. FoundationStereo itself is about
10 times slower and is not used.

Why 480x736: it is the size NVIDIA exported, it is the 640x480 class upstream benchmarks its
real-time numbers at, and its 1.53 aspect sits between 4:3 and 16:9, so 1280x720 and 848x480
camera frames and 640x480 frames all fit it without much stretch. Each image is resized to the
model's size with its aspect ratio changed, not letterboxed: a horizontal stretch keeps rows on
rows, so the pair stays rectified, and the disparity is multiplied back by `W / 736`. On the 720p
recording, a stretched 736x480 input agreed better with the camera than a letterboxed 736x414
one (within 5 % of the camera's depth: 0.75 against 0.58 to 0.66), and running the model at
1280x736 instead gained almost nothing (0.76). For another size, `prepare_model.py --source export`
exports any multiple of 32 (see its README).

The engine is built in FP16 by default; on the 720p recording FP16 gives the same accuracy as FP32
ONNX Runtime. FP8 and INT8 quantization are out of scope.

## What it computes

Each input image is resized bilinearly to the model's `W x H` with pixel centers at half-integer
coordinates (OpenCV's `INTER_LINEAR`), in RGB order (MONO8 repeated in all three channels, alpha
ignored), and normalized with ImageNet's mean `(123.675, 116.28, 103.53)` and deviation
`(58.395, 57.12, 57.375)` into float32 `[1, 3, H, W]` planes, as the upstream single-file export
expects. The model's disparity, in its own pixels, is resized bilinearly to the image's size the same
way and multiplied by `image width / W`; a non-positive value, or one whose match would lie left of
the right image, becomes 0. The depth is `depth_from_disparity()`: `fx * baseline / (d + doffs)`,
0 where the disparity is 0.

`<holoscan/foundation_stereo/kernels.hpp>` has the two kernels, `launch_preprocess()` and
`launch_postprocess()`, which need no graph. `<holoscan/foundation_stereo/reference.hpp>` computes
the same on the CPU, and `<holoscan/foundation_stereo/stub_model.hpp>` writes the stand-in model: an
ONNX network with the tensors of a real export whose output is a known function of its inputs, so a
test can check the whole chain, model included, against the CPU. `<holoscan/foundation_stereo/synthetic_scene.hpp>`
renders a rectified pair of textured planes at known depths, with their true disparity and their
camera info. These four headers are test support: they may change in a minor release.

## Accuracy, latency and memory

With the pinned 480x736 C-Fast-FoundationStereo model, an FP16 engine, SDK 5.0.0, CUDA 13.2 and
TensorRT 10.16, on an RTX 5090 (x86_64) and a Jetson Orin (JetPack 7.2, MAXN). The real model has
not been run on a Jetson Thor; there, as on every CI lane, the tests use the stub model.

**Synthetic scene** (textured planes at 1.5 to 8 m, so known disparity everywhere both cameras see),
mean end-point error over those pixels, from `foundation_stereo_real_model`:

| Image | RTX 5090 | Jetson Orin |
| --- | --- | --- |
| 640x480 RGB8 | 0.03 px mean, 0.02 px median, 0.2 % over 1 px | the same |
| 1280x720 MONO8 | 0.15 px mean, 0.015 px median, 0.9 % over 1 px | the same |

**The local stereo camera recordings** (`$HOLOSCAN_TEST_DATA/stereo-camera/`, 60 frames each), against
the camera's own depth: median relative error, share within 5 %, coverage of the camera's pixels.
The camera's depth is not ground truth: it is the camera's own semi-global matching of the same
infrared pair.

| Clip | RTX 5090 | Jetson Orin | VPI `cuda` (classic SGM), for comparison |
| --- | --- | --- | --- |
| 720p, projector on | 1.15 %, 0.776, 0.995 | 1.15 %, 0.776, 0.995 | 1.05 %, 0.897, 0.994 |
| 480p, projector on | 1.05 %, 0.776, 0.996 | 1.05 %, 0.775, 0.996 | 1.05 %, 0.903, 0.977 |
| 720p, projector off (camera depth on 1.8 % of the pixels; reported only) | valid 0.987, 6.65 % | valid 0.987, 6.75 % | valid 0.902, 6.65 % |
| 480p, projector off (camera depth on 2.1 %; reported only) | valid 0.987, 7.15 % | valid 0.987, 7.25 % | valid 0.932, 6.75 % |

The median agrees with the camera as well as classic SGM does, but fewer pixels are within 5 %: the
model and the camera disagree over a dark, nearly textureless area of the 720p clip, lit only by
sparse projector dots, where the model puts the surface at about 1 m and the camera (and SGM) at
about 3 m (90th percentile of the relative error: 0.48, against 0.05 for SGM). Which is right there
cannot be told from the recording. With the projector off, the model gives a disparity for 99 % of
the pixels where the camera gives depth for 2 %.

**Latency and memory** of the 480x736 model, per frame, from the demo on the clips and the synthetic
scene:

| | RTX 5090 (shared) | Jetson Orin |
| --- | --- | --- |
| GPU time, first resize to output written (mean) | 10.7 to 12 ms | 152 to 176 ms |
| The same without CUDA graphs (`cuda_graphs = false`) | 10.7 to 11.3 ms, no measurable change (10.8 to 10.9 ms with them, same runs) | not measured |
| TensorRT engine alone (`trtexec`, FP16) | 10.5 ms | 149 ms |
| Capture to disparity done on the GPU (median), lossless, frames every 33 ms (RTX 5090) or 200 ms (Orin) | 11 to 13 ms (not shared) | 152 to 175 ms (measured to the disparity on the host) |
| GPU memory for the model | about 0.78 GB (drop in free memory while loading: engine, activations and buffers) | 1.86 GB (39 MB engine, 1.82 GB execution context) |
| Building the engine, once | about 2 minutes | about 46 minutes |

On the RTX 5090 the memory is the drop in the device's free memory while the model loaded
(`FoundationStereoStats::model_bytes`), an approximation that any other process allocating on the
device at the same time distorts; on the Orin, whose GPU shares the system's memory with other
processes, that measure is not reliable, and the figure is TensorRT's own. The RTX 5090 was shared
with an inference server at the time. The Orin keeps up with about 6 frames a second: played in real
time at 30 frames a second (`--realtime true`), it processed 7 of 60 frames and dropped the rest, as
a live camera's frames would be. For comparison, VPI's classic matcher took 9.2 ms (RTX 5090) and
61.8 ms (Orin, `cuda`), or 46.2 ms on the Orin's OFA, per 720p frame; see the stereo disparity
module. On the Orin, the learned model costs about 2.5 times the classic matcher's GPU time for
similar agreement with the camera, and much better coverage where the camera's projector is off.

## Use it in your graph

```cmake
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core sensor_io)
find_package(holoscan-messages 1.0 REQUIRED CONFIG)
find_package(holoscan-foundation-stereo 1.0 REQUIRED CONFIG)
target_link_libraries(my_app PRIVATE holoscan::foundation_stereo holoscan::messages)
```

A live camera, whose frames may be dropped: every frame edge keeps only the newest frame
(`kDropOldest`, depth 1), so that no operator waits for a slower one, on whichever worker each runs.
The calibration comes from the rectified pair's ROS camera_info files.

```cpp
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/foundation_stereo/foundation_stereo_op.hpp>
#include <holoscan/messages/camera_info_input.hpp>
#include <holoscan/messages/camera_info_source_op.hpp>

namespace fs = holoscan::foundation_stereo;
namespace hm = holoscan::messages;

// First thing in main(), while it is the only thread: TensorRT starts threads of its own.
auto signals = holoscan::ManagedShutdownSignals::create();

auto camera_info = graph.op<hm::CameraInfoSourceOp>(
    "camera_info", hm::CameraInfoSourceConfig{.camera_info_path = "left.yaml",
                                              .right_camera_info_path = "right.yaml"});
const fs::FoundationStereoConfig config{
    .model = "/data/foundation_stereo/c_fast_foundationstereo_480x736.onnx",
    .depth = true,
    .depth_camera_info = true,
    .max_width = 1280,
    .max_height = 720};
auto stereo = graph.op<fs::FoundationStereoOp>("stereo", config);
const holoscan::ConnectionOptions live{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                       .queue_depth = 1U};
graph.add_flow(camera->left, stereo->left, live);  // Output<holoscan::schema::ImageT>
graph.add_flow(camera->right, stereo->right, live);
graph.add_flow(camera_info->stereo_camera_info, stereo->camera_info,
               hm::latest_camera_info_connection());
graph.add_flow(stereo->disparity, viewer->disparity, live);  // always declared: connect it
graph.add_flow(stereo->depth, cloud->depth, live);  // e.g. holoscan-point-cloud's DepthToPointCloudOp
graph.add_flow(stereo->depth_camera_info, cloud->camera_info, hm::latest_camera_info_connection());

// The image outputs are in GPU memory: bind their device when compiling.
holoscan::CompileOptions options;
for (const char* port : {"disparity", "depth"}) {
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "stereo", .output_port = port, .device = holoscan::DeviceId{0}});
}
const auto plan = holoscan::compile(graph, std::move(options));
fs::prepare_engine(config);              // Before the run: see above.
const int status = holoscan::run(plan);  // Adopts `signals`.
```

Every declared output must be connected: an emit into an unconnected output fails. For a pair that
a rectifier sends with its camera info on every frame, connect that output instead and set
`.camera_info_input = {.timing = hm::CameraInfoTiming::kEveryFrame}`, with the images' edge
options. On a live graph, `compute()` waits for the frame before last to finish before it queues
another, so at most two frames are on the GPU and a disparity is at most about two model latencies
old: the newest pair waits on the input's edge meanwhile.

A lossless chain, where every frame must arrive (a recording, a test), connects the frame edges
with `QueuePolicy::kBlocking` instead, and must stay on one worker, as the demo does:
`graph.partition("name").include_all()` runs the whole graph on one. A graph with a second source,
such as a `CameraInfoSourceOp`, or one that forks or joins, is split by the SDK's automatic
partitioning onto several workers. Across workers, the SDK 5.0 runs a producer even while its
consumer's lossless queue is full, and refuses what it sends: an emit fails with `kBackpressured`,
and an allocation with `kResourceExhausted` while every slot of the output's pool is still held. A
source can treat both as backpressure and send again on its next tick, but `FoundationStereoOp` has
taken its pair by then and returns the error, which fails the run. A lossless graph runs at the
model's speed, whatever its source's period: the source is refused and sends again, so on a Jetson
Orin a frame comes about every 160 ms.

## Requirements

- Holoscan SDK 5.0 (Core, Sensor I/O and Inference, which brings HoloInfer and TensorRT 10; Testing
  for the tests), CUDA 13, CMake 4.0 or later and a C++20 compiler.
- [holoscan-messages](../messages/) 1.0, installed or built in the same tree.
- For the model only: Python 3.12 and the network; to export another size, PyTorch in an environment
  of its own (see [tools/README.md](tools/README.md)).

## Build, test and install

From the repository root, in the repository's CI image or with the requirements above:

```bash
cmake -S modules/foundation-stereo -B build/foundation-stereo -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON -DCMAKE_PREFIX_PATH="$PWD/install"  # where holoscan-messages is installed
cmake --build build/foundation-stereo
ctest --test-dir build/foundation-stereo --output-on-failure
cmake --install build/foundation-stereo --prefix "$PWD/install"
```

| CMake option | Default | |
| --- | --- | --- |
| `BUILD_TESTING` | `OFF` | Build the tests, when the module is the top-level project |
| `HOLOSCAN_FOUNDATION_STEREO_BUILD_TESTING` | `OFF` | Build the tests when another project includes the module |
| `HOLOSCAN_FOUNDATION_STEREO_MODEL` | empty | The model for the real-model test, as `prepare_model.py` wrote it; empty skips the test |
| `CMAKE_CUDA_ARCHITECTURES` | `native` (`all-major` for the package) | GPUs to compile for |

| Test | Needs | Checks |
| --- | --- | --- |
| `foundation_stereo_kernels` | GPU | The two kernels against the CPU reference: every pixel format, down- and upscaling, padded rows; the disparity rescaled by the width ratio (a constant model disparity of 5 at 160 wide is 40 at 1280), its invalidation, and the depth exactly `fx * B / d` |
| `foundation_stereo_operator` | GPU | The operator through the SDK's operator test harness with the stub model and TensorRT: each frame against the CPU reference of the whole chain, for every format, from GPU, pinned, pageable and imported pinned memory, images larger and smaller than the model, FP32 and FP16 engines; the depth and its camera info; provenance; the camera info at its own rate and with every frame; a model replaced under the same name; unusable inputs, camera infos, models and settings |
| `foundation_stereo_real_model` | GPU, the model | The real model on the synthetic scene, RGB8 640x480 and MONO8 1280x720: valid share, mean and median end-point error, share over 1 px. Skipped (exit 77) without the model |
| `foundation_stereo_host` | Nothing | The stub model file and formula, the camera info rules, the depth's camera and the depth formula, the CPU references on hand-worked cases |
| `foundation_stereo_prepare_model` | Python | `prepare_model.py` without PyTorch or the network: reading and checking models, digests, fetching, exporting with a stand-in Python, failures |

CI never has the model, so `foundation_stereo_real_model` is skipped there.
[`tests/installed_consumer/`](tests/installed_consumer/) builds a lossless graph on one worker
against the installed package and checks the stub model's output against the CPU reference, and the
depth's camera info.

## Debian package

Inside the repository's CI image, with the module's folder as the Holoscan CLI project root:

```bash
holoscan --project-root modules/foundation-stereo package holoscan-foundation-stereo \
  --language cpp --pkg-generator DEB --local
```

It installs to `/usr`, depends on `holoscan (>= 5.0.0)` and `holoscan-messages (>= 1.0.0)`, and
holds no model.

## Security

Treat a model file as trusted input: TensorRT parses an ONNX model or deserializes an engine as it is
given. Exporting runs the upstream code and unpickles a PyTorch checkpoint, which can run arbitrary
code; do it in an environment of its own, with checkpoints from sources you trust.

## Licenses

Apache-2.0. The models are not covered by it; see the note at the top and [NOTICE](NOTICE).
HoloInfer and the rest of the SDK are Apache-2.0; TensorRT and CUDA are under NVIDIA's own terms.
