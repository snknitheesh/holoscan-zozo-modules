# holoscan-yolo26

A Holoscan SDK 5 Module for end-to-end YOLO26 object detection on the GPU. `Yolo26DetectionOp` takes
an image and emits the objects in it as GPU tensors: boxes in the image's own pixels, a score and a
class for each, and their count. Inside, a CUDA kernel letterboxes the image into the model's
input, the SDK's TensorRT inference library (HoloInfer) runs the model, and a second kernel turns
its output into detections. YOLO26 needs no non-maximum suppression, so nothing else runs.
`DetectionOverlayOp` turns the detections into HoloViz layers. The
[YOLO26 Detection demo](../../applications/yolo26_detection/) runs both end to end.

> **The model is AGPL-3.0, and it is not part of this module.** YOLO26 weights and the Ultralytics
> software that exports them are licensed under the
> [GNU Affero General Public License v3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
> (or an Ultralytics Enterprise License). This module is Apache-2.0 and contains no Ultralytics code
> or weights. [`tools/prepare_model.py`](tools/prepare_model.py) downloads pinned weights and
> exports them with an Ultralytics tool that you install separately, on your machine (see
> "Prepare the model" below). The terms that apply to using and distributing the model, and anything
> built with it, are yours to review. The tests need neither: they use a stand-in model that the
> module writes itself.

| | |
| --- | --- |
| CMake package | `find_package(holoscan-yolo26 [COMPONENTS overlay])` |
| Targets | `holoscan::yolo26`, `holoscan::yolo26_overlay` |
| Headers | `<holoscan/yolo26/...>` |
| C++ namespace | `holoscan::yolo26` |
| Debian package | `holoscan-yolo26` |
| Model tool | `tools/prepare_model.py`, installed as `share/holoscan-yolo26/prepare_model.py` |
| Platforms | x86_64; aarch64 (Jetson Thor, JetPack 7) |

## Operators

### `Yolo26DetectionOp`

| Port | Direction | Type |
| --- | --- | --- |
| `image` | in | `holoscan::schema::ImageT`: RGB8, BGR8, RGBA8 or BGRA8, uint8 `[H, W, C]`, pixels packed within each row; GPU memory on device 0, or host or pinned memory |
| `boxes` | out | `holoscan::Tensor`: float32 `[max_detections, 4]`, `x1, y1, x2, y2` in image pixels; GPU memory |
| `scores` | out | `holoscan::Tensor`: float32 `[max_detections]`; GPU memory |
| `class_ids` | out | `holoscan::Tensor`: int32 `[max_detections]`; GPU memory |
| `count` | out | `holoscan::Tensor`: int32 `[1]`, the detections in this image; GPU memory |

Slots from `count` on are zero. All four outputs carry the image's capture time and frame id. They
go out one after another: should one fail to go out, those before it have been delivered alone and
the frame fails. Pair them by capture time, with an `OnAligned` contract, as `DetectionOverlayOp`
does. Images may change size from frame to frame; images in host memory are copied to the GPU
through a staging buffer that grows with the largest one. The frames share the model's buffers, and
the runtime may give each one another CUDA stream, so each frame's stream waits on a CUDA event that
the previous frame recorded.

| `Yolo26DetectionConfig` | Default | Meaning |
| --- | --- | --- |
| `model` | none | An `.onnx` end-to-end YOLO26 export, or a TensorRT engine built for this GPU |
| `input_size` | `640` | The model's input side `S`, a multiple of 32 in 32..4096 |
| `score_threshold` | `0.25` | Minimum score, in `[0, 1]` |
| `max_detections` | `300` | Capacity of the outputs; further detections are dropped, the lowest scores first, with a warning once per run |
| `num_classes` | `80` | The model's classes; rows with another class value are dropped |
| `pad_value` | `114` | The letterbox border's gray level |
| `fp16` | `false` | Build the engine with FP16 kernels allowed |
| `cuda_graphs` | `true` | Let HoloInfer replay the inference as a CUDA graph |
| `build_workspace_bytes` | 1 GiB | TensorRT's workspace limit while it builds the engine |

Given an ONNX file, HoloInfer builds a TensorRT engine and caches it beside the file, named after
the GPU, the TensorRT version and the settings, so later runs load it in seconds. The first build of
YOLO26n took 10 to 30 s on an RTX 5090. The cache is not keyed on the file's contents: after
exporting a model again to the same name, delete its `<name>.*.engine.*` files (`prepare_model.py`
does). `start()` fails, naming the problem, when the model does not take `[1, 3, S, S]` or give
`[1, N, 6]` float32.

**Call `holoscan::yolo26::prepare_engine(config)` before running the graph.** It builds and caches
the engine of an `.onnx` model. Some TensorRT releases (10.13 among them) change the SIGINT and
SIGTERM handlers while they build an engine, and the SDK's managed signal handling accepts that only
before a run starts. `start()` then loads the cached engine; without `prepare_engine()` it builds the
engine itself. TensorRT also starts threads of its own, after which the SDK no longer takes over
SIGINT and SIGTERM by itself (`RUNTIME_MANAGED_SIGNAL_DOMAIN_REQUIRED`): create the process's
`holoscan::ManagedShutdownSignals` at the start of `main()`, while it is the only thread, and keep it
until the run ends, as below.

The operator uses HoloInfer directly rather than the SDK's `InferenceOp`, which emits its outputs
with its own activation time and no frame id: the decode needs each image's letterbox, and every
output must keep the image's capture time.

### `DetectionOverlayOp`

| Port | Direction | Type |
| --- | --- | --- |
| `boxes`, `scores`, `class_ids`, `count` | in | `Yolo26DetectionOp`'s outputs, in GPU or host memory |
| `labels` | out | `holoscan::Tensor`: a HoloViz text layer; per detection, a dark shadow and a `<class> 0.87` label above its box, in its class's color |
| `overlay` | out | `holoscan::schema::ImageT`: a HoloViz RGBA8 image layer; the boxes, 2 px wide in a color per class, on a transparent background |

`DetectionOverlayConfig` gives the image's size, the detector's `max_detections` and the class
names (ids without one show as `class N`). The overlay emits the labels first and then the boxes,
both stamped with the frame's capture time, so a HoloViz `FrameTrigger` on the boxes layer draws
each frame's labels and boxes together over the image they came from. Keep at least four images
queued on the HoloViz image edge, so that the frame's image is still there when its detections
arrive; the demo shows the full wiring.

## What it computes

The model takes a square `S x S` input. An image of `W x H` pixels is scaled by
`r = min(S / W, S / H)` to `round(W * r) x round(H * r)` (rounded half to even), centered, and the
rest is filled with gray 114; where the padding is odd, the extra pixel goes on the right or the
bottom. This is the geometry of the Ultralytics predictor, so boxes map back to the image as they do
there. Each input pixel is a bilinear sample of the image with pixel centers aligned and edges
clamped, as OpenCV's `INTER_LINEAR` computes it, in RGB order and divided by 255, written as float32
`[1, 3, S, S]` planes. Alpha is ignored.

An end-to-end YOLO26 export gives `[1, N, 6]` rows of `[x1, y1, x2, y2, score, class]` in input
pixels, best first (`N` = 300). The decode keeps a row when its score is finite and at least the
threshold, its class value rounds to a class below `num_classes`, and its box, mapped back as
`x = (u - pad_left) / r`, `y = (v - pad_top) / r` and clamped to `[0, W] x [0, H]`, still has a
positive width and height. Kept rows keep their order, up to `max_detections`.

On the COCO image the demo's tests use, the detections match those of Ultralytics' own predictor
running the same ONNX file with ONNX Runtime to within 0.002 in score and 0.1 pixel in each box
edge.

## Use it in your graph

```cmake
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core sensor_io)
find_package(holoscan-yolo26 1.0 REQUIRED CONFIG)
target_link_libraries(my_app PRIVATE holoscan::yolo26)   # add holoscan::yolo26_overlay for the overlay
```

```cpp
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/yolo26/yolo26_detection_op.hpp>

namespace y26 = holoscan::yolo26;

// First thing in main(), while it is the only thread.
auto signals = holoscan::ManagedShutdownSignals::create();

const y26::Yolo26DetectionConfig config{.model = "/data/yolo26n.onnx",
                                        .score_threshold = 0.4F,
                                        .max_detections = 100};
auto detector = graph.op<y26::Yolo26DetectionOp>("detector", config);
graph.add_flow(camera->image, detector->image);      // Output<holoscan::schema::ImageT>
graph.add_flow(detector->boxes, consumer->boxes);    // Input<holoscan::Tensor>, and so on

// The outputs are in GPU memory: bind their device when compiling.
holoscan::CompileOptions options;
for (const char* port : {"boxes", "scores", "class_ids", "count"}) {
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "detector", .output_port = port, .device = holoscan::DeviceId{0}});
}
const auto plan = holoscan::compile(graph, std::move(options));
y26::prepare_engine(config);   // Before the run: see above.
const int status = holoscan::run(plan);   // Adopts `signals`.
```

The kernels need no Holoscan graph: `<holoscan/yolo26/kernels.hpp>` has `launch_letterbox()` and
`launch_decode()`, and `<holoscan/yolo26/letterbox.hpp>` the letterbox geometry, which needs no CUDA
either. For tests, `<holoscan/yolo26/stub_model.hpp>` writes the stand-in model, an ONNX network
with the tensors of an end-to-end YOLO26 export that "detects" red (class 0) and blue (class 1)
regions, and `<holoscan/yolo26/synthetic_scene.hpp>` renders frames with a red and a blue rectangle
and checks detections against them.

## Prepare the model

`Yolo26DetectionOp` needs YOLO26 exported to ONNX with its end-to-end head. `prepare_model.py`
downloads the pinned `yolo26n.pt` from the Ultralytics release
[v8.4.0](https://github.com/ultralytics/assets/releases/tag/v8.4.0) and checks its SHA-256 digest,
exports it with the Ultralytics `yolo` command (`format=onnx nms=False`), checks the export's input
and output, and writes `yolo26n.onnx` and `yolo26n.labels.txt`. It also downloads, and checks, the
COCO validation image `000000039769.jpg`, which the demo uses and which is subject to the COCO terms
of use (the Terms of Use page at [cocodataset.org](https://cocodataset.org/)), into `images/`;
`--no-samples` skips it. The script only runs the `yolo` command; it does not import Ultralytics.

Installing `ultralytics` runs code from the Python package index, and its version and dependencies
are not pinned by this repository. Install it in its own virtual environment, not the one that holds
the repository's tools, and review what it pulls in. The script runs the exporter with
`YOLO_AUTOINSTALL=false`, so the exporter installs no further packages on its own; install the ONNX
packages it needs with it. Tested with `ultralytics` 8.4.190:

```bash
python3 -m venv ~/ultralytics-venv
~/ultralytics-venv/bin/pip install ultralytics onnx onnxslim onnxruntime
python3 modules/yolo26/tools/prepare_model.py --output data/yolo26_detection \
  --yolo ~/ultralytics-venv/bin/yolo
```

From the installed module, run `/usr/share/holoscan-yolo26/prepare_model.py` (or
`<prefix>/share/holoscan-yolo26/prepare_model.py`); CMake projects find its path in
`holoscan-yolo26_PREPARE_MODEL`. Pass `--model yolo26s` (or `m`, `l`, `x`) for a larger pinned
model, `--weights your_model.pt` to export your own trained weights instead (without a digest to
check), `--imgsz` for another input size, and `--force` to export again over an existing model.

## Security

Treat a model file as trusted input. The operator hands the file it is given to TensorRT, which
parses an ONNX model or deserializes an engine; load only models and engines that you exported or
built yourself, or otherwise trust. The model files come from a third-party exporter, Ultralytics, which
you install from the Python package index outside this repository's pinned tools: install it in an
isolated environment, as above, and review what it installs.

## Requirements

- Holoscan SDK 5.0 (Core, Sensor I/O and Inference, which brings HoloInfer and TensorRT 10;
  Visualization for the overlay; Testing for the tests), CUDA 13, CMake 4.0 or later and a C++20
  compiler.
- For the model only: Python 3.12 or later and, in a separate environment, the `ultralytics`
  package and its ONNX exporter's dependencies.

## Build, test and install

From the repository root, in the repository's CI image or with the requirements above:

```bash
cmake -S modules/yolo26 -B build/yolo26 -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build/yolo26
ctest --test-dir build/yolo26 --output-on-failure
cmake --install build/yolo26 --prefix "$PWD/install"
```

| CMake option | Default | |
| --- | --- | --- |
| `HOLOSCAN_YOLO26_BUILD_OVERLAY` | `ON` | Build `holoscan::yolo26_overlay` (needs Holoscan Visualization) |
| `BUILD_TESTING` | `OFF` | Build the tests |
| `CMAKE_CUDA_ARCHITECTURES` | `native` (`all-major` for the package) | GPUs to compile for |

The tests need no download: the stub model stands in for YOLO26.

| Test | Needs | Checks |
| --- | --- | --- |
| `yolo26_letterbox` | GPU | The letterbox kernel against a host evaluation, bit for bit, for every encoding, with padded rows, downscaling and upscaling; pixel centers aligned, channels in RGB order, the padding in place and alpha ignored |
| `yolo26_decode` | GPU | The decode kernel against a host decode: the threshold and its boundary, invalid and non-finite classes, scores and corners, clamping, order, capacity, the count and the zeroed slots, inputs larger than one block, and letterboxes padded on either axis |
| `yolo26_operators` | GPU | Both operators through the SDK's operator test harness, with the stub model and TensorRT: every encoding, from GPU, host and pinned memory, wide, tall and small images, input sizes of 640 and 320, changing image sizes, mislabeled channels (which swap the classes), the threshold and the capacity, provenance on every output, unusable images, settings and models; the overlay's boxes, labels and their position, and a clear overlay without detections |
| `yolo26_host` | Nothing | The letterbox geometry for wide and tall images, even and odd padding, half-to-even rounding and tiny and huge images; the stub model file; the synthetic scenes and the truth matching |
| `yolo26_prepare_model` | Python | `prepare_model.py` without Ultralytics or the network: reading and checking exports, checksums, a failed or missing exporter, and reuse of prepared files |

`yolo26_host` and `yolo26_prepare_model` are also a project of their own in
[`tests/host/`](tests/host/), which the hosted CI job builds and runs.
[`tests/installed_consumer/`](tests/installed_consumer/) is a separate project that uses the
installed package: a small graph that runs `Yolo26DetectionOp` with the stub model on synthetic
frames and checks every frame's detections, and a run of the installed `prepare_model.py`. Build it
with the install prefix on `CMAKE_PREFIX_PATH` and run `ctest` there.

## Debian package

Inside the repository's CI image, with the module's folder as the Holoscan CLI project root:

```bash
holoscan --project-root modules/yolo26 package holoscan-yolo26 --language cpp \
  --pkg-generator DEB --local
```

`--language cpp` selects the module rather than its package entry, which has the same name. This
builds for every major GPU architecture and writes `holoscan-yolo26_1.0.0_<arch>.deb` to the current
directory. It installs to `/usr` and depends on `holoscan (>= 5.0.0)`, so
`apt install ./holoscan-yolo26_1.0.0_<arch>.deb` needs NVIDIA's CUDA repository. It holds no model.

## Performance

On an RTX 5090 (driver 595.91.07, shared with another process) with SDK 5.0.0, CUDA 13.2 and
TensorRT 10.16, for YOLO26n at 640 on the 640x480 COCO image, the GPU work per frame is a median
of:

| Stage | FP32 engine | FP16 engine |
| --- | --- | --- |
| Letterbox kernel | 2.4 µs | 2.4 µs |
| TensorRT, 210 to 216 kernels | 706 µs | 386 µs |
| Decode kernel | 1.7 µs | 1.7 µs |
| Letterbox start to decode end | 754 µs | 442 µs |

Measured with Nsight Systems over 300 frames of the demo, tracing CUDA graph nodes; the demo's
README has the command and the latency from capture to detections on the host.

The operator's code was validated, before it became this module, with SDK 5.0.0 on x86_64 (RTX
5090, CUDA 13.2, TensorRT 10.16; RTX 5080 Laptop GPU, CUDA 13.0, TensorRT 10.13) and on Jetson AGX
Thor (JetPack 7, CUDA 13.0, TensorRT 10.15), where the real model's FP32 and FP16 engines give the
same detections as on x86_64.

## Licenses

Apache-2.0. The YOLO26 model is not covered by it; see the note at the top and [NOTICE](NOTICE).
HoloInfer and the rest of the SDK are Apache-2.0; TensorRT and CUDA are under NVIDIA's own terms.
