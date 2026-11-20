# holoscan-stereo-disparity

A Holoscan SDK 5 Module that turns a rectified stereo pair and its camera info into a disparity
map, a depth image and the depth image's camera info. `StereoDisparityOp` runs NVIDIA VPI 4's
stereo disparity estimator (`vpiSubmitStereoDisparityEstimator`) on the GPU, or on the Optical Flow
Accelerator and the PVA of Jetson Orin and Thor. It takes any rectified pair, from any camera: a
depth camera's infrared pair, the halves of a side-by-side stereo camera, two cameras rectified
together, or a recording. The camera info is
[holoscan-messages](../messages/)' `StereoCameraInfo`, and the depth goes out with a `CameraInfo`,
so that a point cloud can take it as it comes. The
[Stereo Disparity demo](../../applications/stereo_disparity/) runs it end to end.

| | |
| --- | --- |
| CMake package | `find_package(holoscan-stereo-disparity)` |
| Targets | `holoscan::stereo_disparity` |
| Headers | `<holoscan/stereo_disparity/...>` |
| C++ namespace | `holoscan::stereo_disparity` |
| Debian package | `holoscan-stereo-disparity` |
| Depends on | [holoscan-messages](../messages/) |
| Platforms | x86_64; aarch64 (Jetson Thor and Orin, JetPack 7.2) |

## The input must be rectified

The estimator searches along image rows: a point must appear on the same row in both images. Give
the operator rectified images, and camera info that says so: both cameras `Rectified`, which also
gives the one focal length, the baseline and `doffs`. Rectifying a raw pair is a separate step,
which this module does not do. Some depth cameras rectify their infrared pair themselves; their
calibration then has no distortion, R the identity and one focal length, which
`read_stereo_camera_info_yaml()` reads as a rectified pair.

## `StereoDisparityOp`

| Port | Type | |
| --- | --- | --- |
| `left`, `right` (inputs) | `holoscan::schema::ImageT` | MONO8, RGB8, BGR8, RGBA8 or BGRA8, `[H, W]` or `[H, W, C]`, rows packed, in host or CUDA device memory; both the same size. Color is matched as its BT.601 luma, 0.299 R + 0.587 G + 0.114 B |
| `camera_info` (input, mandatory) | `holoscan::messages::schema::StereoCameraInfoT` | The pair's camera info: both cameras `Rectified` and of the images' size (below) |
| `disparity` (output) | `holoscan::schema::ImageT` | MONO32F `[H, W]` on the GPU (smaller with `downscale`): the left image's disparity in the input's pixels (left column minus right column) |
| `depth` (output, with `depth`) | `holoscan::schema::ImageT` | MONO32F, the disparity's size, on the GPU: meters along the optical axis, `fx * baseline / (disparity + doffs)` |
| `depth_camera_info` (output, with `depth_camera_info`) | `holoscan::messages::schema::CameraInfoT` | The camera of `depth` and `disparity` (below), with every frame |
| `confidence` (output, with `confidence`) | `holoscan::schema::ImageT` | MONO16, the disparity's size, on the GPU: VPI's confidence, 0 to 65535 (most confident); `cuda` and `pva` |

**Invalid pixels are 0** in both `disparity` and `depth`: where VPI found no match (its CUDA backend
writes `0x7fff` there), where its confidence is below `confidence_threshold`, and where the
disparity is 0 or less, or beyond the search. A disparity of 0 would put the point at infinity, so
no valid pixel has it. `holoscan-point-cloud`'s `DepthToPointCloudOp` takes the `depth` output as
it is (MONO32F, `depth_scale` 1) and treats 0 as invalid, as depth cameras do.

The outputs carry the left image's capture time, frame id and header. The left and right images
pair by capture time: the operator runs when both with exactly the same capture time have arrived.
An image with another time never completes its frame, which then gives no output.

### Camera info

The `camera_info` input is mandatory: an unwired one fails `compile()`. It takes
`StereoCameraInfo` with holoscan-messages' `camera_info_contract()`, as
`StereoDisparityConfig::camera_info_input` says:

- `CameraInfoTiming::kLatest` (the default): each frame uses the newest camera info received before
  it, at any rate. `CameraInfoSourceOp` sends a pair's camera_info files periodically; connect it
  with `latest_camera_info_connection()`. Frames before the first camera info are skipped, so the
  first frame of a run may be.
- `CameraInfoTiming::kEveryFrame`: each frame waits for the camera info with its own capture time,
  for a producer that sends one with every pair, such as a rectifier.

Each new camera info is validated (`messages::validate()`); then the operator needs both cameras
`Rectified` and of the images' size, and fails the run with a message that says what is wrong
otherwise. The depth uses the pair's `fx`, `baseline` and `doffs` (`cx_right - cx_left`, 0 for a
pair rectified to one principal point):

```text
depth = fx * baseline / (disparity + doffs)
```

With `depth_camera_info`, the operator sends the depth's camera with every frame, before the depth,
with the frame's capture time, so that a consumer with the default contract uses each frame's own
(connect it with `latest_camera_info_connection()`; a consumer without room keeps the camera info it
has, and the frame goes on without it):
the rectified left camera, of the output's size, its intrinsics in the output's pixels. With
`downscale` `f`, output pixel `u` covers input pixels `u * f` to `u * f + f - 1`, so its `fx` and
`fy` are divided by `f`, `cx' = (cx - (f - 1) / 2) / f` (and so for `cy`), and its binning is
multiplied by `f`. `depth_camera_info()` in `<holoscan/stereo_disparity/camera_info.hpp>` computes
it, and `camera_info_problem()` says what the operator refuses; neither needs the SDK.

### Backends

`StereoDisparityConfig::backend` chooses where VPI runs. VPI 4.1 has no CPU stereo estimator.

| Backend | Runs on | Platforms | Notes |
| --- | --- | --- | --- |
| `Backend::kCuda` (default) | GPU | x86_64, Jetson Thor, Jetson Orin | VPI's semi-global matching (SGM). `max_disparity` 1 to 256, which VPI rounds up to 64, 128 or 256; confidence map |
| `Backend::kOfa` | Optical Flow Accelerator | Jetson Thor, Jetson Orin | VPI's advanced SGM. `max_disparity` 128 or 256, image sides 16 to 16384 (at least 16 once downscaled), no confidence map. Its input and output are block-linear: VPI's CUDA backend converts the pair to it, and the disparity back |
| `Backend::kPva` | OFA, PVA and VIC | Jetson Thor, Jetson Orin | The OFA's SGM with the PVA's post-processing: a median filter (`window` 1, 3, 5 or 7) and a confidence map, also a learned one (`ConfidenceType::kInference`). `max_disparity` 128 or 256; the image at least `max(64, max_disparity / downscale) * downscale` wide |
| `Backend::kAuto` | One of the above | All | Opt-in. Chosen on the first frame, when the image size is known: `ofa` where VPI reports an OFA and a VIC and the parameters and size suit it, else `cuda`. Logged once, for example `StereoDisparityOp: backend auto chose ofa for 1280x720 images: VPI reports an OFA, and the parameters and image size suit it` |

On Jetson Thor, VPI 4.1.4's `ofa` and `pva` give whole-pixel disparities, up to half a pixel off,
where Orin's have a sub-pixel part; and `pva` there leaves a few percent of the pixels far off. The
[demo's README](../../applications/stereo_disparity/README.md#accuracy) has the numbers for each
backend and platform.

Two limits of VPI 4.1.4 are refused before VPI sees them, with a message that names them:

- **One or two passes on Jetson Thor.** There, `ofa` and `pva` take 1.5 s per 1280x720 frame with
  `num_passes` 1 or 2 (`pva` up to 3.1 s), against 15 to 24 ms with 3; on Orin,
  fewer passes are faster. The operator refuses `num_passes` 1 and 2 on `ofa` and `pva` on Thor,
  and `auto` chooses `cuda` there instead.
  `probe_platform()` recognizes Thor by its integrated GPU's compute capability: 11.0 (10.1 before
  CUDA 13), where Orin's is 8.7.
- **Some widths on `pva` with `max_disparity` 256 and a confidence map, at full size.** VPI pads the
  right edge by `384 - width % 256` pixels, and the PVA takes at most 255: a width whose remainder
  modulo 256 is 1 to 127 fails the frame (`Valid right padding range is [0, 255] inclusive, but
  input is 304` at 848x480). Measured on Jetson Orin at every width from 256 to 1024; `downscale` 2
  or more, `max_disparity` 128 or no confidence map all avoid it.

A backend this platform lacks is refused when the operator is made, naming what is missing.
`available_backends()` and `probe_platform()` in `<holoscan/stereo_disparity/platform.hpp>` say what
the platform has; `resolve_auto()`, `params_problem()` and `size_problem()` in
`<holoscan/stereo_disparity/disparity_options.hpp>`, with `platform_problem()`, are the rules
themselves, which need no SDK or GPU. `StereoDisparityOp::resolved_backend()` and `StereoDisparityStats::backend` report the backend
that runs.

VPI reads device images where they are: `cuda` matches MONO8 in place and converts color to gray on
the GPU; `ofa` and `pva` convert either to the OFA's block-linear gray on the GPU (the VIC would
take 3.6 ms per 1280x720 image on Jetson Thor, CUDA takes 0.2 ms). One kernel turns the result into
the outputs, on the runtime's stream, which the operator never waits for. Per frame:

| Backend | Host waits | Copies on the GPU |
| --- | --- | --- |
| `cuda` | 1: `vpiStreamSync` | None: VPI reads device images in place and writes the operator's buffers |
| `ofa` | 2: `vpiStreamSync`; then, with the disparity locked for CUDA (which converts it from block-linear on CUDA, 0.07 ms at 1280x720 on Thor against 2.6 ms on the VIC), the copy into the operator's buffer, before the unlock | 1: the disparity, out of VPI's image |
| `pva` | 2: the operator's stream, before VPI's own stream reads the pair; `vpiStreamSync` | 1, 2 with the confidence map: VPI copies them out of the PVA's images |

`cuda` and `ofa` run on a VPI stream that wraps the operator's CUDA stream. `pva` has a VPI stream
of its own: on Jetson Thor, VPI 4.1.4 loses the PVA's result on a stream that wraps a CUDA stream
(CUDA reads it unwritten). Host images are copied to the device first, one copy per image, which
the host also waits for when the memory is pageable (CUDA copies from pageable memory before the
call returns); the operator logs it once, at the first such frame:

```text
StereoDisparityOp (cuda backend): the images are in host memory and VPI's matcher reads CUDA device memory, so each frame's pair is copied host to device, 1843200 bytes (1.8 MB) per frame; to avoid the copy, send CUDA device images
```

### Parameters

`DisparityParams` holds VPI's parameters; each backend reads some and ignores the rest. The
defaults are VPI's, except `max_disparity`, and `include_diagonals` on `cuda` and `num_passes` on
Jetson Orin, which were measured ([Defaults](#defaults)). They suit 1280x720: 128 pixels reach a
pair with fx 337 px and a 95 mm baseline down to 0.25 m, and the OFA takes only 128 or 256.

| Field | Default | Read by | |
| --- | --- | --- | --- |
| `max_disparity` | 128 | all | Largest disparity searched, pixels |
| `window` | 5 | `pva` | Median filter of 1, 3, 5 or 7 (`cuda` uses a fixed 9x7 census window) |
| `confidence_threshold` | 32767 | `cuda`, `pva` | Confidence (0 to 65535) below which a disparity is invalid; 0 keeps all |
| `confidence_type` | `kAbsolute` | `cuda`, `pva` | `kAbsolute`, `kRelative`, or `kInference` (`pva` only) |
| `p1`, `p2` | 3, 48 | `cuda`, `ofa`, `pva` | SGM penalties for disparity steps of one pixel and of more; `p2` at most 255 on `cuda`, `p1 + p2` at most 217 on `ofa` and `pva` (89 with `kInference`) |
| `p2_alpha` | 0 | `ofa` | Adaptive `p2` across intensity edges: 0 (off), 1, 2, 4 or 8 |
| `uniqueness` | -1 | `cuda` | How far the best match must beat the second, 0 to 1; -1 skips the check |
| `num_passes` | unset: 1 on Jetson Orin, 3 on Jetson Thor | `ofa`, `pva` | SGM passes, 1 to 3; Jetson Thor takes only 3 |
| `include_diagonals` | unset: off on `cuda`, on on `ofa` | `cuda`, `ofa` | Diagonal SGM paths; set it to choose |
| `downscale` | 1 | `ofa`, `pva` | Divide the outputs' width and height by 1, 2, 4 or 8 (rounded up, as VPI does); `cuda` and `auto` take 1 only |

With `downscale`, output pixel `(u, v)` covers input pixels `u * downscale` to
`u * downscale + downscale - 1` (and so on for `v`), and the disparity stays in the input's pixels,
so `depth` needs no change. A consumer that projects the smaller depth image uses `fx / f`,
`fy / f`, `(cx - (f - 1) / 2) / f` and `(cy - (f - 1) / 2) / f` for a factor `f`, as the
`depth_camera_info` output says. VPI's `quality` parameter is ignored by VPI 4 and is not exposed.

### Defaults

Each backend's default is the fastest setting whose depth is nearly as good as the old default's
(`max_disparity` 128, diagonals on, 3 passes): a median depth error at most 0.5 percentage points
higher, and a share of pixels within 5 % of the true depth at most 0.03 lower, on the synthetic
scene and on the local 1280x720 and 848x480 recordings, on every platform measured. The output
keeps the input's size (`downscale` 1), which consumers expect of a default.

Time per frame in `compute()` on the synthetic 1280x720 MONO8 scene in GPU memory, and the depth
against the camera's own on the 1280x720 recording with its projector on (`cuda` on x86_64,
`ofa` and `pva` on Jetson Orin; `cuda` gives the same numbers everywhere). The time is the host's,
from taking the pair to having the outputs queued, so it includes waiting for the producer's GPU
work on the pair (here the scene's rendering) when that is still running: the operator's waits for
VPI also wait for it. Measured with
[`tools/sweep.py`](../../applications/stereo_disparity/tools/sweep.py) (SDK 5.0.0, CUDA 13.2,
VPI 4.1.4; driver 595.99.02 on x86_64, JetPack 7.2 on both Jetsons, with driver 595.78 on Jetson
Orin). The Jetson Thor is a shared CI runner, so its times vary more.

| Backend | Setting | RTX 5090 | Jetson Orin | Jetson Thor | Median error | Within 5 % | |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda` | 128, diagonals on | 1.84 ms | 70.1 ms | 79.0 ms | 1.05 % | 0.897 | Old default |
| `cuda` | **128, diagonals off** | 1.36 ms | 20.8 ms | 11.8 ms | 1.15 % | 0.884 | **Default** |
| `cuda` | 256, diagonals off | 1.85 ms | 29.8 ms | 47.2 ms | 1.15 % | 0.881 | Slower |
| `cuda` | 64, diagonals off | 1.37 ms | 24.1 ms | 7.7 ms | 1.65 % | 0.758 | Worse depth |
| `ofa` | **128, diagonals on, 3 passes** | | 47.0 ms | 18.9 ms | 1.15 % | 0.864 | Old default; **default on Thor** |
| `ofa` | 128, diagonals off, 3 passes | | 53.2 ms | 19.5 ms | 1.15 % | 0.848 | Slower |
| `ofa` | 128, diagonals on, 2 passes | | 32.1 ms | 1559 ms | 1.15 % | 0.859 | Thor refuses it |
| `ofa` | **128, diagonals on, 1 pass** | | 23.0 ms | refused | 1.15 % | 0.840 | **Default on Orin** |
| `ofa` | 128, diagonals off, 1 pass | | 27.9 ms | refused | 1.25 % | 0.815 | Worse depth |
| `pva` | **128, 3 passes** | | 56.9 ms | 15.1 ms | 1.05 % | 0.876 | Old default; **default on Thor** |
| `pva` | 128, 2 passes | | 46.4 ms | 1553 ms | 1.05 % | 0.872 | Thor refuses it |
| `pva` | **128, 1 pass** | | 35.0 ms | refused | 1.15 % | 0.852 | **Default on Orin** |
| `pva` | 256, 3 passes | | 78.1 ms | 18.1 ms | 1.05 % | 0.869 | Slower |

So `cuda` leaves the diagonal paths out everywhere (a third of the time on Jetson Orin, 0.1 point
more error), and `ofa` and `pva` run 1 pass on Jetson Orin (half the time) and 3 on Jetson Thor,
where 1 and 2 take over a second. `downscale` 2 or 4 is faster still on `ofa` and `pva` (21 and
23 ms for `ofa` on Orin) with the same error, but gives a smaller depth image, so it stays a choice.
The [demo's README](../../applications/stereo_disparity/README.md#accuracy) has every backend's
accuracy on every recording.

## Use it in your graph

```cmake
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core sensor_io)
find_package(holoscan-messages 1.0 REQUIRED CONFIG)
find_package(holoscan-stereo-disparity 1.0 REQUIRED CONFIG)
target_link_libraries(my_app PRIVATE holoscan::stereo_disparity holoscan::messages)
```

A live camera, whose frames may be dropped: every frame edge keeps only the newest frame
(`kDropOldest`, depth 1), so that no operator waits for a slower one, on whichever worker each runs.

```cpp
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/messages/camera_info_input.hpp>
#include <holoscan/messages/camera_info_source_op.hpp>
#include <holoscan/stereo_disparity/stereo_disparity_op.hpp>

namespace hm = holoscan::messages;
namespace sd = holoscan::stereo_disparity;

// First thing in main(): an ofa or pva operator starts VPI's threads when it is made.
auto signals = holoscan::ManagedShutdownSignals::create();

auto camera_info = graph.op<hm::CameraInfoSourceOp>(
    "camera_info", hm::CameraInfoSourceConfig{.camera_info_path = "left.yaml",
                                              .right_camera_info_path = "right.yaml"});
auto stereo = graph.op<sd::StereoDisparityOp>(
    "stereo", sd::StereoDisparityConfig{.backend = sd::Backend::kCuda,
                                        .depth = true,
                                        .depth_camera_info = true,
                                        .max_width = 1280,
                                        .max_height = 720});
const holoscan::ConnectionOptions live{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                       .queue_depth = 1U};
graph.add_flow(camera->left, stereo->left, live);  // Output<holoscan::schema::ImageT>
graph.add_flow(camera->right, stereo->right, live);
graph.add_flow(camera_info->stereo_camera_info, stereo->camera_info,
               hm::latest_camera_info_connection());
graph.add_flow(stereo->disparity, viewer->disparity, live);  // always declared: connect it
graph.add_flow(stereo->depth, cloud->depth, live);  // e.g. a depth-to-point-cloud operator
graph.add_flow(stereo->depth_camera_info, cloud->camera_info, hm::latest_camera_info_connection());
```

For a pair that a rectifier sends with its camera info on every frame, connect that output instead
and set `.camera_info_input = {.timing = hm::CameraInfoTiming::kEveryFrame}`, with the images'
edge options. Bind the image outputs you connect to a GPU (`bind_tensor_output_device`), as the demo
does. Every declared output must be connected: an emit into an unconnected output fails.

A lossless chain, where every frame must arrive (a recording, a test), connects the frame edges
with `QueuePolicy::kBlocking` instead, and must stay on one worker, as the demo does:
`graph.partition("name").include_all()` runs the whole graph on one. A graph with a second source,
such as a `CameraInfoSourceOp`, or one that forks or joins, is split by the SDK's automatic
partitioning onto several workers; the demo's, with a recording's depth, would get one per operator.
Across workers, the SDK 5.0 runs a producer even while its consumer's lossless queue is full, and
refuses what it sends: an emit fails with `kBackpressured`, and an allocation with
`kResourceExhausted` while every slot of the output's pool is still held. A source can treat both
as backpressure and send again on its next tick, but `StereoDisparityOp` has taken its pair by then
and returns the error, which fails the run.

The SDK takes over SIGINT and SIGTERM only while the process has no other thread. Making a
`StereoDisparityOp` with `Backend::kOfa` or `kPva`, `validate_config()` of one, `probe_platform()`
and `available_backends()` ask VPI and CUDA, which start threads; so create
`holoscan::ManagedShutdownSignals` at the start of `main()`, as above, before any of them (a plain
`holoscan::run(plan)` then adopts it), or run with `holoscan::RunSignalPolicy::kCallerManaged`.
Otherwise `holoscan::run(plan)` returns nonzero before the run starts.

`<holoscan/stereo_disparity/synthetic_scene.hpp>` renders a stereo pair of textured planes with
known disparity, and gives its truth and its camera info (`scene_camera_info()`), for tests.

## Requirements

- Holoscan SDK 5.0 (Core and Sensor I/O; Testing for the tests), CUDA 13, CMake 4.0 or later and a
  C++20 compiler.
- holoscan-messages 1.0, installed, or added to the same build first (the repository root does).
- NVIDIA VPI 4.1 or later 4.x (`vpi4-dev`; 4.1.4 tested), found in `/opt/nvidia/vpi4`. The repository's CI image
  installs it from NVIDIA's Jetson package repository; see [`.github/docker/`](../../.github/docker/).
  On a Jetson in a container, run with the NVIDIA runtime and `--privileged` for the OFA, PVA and
  VIC, and leave `DISPLAY` unset unless there is a display, or VPI fails with
  `EGL_NOT_INITIALIZED`.

## Build, test and install

From the repository root, in the repository's CI image or with the requirements above:

```bash
cmake -S modules/stereo-disparity -B build/stereo-disparity -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON -DCMAKE_PREFIX_PATH="$PWD/install"  # where holoscan-messages is installed
cmake --build build/stereo-disparity
ctest --test-dir build/stereo-disparity --output-on-failure
cmake --install build/stereo-disparity --prefix "$PWD/install"
```

| CMake option | Default | |
| --- | --- | --- |
| `BUILD_TESTING` | `OFF` | Build the tests, when the module is built on its own |
| `HOLOSCAN_STEREO_DISPARITY_BUILD_TESTING` | `OFF` | Build the tests inside another project |
| `CMAKE_CUDA_ARCHITECTURES` | `native` (`all-major` for the package) | GPUs to compile for, unless already set |

Another project can also build the module as part of its own build, with `add_subdirectory()` or
`FetchContent`, and link `holoscan::stereo_disparity` without `find_package()`. The module then
leaves the including project's `BUILD_TESTING` and `BUILD_SHARED_LIBS` alone.

The tests:

| Test | Needs | Checks |
| --- | --- | --- |
| `stereo_disparity_operator` | GPU, VPI | The operator through the SDK's operator test harness, on every backend the platform has: the synthetic scene against its truth from MONO8 and color, host and device images, at full size and downscaled; depth against the formula, pixel for pixel; the depth's camera info, its size, intrinsics and capture time; the confidence map; provenance; the camera info taken as the newest one and with every frame, frames before the first one skipped, and a calibration that changes; `auto`'s choice and its log (on a CI runner, `ofa` on a Jetson and `cuda` on x86_64); the copy log; bad frames, camera info (another size, a raw pair, an invalid one) and configurations. On a Jetson it requires the `ofa` backend |
| `stereo_disparity_options` | Nothing | Every backend's parameter and size rules, `auto`'s choice for every platform, the camera info the operator refuses, the depth's camera info (a point lands on the output pixel that covers it, at every factor) and the depth formula; also built from `tests/host/`, with holoscan-messages' SDK-free sources, by the hosted CI job |

[`tests/installed_consumer/`](tests/installed_consumer/) is a separate project that uses the
installed packages: a graph that writes the synthetic scene's camera_info files, sends them with
`CameraInfoSourceOp`, runs `StereoDisparityOp` on every backend the platform has and checks it
against the scene.

## Debian package

Inside the repository's CI image, with the module's folder as the Holoscan CLI project root:

```bash
holoscan --project-root modules/stereo-disparity package holoscan-stereo-disparity --language cpp \
  --pkg-generator DEB --local
```

It builds for every major GPU architecture, installs to `/usr`, and depends on
`holoscan (>= 5.0.0)`, `holoscan-messages (>= 1.0.0)` and `libnvvpi4 (>= 4.1)`.

## Licenses

Apache-2.0. VPI is under NVIDIA's own terms, installed as `/opt/nvidia/vpi4/doc/VPI_EULA.txt`.
See [NOTICE](NOTICE).
