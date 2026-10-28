# holoscan-stereo-disparity

A Holoscan SDK 5 Module that turns a rectified stereo pair into a disparity map and, given the
pair's geometry, a depth image. `StereoDisparityOp` runs NVIDIA VPI 4's stereo disparity estimator
(`vpiSubmitStereoDisparityEstimator`) on the GPU, or on the Optical Flow Accelerator and the PVA of
Jetson Orin and Thor. It takes any rectified pair, from any camera: a RealSense's infrared eyes, the
halves of a side-by-side camera such as a ZED, two cameras rectified together, or a recording. The
[Stereo Disparity demo](../../applications/stereo_disparity/) runs it end to end.

| | |
| --- | --- |
| CMake package | `find_package(holoscan-stereo-disparity)` |
| Targets | `holoscan::stereo_disparity` |
| Headers | `<holoscan/stereo_disparity/...>` |
| C++ namespace | `holoscan::stereo_disparity` |
| Debian package | `holoscan-stereo-disparity` |
| Platforms | x86_64; aarch64 (Jetson Thor and Orin, JetPack 7.2) |

## The input must be rectified

The estimator searches along image rows: a point must appear on the same row in both images. Give
the operator rectified images; rectifying a raw pair is a separate step, which this module does not
do. A RealSense's infrared eyes are rectified by the camera (its calibration reports a
Brown-Conrady model with all coefficients zero, and the eyes share the focal length and the
principal point).

## `StereoDisparityOp`

| Port | Type | |
| --- | --- | --- |
| `left`, `right` (inputs) | `holoscan::schema::ImageT` | MONO8, RGB8, BGR8, RGBA8 or BGRA8, `[H, W]` or `[H, W, C]`, rows packed, in host or CUDA device memory; both the same size. Color is matched as its BT.601 luma |
| `geometry` (input, with `geometry_input`) | `holoscan::stereo_disparity::schema::StereoGeometryT` | The pair's geometry ([schema](schemas/stereo_geometry.fbs)) |
| `disparity` (output) | `holoscan::schema::ImageT` | MONO32F `[H, W]` on the GPU: the left image's disparity in pixels (left column minus right column) |
| `depth` (output, with `depth`) | `holoscan::schema::ImageT` | MONO32F `[H, W]` on the GPU: meters along the optical axis |
| `confidence` (output, with `confidence`) | `holoscan::schema::ImageT` | MONO16 `[H, W]` on the GPU: VPI's confidence, 0 to 65535 (most confident); `cuda` and `pva` |

**Invalid pixels are 0** in both `disparity` and `depth`: where VPI found no match (its CUDA backend
writes `0x7fff` there), where its confidence is below `confidence_threshold`, and where the
disparity is 0 or less, or beyond the search. A disparity of 0 would put the point at infinity, so
no valid pixel has it. `holoscan-point-cloud`'s `DepthToPointCloudOp` takes the `depth` output as
it is (MONO32F, `depth_scale` 1) and treats 0 as invalid; so does a RealSense's own depth.

The outputs carry the left image's capture time, frame id and header. The inputs pair by capture
time: the operator runs when the left and right images (and the geometry, when it is an input) with
exactly the same capture time have arrived. A part with another time never completes its frame,
which then gives no output.

### The geometry

`StereoGeometry` (`<holoscan/stereo_disparity/stereo_geometry.hpp>`) is what depth needs, and no
more: the rectified images' focal length `fx` in pixels, the `baseline` in meters, the image size,
optionally the left principal point `cx`, `cy` (carried for consumers), and `doffs`, the right
principal point's x minus the left's, which is 0 when the pair is rectified to one principal point:

```text
depth = fx * baseline / (disparity + doffs)
```

Give it in `StereoDisparityConfig::geometry`, for a fixed camera, or enable `geometry_input` and
send a `StereoGeometryT` with every frame (`to_message()` and `from_message()` in
`<holoscan/stereo_disparity/geometry_message.hpp>` convert), for one whose calibration arrives with
its images. Its size must be the images' size. The module needs no camera module: build the
geometry from whatever calibration your camera reports. For a RealSense, `fx` is the infrared
eye's, and `baseline` is minus the x translation of its right-from-left extrinsics.

### Backends

`StereoDisparityConfig::backend` chooses where VPI runs. VPI 4.1 has no CPU stereo estimator.

| Backend | Runs on | Platforms | Notes |
| --- | --- | --- | --- |
| `Backend::kCuda` (default) | GPU | x86_64, Jetson Thor, Jetson Orin | VPI's semi-global matching (SGM). `max_disparity` 1 to 256, which VPI rounds up to 64, 128 or 256; confidence map |
| `Backend::kOfa` | Optical Flow Accelerator, VIC | Jetson Thor, Jetson Orin | VPI's advanced SGM. `max_disparity` 128 or 256, image sides 16 to 16384, no confidence map. Its input is block-linear: VIC converts the gray pair, and converts the block-linear disparity back |
| `Backend::kPva` | OFA, PVA and VIC | Jetson Thor, Jetson Orin | The OFA's SGM with the PVA's post-processing: a median filter (`window` 1, 3, 5 or 7) and a confidence map, also a learned one (`ConfidenceType::kInference`). `max_disparity` 128 or 256; the image at least `max(64, max_disparity)` wide |
| `Backend::kAuto` | One of the above | All | Opt-in. Chosen on the first frame, when the image size is known: `ofa` where VPI reports an OFA and a VIC and the parameters and size suit it, else `cuda`. Logged once, for example `StereoDisparityOp: backend auto chose ofa for 1280x720 images: VPI reports an OFA, and the parameters and image size suit it` |

A backend this platform lacks is refused when the operator is made, naming what is missing.
`available_backends()` and `probe_platform()` in `<holoscan/stereo_disparity/platform.hpp>` say what
the platform has; `resolve_auto()`, `params_problem()` and `size_problem()` in
`<holoscan/stereo_disparity/disparity_options.hpp>` are the rules themselves, which need no SDK or
GPU. `StereoDisparityOp::resolved_backend()` and `StereoDisparityStats::backend` report the backend
that runs.

VPI reads the pair from its own images on the GPU. Device images are converted into them there;
host images are copied to the device first, which the operator logs once, at the first such frame:

```text
StereoDisparityOp (cuda backend): the images are in host memory and VPI's matcher reads CUDA device memory, so each frame's pair is copied host to device, 1843200 bytes (1.8 MB) per frame; to avoid the copy, send CUDA device images
```

### Parameters

`DisparityParams` holds VPI's parameters; each backend reads some and ignores the rest. The
defaults are VPI's, except `max_disparity`, and suit 1280x720: 128 pixels reach a RealSense at
1280x720 (fx 337 px, baseline 95 mm) down to 0.25 m, and the OFA takes only 128 or 256.

| Field | Default | Read by | |
| --- | --- | --- | --- |
| `max_disparity` | 128 | all | Largest disparity searched, pixels |
| `window` | 5 | `pva` | Median filter of 1, 3, 5 or 7 (`cuda` uses a fixed 9x7 census window) |
| `confidence_threshold` | 32767 | `cuda`, `pva` | Confidence (0 to 65535) below which a disparity is invalid; 0 keeps all |
| `confidence_type` | `kAbsolute` | `cuda`, `pva` | `kAbsolute`, `kRelative`, or `kInference` (`pva` only) |
| `p1`, `p2` | 3, 48 | `cuda`, `ofa`, `pva` | SGM penalties for disparity steps of one pixel and of more; `p2` at most 255 on `cuda`, `p1 + p2` at most 217 on `ofa` and `pva` (89 with `kInference`) |
| `p2_alpha` | 0 | `ofa` | Adaptive `p2` across intensity edges: 0 (off), 1, 2, 4 or 8 |
| `uniqueness` | -1 | `cuda` | How far the best match must beat the second, 0 to 1; -1 skips the check |
| `num_passes` | 3 | `ofa`, `pva` | SGM passes, 1 to 3 |
| `include_diagonals` | true | `cuda`, `ofa` | Diagonal SGM paths: slower, better |

VPI's `quality` parameter is ignored by VPI 4 and is not exposed; neither is the OFA's
`downscaleFactor`: the outputs are the input's size.

## Use it in your graph

```cmake
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core sensor_io)
find_package(holoscan-stereo-disparity 1.0 REQUIRED CONFIG)
target_link_libraries(my_app PRIVATE holoscan::stereo_disparity)
```

```cpp
#include <holoscan/stereo_disparity/stereo_disparity_op.hpp>

namespace sd = holoscan::stereo_disparity;

auto stereo = graph.op<sd::StereoDisparityOp>(
    "stereo", sd::StereoDisparityConfig{
                  .backend = sd::Backend::kCuda,
                  .geometry = sd::StereoGeometry{.fx = 336.99, .baseline = 0.09484,
                                                 .width = 1280, .height = 720},
                  .depth = true,
                  .max_width = 1280,
                  .max_height = 720});
graph.add_flow(camera->left, stereo->left);      // Output<holoscan::schema::ImageT>
graph.add_flow(camera->right, stereo->right);
graph.add_flow(stereo->depth, cloud->depth);     // e.g. holoscan-point-cloud's DepthToPointCloudOp
```

Bind the outputs you connect to a GPU (`bind_tensor_output_device`), as the demo does. With
`geometry_input`, connect a `StereoGeometryT` output to `stereo->geometry`. Every declared output
must be connected: an emit into an unconnected output fails.

A process that uses VPI before it runs a graph has VPI's threads running by then, so take over the
signals first: `holoscan::ManagedShutdownSignals::create()` at the start of `main()`, or run with
`holoscan::RunSignalPolicy::kCallerManaged`.

`<holoscan/stereo_disparity/synthetic_scene.hpp>` renders a stereo pair of textured planes with
known disparity, and gives its truth, for tests.

## Requirements

- Holoscan SDK 5.0 (Core and Sensor I/O; Testing for the tests), CUDA 13, CMake 4.0 or later and a
  C++20 compiler.
- NVIDIA VPI 4 (`vpi4-dev`, 4.1 tested), found in `/opt/nvidia/vpi4`. The repository's CI image
  installs it from NVIDIA's Jetson package repository; see [`.github/docker/`](../../.github/docker/).
  On a Jetson in a container, run with the NVIDIA runtime and `--privileged` for the OFA, PVA and
  VIC, and leave `DISPLAY` unset unless there is a display, or VPI fails with
  `EGL_NOT_INITIALIZED`.

## Build, test and install

From the repository root, in the repository's CI image or with the requirements above:

```bash
cmake -S modules/stereo-disparity -B build/stereo-disparity -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON
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
| `stereo_disparity_operator` | GPU, VPI | The operator through the SDK's operator test harness, on every backend the platform has: the synthetic scene against its truth from MONO8 and color, host and device images; depth against the formula, pixel for pixel; the confidence map; provenance; per-frame geometry, and a frame whose geometry has another capture time; `auto`'s choice and its log (on a CI runner, `ofa` on a Jetson and `cuda` on x86_64); the copy log; bad frames and configurations. On a Jetson it requires the `ofa` backend |
| `stereo_disparity_options` | Nothing | Every backend's parameter and size rules, `auto`'s choice for every platform, the geometry and the depth formula; also built from `tests/host/` by the hosted CI job |

[`tests/installed_consumer/`](tests/installed_consumer/) is a separate project that uses the
installed package: a graph that runs `StereoDisparityOp` on every backend the platform has and
checks it against the synthetic scene.

## Debian package

Inside the repository's CI image, with the module's folder as the Holoscan CLI project root:

```bash
holoscan --project-root modules/stereo-disparity package holoscan-stereo-disparity --language cpp \
  --pkg-generator DEB --local
```

It builds for every major GPU architecture, installs to `/usr`, and depends on
`holoscan (>= 5.0.0)` and `libnvvpi4 (>= 4.0)`.

## Licenses

Apache-2.0. VPI is under NVIDIA's own terms, installed as `/opt/nvidia/vpi4/doc/VPI_EULA.txt`.
See [NOTICE](NOTICE).
