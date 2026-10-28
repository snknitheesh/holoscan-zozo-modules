# holoscan-rectify

A Holoscan SDK 5 Module that undistorts camera images and rectifies stereo pairs, for any camera.
`UndistortOp` turns one camera's images into those of a distortion-free pinhole camera;
`StereoRectifyOp` turns a stereo pair into a rectified pair, in which a scene point lies on the
same row in both images, and sends the rectified geometry a disparity or depth operator needs. The
geometry is computed once, on the CPU, from the calibration; every frame is remapped on the GPU
(this module's CUDA kernel, or VPI 4), the VIC or the PVA. The
[Rectify demo](../../applications/rectify/) runs it end to end.

| | |
| --- | --- |
| CMake package | `find_package(holoscan-rectify)` |
| Targets | `holoscan::rectify` |
| Headers | `<holoscan/rectify/...>` |
| C++ namespace | `holoscan::rectify` |
| Debian package | `holoscan-rectify` |
| Platforms | x86_64; aarch64 (Jetson Orin and Thor, JetPack 7.2) |

## Operators

### `UndistortOp`

| Port | Type | |
| --- | --- | --- |
| `image` (input) | `holoscan::schema::ImageT` | MONO8, MONO16, RGB8, BGR8, RGBA8 or BGRA8 of the camera's size, packed rows, in host or CUDA device memory |
| `undistorted` (output) | `holoscan::schema::ImageT` | The input's encoding and header, the output size, in CUDA device memory, with the input's capture time |

`UndistortConfig` holds the `CameraModel`, the `UndistortOptions`, the backend and the
interpolation. The options choose the output camera, as OpenCV's `getOptimalNewCameraMatrix()`
does: `alpha` 0 (the default) keeps only output pixels that see the input, cutting some of the
input off; `alpha` 1 keeps every input pixel, with black around them; values in between blend the
two. An explicit `intrinsics` (and `width`, `height`) sets the output camera instead.
`UndistortOp::undistortion()` returns both cameras and the valid region, from construction on.

### `StereoRectifyOp`

| Port | Type | |
| --- | --- | --- |
| `left`, `right` (inputs) | `holoscan::schema::ImageT` | A pair with the same capture time and encoding, as for `UndistortOp` |
| `rectified_left`, `rectified_right` (outputs) | `holoscan::schema::ImageT` | Each input's encoding and header, the output size, in CUDA device memory, with the pair's capture time |
| `geometry` (output, with `geometry_output`) | `holoscan::rectify::schema::StereoGeometryT` | One message per pair, with its capture time |

`StereoRectifyConfig` holds the `StereoCalibration` (both cameras and `right_from_left`), the
`StereoRectifyOptions`, the backend, the interpolation and `geometry_output`. The rectification is
Bouguet's method, with OpenCV's `stereoRectify()` conventions: each camera turns half of the way
towards the other's orientation, then both turn so that the baseline lies along x; both get the
smaller vertical focal length (shortened for barrel distortion), principal points that center the
images, and the `alpha` scaling between the rectangles inside and around both images' outlines.
`zero_disparity` (the default) gives both cameras the same `cx`. Only horizontal pairs, with the
right camera to the right of the left one, are rectified; anything else is refused when the
operator is constructed. `StereoRectifyOp::rectification()` returns the geometry from construction
on, so a graph can configure the operators after it before it runs.

### The geometry, for a disparity operator

A rectified pair has `fx` (= fy), `cx_left`, `cx_right`, `cy`, the output `width` and `height`,
and the `baseline` in metres, the right camera's distance along +x from the left one. A disparity
`d = x_left - x_right` in pixels is at depth

```text
Z = fx * baseline / (d + doffs)   metres,   doffs = cx_right - cx_left   (0 with zero_disparity)
X = (x_left - cx_left) * Z / fx,  Y = (y - cy) * Z / fx                in the rectified left frame
```

These are Middlebury's conventions. A disparity operator needs `fx`, `baseline` and `doffs` for
depth, `cx_left` and `cy` for points, and the size; `valid_left` and `valid_right` say which
pixels see the input. The `geometry` message ([schema](schemas/stereo_geometry.fbs)) carries them
all, with OpenCV's `Q` matrix (`disparity_to_depth`), the rectifying rotations (from each input
camera's frame to its rectified frame, which a consumer needs to bring points back into a raw
camera's frame) and the `backend` that remapped the pair. Connect it, with the rectified images, to
an operator that triggers on all three with `OnAligned`; or read `rectification()` and configure
the operator once. `StereoRectification::depth_from_disparity()` is `fx * baseline`.

### Backends

| Backend | Remap runs on | Takes | Notes |
| --- | --- | --- | --- |
| `Backend::kCuda` (default) | GPU | Every format, linear and nearest, any size | This module's kernel: one thread per output pixel, a dense float map, on the operator's CUDA stream. |
| `Backend::kVpiCuda` | GPU | Every format, linear and nearest, at least 64x16 | VPI's remap with a dense warp map, on the operator's stream, through wrappers of its images. |
| `Backend::kVpiVic` | VIC | MONO8 and MONO16; even width and height up to 16384; no nearest on Thor | VPI's remap with a warp map sampled every 4 pixels. Frees the GPU. |
| `Backend::kVpiPva` | PVA | MONO8; width a multiple of 256, even height of at least 64 | VPI's remap with a warp map sampled every 4 pixels. Frees the GPU and the VIC; slower than the VIC. |
| `Backend::kAuto` | One of the above | | Opt-in. `vpi-vic` where the VIC takes the images (gray pairs on Jetson), else `cuda`. Chosen at the first image and logged once. |

The constraints of the VIC and the PVA were measured with VPI 4.1.4 on Jetson Orin; VPI's header
lists only CPU, CUDA and VIC for remap, yet the PVA runs it there. An odd image size corrupts VPI's
memory on the VIC, so the module refuses it. Thor's VIC has no nearest interpolation; VPI refuses
it when the backend is built, which fails `vpi-vic` and makes `auto` fall back to `cuda`.
`backend_limitation()` says why a backend cannot take some images, and `resolve_auto()` is auto's
rule as a function of `PlatformFacts`; both are in `<holoscan/rectify/backend.hpp>`.

Every backend samples with pixel centers at integer coordinates. A source within half a pixel
outside the input takes the edge pixel; further out the output is zero. MONO16 holds depth or
other values that must not be blended, so it takes nearest interpolation only; nearest never
blends on any backend. The VIC and the PVA interpolate in fixed point and differ from the CUDA
kernel by at most 2 gray levels (linear); the VIC's nearest picks the other neighbor near a
half-pixel boundary more often (about 5 % of pixels).

| Where the images are | What happens |
| --- | --- |
| CUDA device memory | Read in place, after their producer's work on the operator's stream. `vpi-vic` and `vpi-pva` copy them into VPI's images and the result out, on the GPU: VPI keeps a copy of CUDA memory for the VIC and PVA anyway, and its own images are faster there. |
| Host memory | Copied to the GPU every frame. The operator says so once, at info level, with the bytes per frame. |

```text
StereoRectifyOp (cuda backend): the images are in host memory and every backend reads CUDA device memory, so each frame is copied host to device, 1843200 bytes (1.8 MB) per frame; to avoid the copy, send CUDA device images
```

## Calibration

`CameraModel` is one camera: size, `fx`, `fy`, `cx`, `cy` (pixels, pixel centers at integers) and
a distortion model with its coefficients. `StereoCalibration` is two cameras and
`right_from_left` (`p_right = rotation * p_left + translation`, row-major, metres); its fields are
those of holoscan-stereo-camera's calibration, so one converts field by field.

| Model | Coefficients | Also read as |
| --- | --- | --- |
| `kNone` | none | `none` |
| `kBrownConrady` | k1, k2, p1, p2, k3 (OpenCV's plumb bob) | `brown_conrady`, `plumb_bob` |
| `kRationalPolynomial` | k1, k2, p1, p2, k3, k4, k5, k6 (OpenCV) | `rational_polynomial` |
| `kKannalaBrandt4` | k1, k2, k3, k4 (equidistant fisheye, OpenCV's fisheye model) | `kannala_brandt4`, `equidistant`, `fisheye` |

Two loaders, both validating what they read:

- `read_stereo_calibration(path)` reads a recording's `calibration.json`, as holoscan-stereo-camera
  writes it: `left` and `right` (`width`, `height`, `fx`, `fy`, `cx`, `cy`, `distortion` with
  `model` and `coefficients`) and `right_from_left` (`rotation`, `translation`). Other keys are
  ignored; librealsense's `modified_brown_conrady`, `inverse_brown_conrady` and `ftheta` models
  are refused by name.
- `read_camera_info(path)` reads a ROS `camera_info` YAML file (`image_width`, `image_height`,
  `camera_matrix`, `distortion_model`, `distortion_coefficients`, `rectification_matrix`,
  `projection_matrix`). `stereo_from_camera_info(left, right)` recovers the stereo calibration from
  the pair of files a ROS stereo calibration writes.

`distort()` and `undistort()` map normalized points through a lens; `compute_undistortion()`,
`compute_stereo_rectification()` and `RemapModel` are the geometry the operators use, in
`<holoscan/rectify/rectification.hpp>`, with no SDK.

## Use it in your graph

```cmake
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core sensor_io)
find_package(holoscan-rectify 1.0 REQUIRED CONFIG)
target_link_libraries(my_app PRIVATE holoscan::rectify)
```

```cpp
#include <holoscan/rectify/stereo_rectify_op.hpp>

namespace rc = holoscan::rectify;

auto rectify = graph.op<rc::StereoRectifyOp>(
    "rectify", rc::StereoRectifyConfig{
                   .calibration = rc::read_stereo_calibration("calibration.json"),
                   .options = {.alpha = 0.0},
                   .backend = rc::Backend::kAuto,
                   .geometry_output = true});
graph.add_flow(camera->left, rectify->left);
graph.add_flow(camera->right, rectify->right);
graph.add_flow(rectify->rectified_left, disparity->left);
graph.add_flow(rectify->rectified_right, disparity->right);
graph.add_flow(rectify->geometry, disparity->geometry);

// The rectified images are in device memory.
holoscan::CompileOptions options;
for (const char* port : {"rectified_left", "rectified_right"}) {
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "rectify", .output_port = port, .device = holoscan::DeviceId{0}});
}
```

An emit into an unconnected output fails, so `geometry` exists only with `geometry_output`.
`holoscan-rectify_WITH_VPI` tells whether the installed module has the `vpi-*` backends, and
`available_backends()` lists them at run time. The remap backends also work without a graph:
`create_remapper()` in `<holoscan/rectify/remapper.hpp>`. `<holoscan/rectify/synthetic_stereo.hpp>`
renders a distorted stereo pair of a plane of dots with known geometry and measures each dot in
rectified images, for tests and demos.

## Accuracy and performance

The geometry is exact to rounding: rectified projections of random points share a row to 1e-12
px, and the maps send each rectified pixel to the raw pixel of the same point to 1e-12 px. The
VIC's and PVA's 4-pixel warp grid moves a source by at most 0.003 px at 1280x720. Measured in the
images, on the synthetic pair (Brown-Conrady and fisheye, 1280x720, about 500 dots), each dot lies
on the same row in both rectified images to 0.05 px (0.015 px RMS) on every backend, the VIC's
0.08 px, and where the geometry puts it to 0.1 px; that is the precision of measuring a dot, not of
the remap. Undistortion puts dots where they belong to 0.1 px. The RealSense recordings, which the
camera has rectified already, come out unchanged.

Time per 1280x720 frame, from submit to the result on the operator's stream, mean of 100 frames:

| Backend | x86_64, RTX 5090, MONO8 / RGBA8 | Jetson Orin, MONO8 / RGBA8 |
| --- | --- | --- |
| `cuda` | 0.15 / 0.84 ms | 0.22 / 0.15 ms |
| `vpi-cuda` | 0.51 / 0.47 ms | 0.36 / 0.90 ms |
| `vpi-vic` | | 5.3 ms |
| `vpi-pva` | | 7.8 ms |

The GPU times include a stream synchronization and vary with the GPU's clocks and load; the VIC's
and PVA's include VPI's synchronization and the copies into and out of VPI's images. The VIC and
PVA are slower than the GPU but leave it free for other work.

## Requirements

- Holoscan SDK 5.0 (Core and Sensor I/O; Testing for the tests), CUDA 13, CMake 4.0 or later and a
  C++20 compiler.
- Optional: NVIDIA VPI 4 (`vpi4-dev`) for the `vpi-*` backends. The repository's CI images install
  it from NVIDIA's Jetson package repository; see [`.github/docker/`](../../.github/docker/).

## Build, test and install

From the repository root, in the repository's CI image or with the requirements above:

```bash
cmake -S modules/rectify -B build/rectify -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build/rectify
ctest --test-dir build/rectify --output-on-failure
cmake --install build/rectify --prefix "$PWD/install"
```

| CMake option | Default | |
| --- | --- | --- |
| `HOLOSCAN_RECTIFY_WITH_VPI` | `ON` | Build the `vpi-*` backends when VPI 4 is found |
| `BUILD_TESTING` | `OFF` | Build the tests, when the module is built on its own |
| `HOLOSCAN_RECTIFY_BUILD_TESTING` | `OFF` | Build the tests inside another project |
| `CMAKE_CUDA_ARCHITECTURES` | `native` (`all-major` for the package) | GPUs to compile for, unless already set |

Another project can also build the module as part of its own build, with `add_subdirectory()` or
`FetchContent`, and link `holoscan::rectify` without `find_package()`. The module then leaves the
including project's `BUILD_TESTING` and `BUILD_SHARED_LIBS` alone.

The tests:

| Test | Needs | Checks |
| --- | --- | --- |
| `rectify_calibration` | Nothing | `calibration.json` and `camera_info` YAML, what each refuses, and a stereo pair recovered from two `camera_info` files |
| `rectify_geometry` | Nothing | Distortion round trips for every model (1e-9); alpha 0 and 1; Bouguet rectification of pairs with known R and t, including a 7-degree rotation and a tilted baseline: rows, maps and depth exact; an already rectified pair maps to itself; synthetic images undistorted and rectified on the CPU; the sparse warp grid's error |
| `rectify_backend_rules` | Nothing | What every backend takes, and auto's choice on x86_64, Jetson and without VPI |
| `rectify_remap_backends` | GPU | Every backend this machine runs against the CPU reference for every format and interpolation (nearest never blends); rectified rows and undistorted positions on the synthetic pair on every backend; auto's choice (on the CI runners, `vpi-vic` on Jetson and `cuda` on x86_64); refusals; time per frame |
| `rectify_operators` | GPU | Both operators through the SDK's operator test harness: host and device input, every backend and auto, the geometry message, RGB8 and MONO16, capture time and header, and refused input and configuration |
| `rectify_real_data` | GPU, `$HOLOSCAN_TEST_DATA` | The RealSense recordings, already rectified: the geometry keeps their camera, the maps are the identity, and the images come out unchanged. Skipped without the data. |

The first three are also built on their own from `tests/host/` by the hosted CI job.
[`tests/installed_consumer/`](tests/installed_consumer/) is a separate project that uses the
installed package: a small graph that rectifies the synthetic pair and checks its rows and geometry
message.

## Debian package

Inside the repository's CI image, with the module's folder as the Holoscan CLI project root:

```bash
holoscan --project-root modules/rectify package holoscan-rectify --language cpp \
  --pkg-generator DEB --local
```

It writes `holoscan-rectify_1.0.0_<arch>.deb`, which installs to `/usr` and depends on
`holoscan (>= 5.0.0)` and, when built with VPI, `libnvvpi4`.

## Licenses

Apache-2.0. The output cameras follow the methods OpenCV implements (Apache-2.0); no OpenCV code is
included. VPI is under NVIDIA's own terms, installed as `/opt/nvidia/vpi4/doc/VPI_EULA.txt`. See
[NOTICE](NOTICE).
