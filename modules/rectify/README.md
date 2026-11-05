# holoscan-rectify

A Holoscan SDK 5 Module that undistorts camera images and rectifies stereo pairs, for any camera.
`UndistortOp` turns one camera's images into those of a distortion-free pinhole camera;
`StereoRectifyOp` turns a stereo pair into a rectified pair, in which a scene point lies on the
same row in both images, and sends the rectified pair's camera info, which a disparity or depth
operator takes on its `camera_info` input. The calibration is a ROS `camera_info` file per camera,
read with [holoscan-messages](../messages/); the geometry is computed once, on the CPU; every frame
is remapped on the GPU (this module's CUDA kernel) or on the VIC or the PVA (VPI 4). The
[Rectify demo](../../applications/rectify/) runs it end to end.

| | |
| --- | --- |
| CMake package | `find_package(holoscan-rectify)` |
| Targets | `holoscan::rectify` |
| Headers | `<holoscan/rectify/...>` |
| C++ namespace | `holoscan::rectify` |
| Depends on | [holoscan-messages](../messages/) 1.x (`CameraInfo`, `StereoCameraInfo`, the camera_info files) |
| Debian package | `holoscan-rectify` |
| Platforms | x86_64; aarch64 (Jetson Orin and Jetson Thor, JetPack 7.2) |

## Operators

### `UndistortOp`

| Port | Type | |
| --- | --- | --- |
| `image` (input) | `holoscan::schema::ImageT` | MONO8, MONO16, RGB8, BGR8, RGBA8 or BGRA8 of the camera's size, packed rows, in host or CUDA device memory |
| `undistorted` (output) | `holoscan::schema::ImageT` | The input's encoding and header, the output size, in CUDA device memory, with the input's capture time |
| `camera_info` (output) | `holoscan::messages::schema::CameraInfoT` | The undistorted camera (`Undistorted`, no distortion), one per image, with its capture time |

| `UndistortConfig` | Default | |
| --- | --- | --- |
| `camera_info_path` | required | The camera's ROS `camera_info` file |
| `options` | `{}` | `UndistortOptions`: the output camera, below |
| `backend` | `Backend::kCuda` | [Backends](#backends) |
| `interpolation` | `Interpolation::kLinear` | `kNearest` for MONO16 |

The options choose the output camera, as OpenCV's `getOptimalNewCameraMatrix()` does: `alpha` 0 (the default) keeps only output pixels that see the input, cutting some of the
input off; `alpha` 1 keeps every input pixel, with black around them; values in between blend the
two. An explicit `intrinsics` (and `width`, `height`) sets the output camera instead.
`UndistortOp::undistortion()` returns both cameras and the valid region, from construction on.

### `StereoRectifyOp`

| Port | Type | |
| --- | --- | --- |
| `left`, `right` (inputs) | `holoscan::schema::ImageT` | A pair with the same capture time and encoding, as for `UndistortOp` |
| `rectified_left`, `rectified_right` (outputs) | `holoscan::schema::ImageT` | Each input's encoding and header, the output size, in CUDA device memory, with the pair's capture time |
| `camera_info` (output) | `holoscan::messages::schema::StereoCameraInfoT` | The rectified pair (`Rectified`), one per pair, with its capture time |

| `StereoRectifyConfig` | Default | |
| --- | --- | --- |
| `left_camera_info_path`, `right_camera_info_path` | required | The pair's ROS `camera_info` files, as a stereo calibration writes them |
| `options` | `{}` | `StereoRectifyOptions`: `alpha`, output `width` and `height`, `zero_disparity` (on) |
| `backend` | `Backend::kCuda` | [Backends](#backends) |
| `interpolation` | `Interpolation::kLinear` | `kNearest` for MONO16 |

The rectification is Bouguet's method, with OpenCV's `stereoRectify()` conventions: each camera turns half of the way
towards the other's orientation, then both turn so that the baseline lies along x; both get the
smaller vertical focal length (shortened for barrel distortion), principal points that center the
images, and the `alpha` scaling between the rectangles inside and around both images' outlines.
`zero_disparity` (the default) gives both cameras the same `cx`. Only horizontal pairs, with the
right camera to the right of the left one, are rectified; anything else is refused when the
operator is constructed. The rotations come from the extrinsics the files hold; a pair that is
rectified already maps onto itself. `StereoRectifyOp::rectification()` returns the geometry from
construction on, so a graph can size the operators after it before it runs.

Both operators read their files when they are constructed, and again when a run starts (the SDK
constructs operators again then), so keep the files until the run has started.

### The camera info, for a disparity operator

`camera_info` is the rectified pair as a `StereoCameraInfo` of holoscan-messages: both cameras
`Rectified`, with one `fx` (= fy), `cx` each, one `cy`, the output size, no distortion; the
`baseline` in metres, the right camera's distance along +x of the left one; and
`doffs = cx_right - cx_left` (0 with `zero_disparity`). Frame ids are the files' `camera_name`s. A
disparity `d = x_left - x_right` in pixels is at depth

```text
Z = fx * baseline / (d + doffs)   metres                                 (messages::depth_from_disparity())
X = (x_left - cx_left) * Z / fx,  Y = (y - cy) * Z / fx                in the rectified left frame
```

The operator sends one with every pair, carrying the pair's capture time in its sample and its
headers, so a consumer takes it with `messages::camera_info_contract()`, either with
`CameraInfoTiming::kEveryFrame` (each pair waits for its own) or the default `kLatest`, and
`messages::LatestCameraInfo`; see [Take camera info in an
operator](../messages/README.md#take-camera-info-in-an-operator). `camera_info` must be connected:
an emit into an unconnected output fails. What the camera info does not carry is in
`rectification()`: the rectifying rotations (`rect_from_left`, `rect_from_right`, from each input
camera's frame to its rectified frame), OpenCV's `P1`, `P2` and `Q` (`disparity_to_depth`), and the
valid regions `valid_left` and `valid_right`.

### Backends

| Backend | Remap runs on | Takes | Notes |
| --- | --- | --- | --- |
| `Backend::kCuda` (default) | GPU | Every format, linear and nearest, any size | This module's kernel: one thread per output pixel, a dense float map, on the operator's CUDA stream. Never makes the CPU wait. |
| `Backend::kVpiVic` | VIC | MONO8 and MONO16; even width and height up to 16384; no nearest on Jetson Thor | VPI's remap with a warp map sampled every 4 pixels. Frees the GPU. |
| `Backend::kVpiPva` | PVA | MONO8; width a multiple of 256, even height of at least 64 | VPI's remap with a warp map sampled every 4 pixels. Frees the GPU and the VIC. |
| `Backend::kAuto` | One of the above | | Opt-in. `vpi-vic` where the VIC takes the images (gray pairs on Jetson), else `cuda`. Chosen at the first image and logged once. |

The constraints of the VIC and the PVA were measured with VPI 4.1.4 on Jetson Orin; VPI's header
lists only CPU, CUDA and VIC for remap, yet the PVA runs it on Jetson Orin and Jetson Thor. An odd
image size corrupts VPI's memory on the VIC, so the module refuses it. Jetson Thor's VIC has no
nearest interpolation; VPI refuses it when the backend is built, which fails `vpi-vic` and makes
`auto` fall back to `cuda`. Only such a refusal makes `auto` fall back; a VPI failure, such as
running out of memory, is an error on every backend.
`backend_limitation()` says why a backend cannot take some images, and `resolve_auto()` is auto's
rule as a function of `PlatformFacts`; both are in `<holoscan/rectify/backend.hpp>`.

Every backend samples with pixel centers at integer coordinates. A source within half a pixel
outside the input takes the edge pixel; further out the output is zero (the VIC and the PVA may
blend a source less than a pixel outside with zero instead). MONO16 holds depth or
other values that must not be blended, so it takes nearest interpolation only; nearest never
blends on any backend. The VIC and the PVA interpolate in fixed point and differ from the CUDA
kernel by at most 3 gray levels (linear, 0.5 on average on the VIC, 0.2 on the PVA); the VIC's
nearest picks the other neighbor near a half-pixel boundary more often (about 5 % of pixels).

The outputs are always in CUDA device memory, borrowed from the runtime's pool and sent on without
a copy. How the inputs are read, and how often the operator's thread waits (CPU waits) per pair:

| Where the images are | Read | CPU waits per pair, `cuda` | CPU waits per pair, `vpi-vic` and `vpi-pva` |
| --- | --- | --- | --- |
| CUDA device memory | In place, after their producer's work on the operator's stream | 0 | 1 |
| Pinned host memory, integrated GPU (Jetson) | In place: the GPU shares the host's memory | 1 | 1 |
| Pinned host memory, discrete GPU | Copied to the GPU, on the operator's stream | 1 | (no VIC or PVA) |
| Pageable host memory | Copied to the GPU; CUDA stages a pageable copy before it returns | 4 | 5 |

- **`vpi-vic` and `vpi-pva` wait once per pair.** Both images' copy in, remap and copy out go to one
  VPI stream that wraps the operator's CUDA stream, and VPI hands each over between the GPU and the
  VIC or PVA itself. VPI allows no other work on that CUDA stream until its own is done, and VPI
  4.1.4 has no fence the GPU could wait on instead, so the operator waits for the pair before it
  sends it. The copies are needed: VPI keeps a copy of CUDA memory for the VIC and PVA anyway, and
  its own images are faster there.
- **Pinned input costs one wait.** The runtime hands a pinned input back to its producer as soon as
  the operator returns, unlike device memory, which it holds until the operator's stream is done
  with it; so the operator waits for its stream before it returns. With `vpi-vic` and `vpi-pva`
  that is the wait it makes anyway.
- **Pageable input** waits for its producer and for each copy, on any backend. Send device images,
  or pinned ones on Jetson.

A copy to the GPU is logged once, at info level, with the bytes per frame:

```text
StereoRectifyOp (cuda backend): the images are in host memory and every backend reads CUDA device memory, so each frame is copied host to device, 1843200 bytes (1.8 MB) per frame; to avoid the copy, send CUDA device images
```

On Jetson it adds `, or pinned host images, which this GPU reads in place`. `remap_statistics()`
counts the waits and copies, for tests and profiling. The remappers also work without a graph:
`create_remapper()` builds one per camera, and a `RemapQueue` on a CUDA stream takes the remaps of
all of them until one `finish()`. A remapper submits again only once the queue of its last submit
has finished (or is gone), to the same queue or another.

## Calibration

The calibration has one format: the ROS `camera_info` YAML file, one per camera, as ROS's
`camera_calibration` writes it, read and checked by holoscan-messages
([Calibration files](../messages/README.md#calibration-files)). A stereo pair's two files carry the
extrinsics in their rectification (R) and projection (P) matrices, as a ROS stereo calibration
writes them; files of a pair that is rectified already (no distortion, R the identity) work too.

| `distortion_model` | Coefficients |
| --- | --- |
| none (`""` or `none`) | none |
| `plumb_bob` | k1, k2, p1, p2, k3 (OpenCV's default) |
| `rational_polynomial` | k1, k2, p1, p2, k3, k4, k5, k6 (OpenCV) |
| `equidistant` | k1, k2, k3, k4 (OpenCV's fisheye) |

Beyond what holoscan-messages checks, this module refuses a camera with skew, with a region of
interest or binning (give the calibration of the images as they are), or larger than 16384 pixels
a side (`check_camera()`); the two cameras of a pair must have the same size. In
`<holoscan/rectify/lens.hpp>` and `<holoscan/rectify/rectification.hpp>`, with no SDK:
`distort()` and `undistort()` map normalized points through a lens, and `compute_undistortion()`,
`compute_stereo_rectification()` and `RemapModel` are the geometry the operators use, from
`messages::CameraInfo` and `messages::StereoCameraInfo`.

## Use it in your graph

```cmake
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core sensor_io)
find_package(holoscan-rectify 1.0 REQUIRED CONFIG)  # Finds holoscan-messages too.
target_link_libraries(my_app PRIVATE holoscan::rectify)
```

```cpp
#include <holoscan/rectify/stereo_rectify_op.hpp>

namespace rc = holoscan::rectify;

auto rectify = graph.op<rc::StereoRectifyOp>(
    "rectify", rc::StereoRectifyConfig{.left_camera_info_path = "left.yaml",
                                       .right_camera_info_path = "right.yaml",
                                       .options = {.alpha = 0.0},
                                       .backend = rc::Backend::kAuto});
graph.add_flow(camera->left, rectify->left);
graph.add_flow(camera->right, rectify->right);
graph.add_flow(rectify->rectified_left, disparity->left);
graph.add_flow(rectify->rectified_right, disparity->right);
graph.add_flow(rectify->camera_info, disparity->camera_info);

// The rectified images are in device memory.
holoscan::CompileOptions options;
for (const char* port : {"rectified_left", "rectified_right"}) {
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "rectify", .output_port = port, .device = holoscan::DeviceId{0}});
}
```

Keep a lossless chain on one worker, as the demo does: `graph.partition("name").include_all()`
runs the whole graph on one. A graph with a second source, or one that forks or joins (a camera
feeding two rectifiers, a viewer taking the raw and the rectified pair), is split by the SDK's
automatic partitioning onto several workers. Across workers, the SDK 5.0 runs a producer even while
its consumer's lossless queue is full, and refuses what it sends: an emit fails with
`kBackpressured`, and an allocation with `kResourceExhausted` while every slot of the output's pool
is still held. A source can treat both as backpressure and send again on its next tick, but
`StereoRectifyOp` and `UndistortOp` have taken their input by then and return the error, which
fails the run.

`holoscan-rectify_WITH_VPI` tells whether the installed module has the `vpi-*` backends, and
`available_backends()` lists them at run time. The remap backends also work without a graph:
`create_remapper()` and `RemapQueue` in `<holoscan/rectify/remapper.hpp>`.
`<holoscan/rectify/synthetic_stereo.hpp>`
renders a distorted stereo pair of a plane of dots with known geometry and measures each dot in
rectified images, for tests and demos; `messages::write_stereo_camera_info_yaml()` writes its
calibration as the files the operator reads.

## Accuracy and performance

The geometry is exact to rounding: rectified projections of random points share a row to 1e-12
px, and the maps send each rectified pixel to the raw pixel of the same point to 1e-12 px. The
VIC's and PVA's 4-pixel warp grid moves a source by at most 0.003 px at 1280x720. Measured in the
images, on the synthetic pair (plumb_bob and equidistant, 1280x720, 275 dots a pair), each dot lies
on the same row in both rectified images to 0.05 px (0.009 px RMS) on cuda and the PVA,
and to 0.085 px on the VIC; and where the geometry puts it to 0.05 px, 0.11 px on the VIC. That is
about the precision of measuring a dot, not of the remap. Undistortion puts dots where they belong
to 0.1 px. The local recordings, which the camera has rectified already, come out unchanged.

Time for a 1280x720 image, and for a pair (both images through one queue and one `finish()`, as
`StereoRectifyOp` does), from submit to the result on the operator's stream, MONO8 / RGBA8:

| Backend | RTX 5090 | Jetson Orin | Jetson Thor |
| --- | --- | --- | --- |
| `cuda`, image | 0.008 / 0.010 ms | 0.28 / 0.42 ms | 0.08 / 0.10 ms |
| `cuda`, pair | 0.012 / 0.016 ms | 0.54 / 0.82 ms | 0.11 / 0.16 ms |
| `vpi-vic`, image | | 2.0 ms | 4.5 ms |
| `vpi-vic`, pair | | 7.3 ms | 4.5 ms |
| `vpi-pva`, image | | 7.2 ms | 2.4 ms |
| `vpi-pva`, pair | | 12.2 ms | 4.4 ms |

The times include a stream synchronization and vary with the clocks and the load: the Jetson Orin
column is the median of seven runs on a board other work shared, and its VIC and PVA times vary
by a millisecond or two; the RTX 5090 was measured with little other load (a GPU busy with other
processes gives each frame a time slice of several milliseconds instead). The VIC and PVA are
slower than the GPU but leave it free for other work; on Jetson Thor the two images of a pair
overlap on them.

## Requirements

- Holoscan SDK 5.0 (Core and Sensor I/O; Testing for the tests), CUDA 13, CMake 4.0 or later and a
  C++20 compiler.
- [holoscan-messages](../messages/) 1.x: its installed package on `CMAKE_PREFIX_PATH`, or its
  `holoscan::messages` target in the same build (the repository root adds it first).
- Optional: NVIDIA VPI 4 (`vpi4-dev`) for the `vpi-*` backends. The repository's CI images install
  it from NVIDIA's Jetson package repository; see [`.github/docker/`](../../.github/docker/).

## Build, test and install

From the repository root, in the repository's CI image or with the requirements above, after
installing holoscan-messages to `install/`:

```bash
cmake -S modules/rectify -B build/rectify -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DCMAKE_PREFIX_PATH="$PWD/install"
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
| `rectify_camera_info` | Nothing | A raw pair rectifies the same from its `camera_info` files as from its values, for every lens model; files of a rectified pair map onto themselves; the rectified pair's camera info is valid and its own files give it back; the cameras this module refuses |
| `rectify_geometry` | Nothing | Distortion round trips for every model (1e-9); alpha 0 and 1; Bouguet rectification of pairs with known R and t, including a 7-degree rotation and a tilted baseline: rows, maps and depth exact; an already rectified pair maps to itself; synthetic images undistorted and rectified on the CPU; the sparse warp grid's error, and across the border of a wide view, that it samples nothing where the dense map does not |
| `rectify_backend_rules` | Nothing | What every backend takes, and auto's choice on x86_64, Jetson and without VPI |
| `rectify_remap_backends` | GPU | Every backend this machine runs against the CPU reference for every format and interpolation (nearest never blends); a wide view (alpha 1) on every backend, black wherever the source is outside the input, the same from packed and padded rows through a second queue; rectified rows and undistorted positions on the synthetic pair on every backend; auto's choice (on the CI runners, `vpi-vic` on Jetson and `cuda` on x86_64); on a Jetson runner, the VIC and the PVA; refusals, a remapper submitted again before the queue of its last submit finished, and rows too short or misaligned; a cuda remap after vpi-* work in one queue waits for it; which VPI statuses are refusals; CPU waits per pair through one queue; time per image and per pair |
| `rectify_operators` | GPU | Both operators through the SDK's operator test harness, from `camera_info` files: host and device input, every backend and auto, the camera info sent with every pair and image with its capture time, RGB8 and MONO16, capture time and header, and refused input, files and configuration |
| `rectify_pipeline` | GPU | `StereoRectifyOp` in a running graph whose next operator reads each rectified pair on the GPU as soon as it arrives, on its own stream, and takes its camera info with the `kEveryFrame` contract; the dots move from pair to pair, so a remap not finished when the pair was sent, or a stale output, fails it. Every backend, from device, pinned and pageable host memory, with the CPU waits and copies per pair of the table above |
| `rectify_real_data` | GPU, `$HOLOSCAN_TEST_DATA` | The local recordings, already rectified, from their `left.yaml` and `right.yaml`: the geometry keeps their camera, the maps are the identity, the images come out unchanged, and the camera info is the files' pair. Skipped when `$HOLOSCAN_TEST_DATA` is unset; fails when it names no recording with a `left.yaml`. |

The first three are also built on their own from `tests/host/` by the hosted CI job.
[`tests/installed_consumer/`](tests/installed_consumer/) is a separate project that uses the
installed package: it writes the synthetic pair's `camera_info` files, rectifies the pair from
them, and checks its rows and camera info in an operator that takes both as a disparity operator
would.

## Debian package

Inside the repository's CI image, with the module's folder as the Holoscan CLI project root:

```bash
holoscan --project-root modules/rectify package holoscan-rectify --language cpp \
  --pkg-generator DEB --local
```

It writes `holoscan-rectify_1.0.0_<arch>.deb`, which installs to `/usr` and depends on
`holoscan (>= 5.0.0)`, `holoscan-messages (>= 1.0.0)` and, when built with VPI, `libnvvpi4`.

## Licenses

Apache-2.0. The output cameras follow the methods OpenCV implements (Apache-2.0); no OpenCV code is
included. VPI is under NVIDIA's own terms, installed as `/opt/nvidia/vpi4/doc/VPI_EULA.txt`. See
[NOTICE](NOTICE).
