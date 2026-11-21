# holoscan-stitching

A Holoscan SDK 5 Module that stitches the images of several cameras into one image on the GPU: a
**cylindrical or spherical panorama** seen from a point of the rig, or a **bird's-eye surround
view** of the ground, as cars and mobile robots show it. `StitchOp` takes one image per camera and
each camera's calibration as a `CameraInfo` message of [holoscan-messages](../messages/), read from
its ROS `camera_info` file, and emits the stitched image. The remap tables are built on the GPU once
every camera's calibration has arrived, and again when one changes; each frame is then sampled
bilinearly and blended, with distance-weighted feathering or hard seams and optional exposure
compensation. The [Stitching demo](../../applications/stitching/) runs it end to end on four
synthetic fisheye cameras.

| | |
| --- | --- |
| CMake package | `find_package(holoscan-stitching)` |
| Target | `holoscan::stitching` |
| Headers | `<holoscan/stitching/...>` |
| C++ namespace | `holoscan::stitching` |
| Debian package | `holoscan-stitching` |
| Depends on | [holoscan-messages](../messages/) 1.x |
| Platforms | x86_64; aarch64 (Jetson Thor and Jetson Orin, JetPack 7.2) |

## `StitchOp`

| Port | Type | |
| --- | --- | --- |
| `image_0` ... `image_<N-1>` (inputs) | `holoscan::schema::ImageT` | One per camera, in the order of `camera_from_rig`: MONO8, RGB8, BGR8, RGBA8 or BGRA8 of the size its camera info gives, `uint8` `[H, W, C]` (or `[H, W]` for MONO8), rows possibly padded, in host or CUDA device memory |
| `camera_info_0` ... `camera_info_<N-1>` (inputs) | `holoscan::messages::schema::CameraInfoT` | Mandatory: camera i's calibration, from a `CameraInfoSourceOp` reading its `camera_info` file or from an operator upstream |
| `stitched` (output) | `holoscan::schema::ImageT` | RGB8 or RGBA8 `[H, W, C]` of the output's size, in CUDA device memory |

A rig has 1 to 8 cameras (`kMaxCameras`). The SDK has no variable-length port list, so the operator
holds fixed arrays of eight inputs, `images` and `camera_info`, and declares only the first N, named
`image_<i>` and `camera_info_<i>` (as `HolovizOp` does for its layers); connect `camera->image` to
`stitch->images[i]` and its calibration to `stitch->camera_info[i]`. An unconnected camera info
input fails `compile()`. With two or more cameras, the operator runs once per frame set: when an
image with the same capture time has arrived on every input, or within `sync_window` of each other.
The output carries the earliest capture time of its set and the set's frame id when they share one;
its header's `frame_id` is the configuration's (`"rig"`). With RGBA8 output, alpha is 255 where a
camera sees the pixel and 0 elsewhere.

`StitchConfig`:

| Field | Default | |
| --- | --- | --- |
| `camera_from_rig` | none | Each camera's pose on the rig, below; 1 to 8, which sets N |
| `output` | bird's-eye, 1024 x 1024, 20 m x 20 m | `OutputGeometry`: the projection, the size and its parameters, below |
| `blend` | feathering, RGBA8 | `BlendSettings`, below |
| `sync_window` | `0` | The largest spread of one frame set's capture times; keep it below half the frame period |
| `camera_info` | `kLatest`, 10 s | How the camera info arrives (`messages::CameraInfoInputOptions`), below |
| `frame_id` | `"rig"` | The output header's frame |

The camera info follows the messages module's [consumer
contract](../messages/README.md#take-camera-info-in-an-operator), for N cameras: with `kLatest`
(the default) each camera's calibration may come at any rate, and each frame set uses the newest of
each received before it (keep `max_age` above the senders' period; connect each with
`messages::latest_camera_info_connection()`); with `kEveryFrame` each camera sends its calibration
with every frame, carrying the frame's capture time, and the operator aligns the two. Frame sets
before every camera's calibration has arrived are skipped (the operator says so once). Each new
calibration is checked; one the stitcher cannot use (skew, a region of interest, binning, a lens
that folds back on its axis, or one `messages::validate()` refuses) fails the run, naming its port.

A wider `sync_window` than half the frame period could pair an image of one frame set with its
neighbours of the next when one is dropped; the operator fails the run when a frame set begins
within twice the window of the previous one, rather than stitch two moments together.

When the tables are built (the first frame set with every calibration, and after a change), the
operator logs their size, and warns about a camera that sees none of the output (its images are
received and ignored). Host images are copied to the GPU every frame, and the operator says so once:

```text
StitchOp: images arrive in host memory and the stitcher reads CUDA device memory, so each frame set is copied host to device, 11059200 bytes (11.1 MB) per frame; to avoid the copy, send CUDA device images
```

### CPU waits and copies per frame set

| Images in | CPU waits | Copies |
| --- | ---: | ---: |
| CUDA device memory | 0 | 0 |
| Pinned host memory whose allocator holds it until the operator's stream is done (the runtime's pools) | 0 | N host to device |
| Other pinned host memory | 1, after all N copies are queued | N host to device |
| Pageable host memory | N: each image waits for its producer; each copy returns once the CUDA runtime has staged it | N host to device |

The exposure gains are solved on the GPU, with no read back. Building the tables waits for the GPU
once per calibration (the first, and each change), not per frame. A host image larger than the last
waits once, to grow its device buffer.

## Calibration

Each camera's calibration has one source: its ROS `camera_info` file, as ROS's `camera_calibration`
writes it, read by the messages module (`CameraInfoSourceOp`, or `read_camera_info_yaml()`) and sent
to `StitchOp` as `CameraInfo`. The stitcher takes its image size, its intrinsics `fx`, `fy`, `cx`,
`cy` (pixel centres at integers) and its lens model:

| `distortion_model` | Coefficients | |
| --- | --- | --- |
| `none` | | A pinhole |
| `plumb_bob` | k1, k2, p1, p2, k3 | A pinhole with radial and tangential distortion |
| `rational_polynomial` | k1, k2, p1, p2, k3, k4, k5, k6 | The same, with a rational radial term |
| `equidistant` | k1, k2, k3, k4 | A fisheye, which sees beyond 90 degrees from its axis |

It refuses a camera with skew, a region of interest or binning (the intrinsics must describe the
images as they are), and does not apply a rectification: the pose is that of the frame the
intrinsics describe. Where a lens' distortion polynomial folds back (strong barrel distortion near
the corners of a wide lens), two rays would land on one pixel; `max_valid_angle()` finds the angle
from the axis where the projection stops moving outwards, and the stitcher ignores rays beyond it,
as it ignores rays behind a pinhole camera.

A `camera_info` file has no pose, so each camera's place on the rig is in `StitchConfig`:
`camera_from_rig[i]`, a `messages::RigidTransform` (`p_camera = R p_rig + t`, row-major, metres),
for the camera on `image_<i>` and `camera_info_<i>`. The rig frame is x forward, y left, z up (ROS
REP 103); camera frames are optical (x right, y down, z forward). `mount(position, yaw, pitch,
roll)` makes the pose of a camera at `position`, turned left by `yaw`, tilted up by `pitch`
(negative looks down) and rolled by `roll`, in degrees; `inverse()` and `compose()` combine poses,
for example with a stereo pair's `right_from_left` from its two files
(`read_stereo_camera_info_yaml()`).

Without a graph, the `Stitcher` and the CPU reference take a `CameraRig`
(`<holoscan/stitching/camera_rig.hpp>`): a `Camera` per camera, its `CameraInfo` and its
`camera_from_rig`.

## Projections

Every output pixel is a 3D point of the rig frame, and every camera that sees it contributes
(`<holoscan/stitching/projection.hpp>`):

| Projection | Columns | Rows | Parameters |
| --- | --- | --- | --- |
| `kCylindrical` | azimuth | the tangent of the elevation: vertical lines stay straight | `PanoramaParams` |
| `kSpherical` | azimuth | elevation (equirectangular) | `PanoramaParams` |
| `kBirdsEye` | -y (the rig's right is right) | x (forward is up) | `BirdsEyeParams` |

`PanoramaParams` gives the azimuth of the middle column (0 along +x, 90 along +y) and the width in
degrees (up to 360, where the image wraps around), the elevations of the bottom and top edges, the
point it is seen from (`center`), and `radius`, where the scene is assumed to be: at 0, at
infinity, so only each camera's rotation counts (right for cameras that share a centre); above 0, on
the cylinder or sphere of that radius, so cameras away from the centre see that surface without
parallax. `BirdsEyeParams` gives the ground area in metres, on the ground plane z = 0;
`birds_eye_view(area, metres_per_pixel)` sizes the image for a resolution. `output_point()` and
`output_pixel()` map between pixels and rig points.

A panorama of cameras that do not share a centre is exact only for the surface it assumes; things
nearer or further are seen with parallax, doubled in the overlaps. A bird's-eye view is exact for
the ground; anything above it (another car, a wall) is smeared away from the cameras.

## Remap tables

For each camera, the table covers the smallest rectangle of the output that the camera sees (across
the right edge of a 360-degree panorama when it straddles it) and holds, per pixel, the camera pixel
`(u, v)` (float) and the **margin**, the distance in camera pixels to where the camera stops seeing:
its image border or the edge of its lens' valid range. 12 bytes per pixel the camera sees. The
tables are built on the GPU in double precision, in two passes (coverage, then the tables over each
camera's rectangle). `build_remap_tables()` builds the same on the CPU; the two agree to 1e-3 px.

## Blending

Per frame, one CUDA thread per output pixel reads each camera's table entry, samples the camera
image bilinearly (taps clamped to the image), and blends:

| `BlendSettings::mode` | |
| --- | --- |
| `kFeather` (default) | Every camera that sees the pixel contributes, weighted by its margin, so the seam fades over the whole overlap (distance-weighted feathering). `feather_width` caps the weight: a camera reaches full weight that many pixels from its edge, and between those ramps the cameras count equally. |
| `kHardSeam` | Each pixel comes from the camera with the largest margin: a seam halfway through each overlap. |

**Exposure compensation** (`exposure_compensation`) scales each camera by a gain, solved on the GPU
every frame from the overlaps' mean brightness (gain compensation over the overlap means): at every
4th output pixel each way, every pair of cameras that both see it adds its luminance to that
pair's sum; one thread then solves the normal equations of

```text
E = sum over pairs (i, j) of N_ij * [ (g_i I_ij - g_j I_ji)^2 / sigma_N^2 + (1 - g_i)^2 / sigma_g^2 ]
```

for the gains `g`, where `I_ij` is camera i's mean over its overlap with camera j and `N_ij` the
overlap's samples. `exposure_sigma_noise` (`sigma_N`, 10 levels) and `exposure_sigma_gain`
(`sigma_g`) weigh matching the overlaps against keeping the gains near 1: with `sigma_g = 0.1`,
overlaps of mean brightness near 100 keep about a third of an exposure difference; the default 0.5
leaves about 2%. `exposure_smoothing` blends each frame's gains with the
previous frame's, against flicker. The sums are integers, so the gains do not depend on the order
of the GPU's additions. A camera with no overlap keeps gain 1. `Stitcher::gains()` returns them.

`background` is the colour where no camera sees; `output_format` RGB8 or RGBA8.

## Use it in your graph

```cmake
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core sensor_io)
find_package(holoscan-messages 1.0 REQUIRED CONFIG)
find_package(holoscan-stitching 1.0 REQUIRED CONFIG)
target_link_libraries(my_app PRIVATE holoscan::stitching holoscan::messages)
```

```cpp
#include <holoscan/messages/camera_info_source_op.hpp>
#include <holoscan/stitching/stitch_op.hpp>

namespace hm = holoscan::messages;
namespace st = holoscan::stitching;

// Four surround cameras, each calibrated in its own camera_info file, and
// where each is mounted.
const std::array<std::string, 4> names{"front", "left", "rear", "right"};
const std::vector<st::RigidTransform> poses{
    st::mount({2.0, 0.0, 0.8}, 0.0, -30.0), st::mount({0.0, 1.0, 1.0}, 90.0, -40.0),
    st::mount({-2.0, 0.0, 0.8}, 180.0, -30.0), st::mount({0.0, -1.0, 1.0}, -90.0, -40.0)};

st::StitchConfig surround{.camera_from_rig = poses,
                          .output = st::birds_eye_view({-8.0, 8.0, -6.0, 6.0}, 0.02)};
surround.blend.exposure_compensation = true;
auto bev = graph.op<st::StitchOp>("bev", surround);

for (std::size_t i = 0; i < names.size(); ++i) {
  auto info = graph.op<hm::CameraInfoSourceOp>(
      names[i] + "_info", hm::CameraInfoSourceConfig{.camera_info_path = names[i] + ".yaml"});
  graph.add_flow(info->camera_info, bev->camera_info[i], hm::latest_camera_info_connection());
  graph.add_flow(cameras[i]->image, bev->images[i]);  // Output<holoscan::schema::ImageT>
}
```

A second `StitchOp` (a panorama, say) takes the same images and the same `CameraInfoSourceOp`s:
connect each output to both.

Bind the outputs (`bev/stitched`) to a CUDA device in the `CompileOptions`, as for any CUDA output.
For live cameras, connect with `QueuePolicy::kDropOldest` and depth 1, and give `sync_window` the
cameras' trigger skew. A chain whose every frame must arrive (`kBlocking` edges, as in tests) runs on
one worker, `graph.partition(name).include_all()`: the camera info sources are sources of their own,
which the automatic partitioning would give workers of their own (see the messages module's ["Run a
graph on one worker"](../messages/README.md#run-a-graph-on-one-worker)).

Without a graph, `Stitcher` (`<holoscan/stitching/stitcher.hpp>`) stitches device images on a
CUDA stream, and `reference_stitch()` with `build_remap_tables()` (`<holoscan/stitching/remap.hpp>`)
on the CPU. `<holoscan/stitching/synthetic_world.hpp>` has the tests' world: the surround and ring
rigs, GPU renders of their cameras and of the ground truth, the checkerboards' corners and
`compare_with_truth()`.

## Accuracy

On the synthetic world (a checkered ground plane and a checkered cylindrical wall around the rig),
rendered by four 1280x720 fisheye cameras (`surround_rig()`, about 215 degrees across) and by rings
of pinhole cameras, against the truth rendered directly from the world; PSNR after scaling the truth
by the ratio of the means; corners refined to sub-pixel precision in both images (the point every
gradient around it is orthogonal to), where the cameras resolve at least a quarter of a camera pixel per output pixel (`resolution_map()`). Three
poses of the moving rig for the bird's-eye view, two for the panorama:

| Output | PSNR | Corners checked | Largest corner error | Mean |
| --- | ---: | ---: | ---: | ---: |
| Bird's-eye 1024x1024, 11 m across, surround rig | 28.4 to 28.9 dB | 274 | 0.52 px | 0.10 to 0.15 px |
| Cylindrical 4096x1024 of the wall, surround rig | 32.3 to 32.5 dB | 484 to 490 | 0.49 px | 0.13 px |
| Spherical 4096x1024, six pinholes 5 cm from the centre | 38.1 dB | 479 | 0.53 px | 0.16 px |
| Cylindrical at infinity, five concentric pinholes | 36.1 dB | 497 | 0.48 px | 0.15 px |

The PSNR is limited by the cameras' resolution: the truth is as sharp as the output, the stitch only
as sharp as the cameras, which see the far ground at a seventh of the bird's-eye view's resolution.
There the corners are left out: one output pixel is a seventh of a camera pixel.

- **Against the CPU reference**: the GPU's tables equal `build_remap_tables()`'s to 0.000 px in
  every entry, and the blend, from the same tables, equals `reference_stitch()` byte for byte, for
  every projection, blend mode and output format; the gains equal `reference_gains()` exactly.
- **Exposure**: cameras rendered at exposures 1, 0.8, 1.2 and 0.9: the gains times the exposures
  agree to 0.3% (bird's-eye) and 1.8% (panorama, whose overlaps include ground seen with parallax).
  Two identical cameras at exposures 1 and 0.5 come out within 3% of each other.
- **Seams** (the mean luminance gradient along the hard seams, where the truth is flat; 0 is
  invisible), with those exposures: hard seam 31.1, feathered 0.65, hard seam with compensation
  0.58, feathered with compensation 0.33 (bird's-eye; the panorama alike).
- **Determinism**: two stitchers, or one stitcher twice, give the same bytes and gains.

## Performance

A frame of four 1280x720 RGBA8 cameras, the GPU time from its submission until the output is in
device memory, with CUDA events: **back to back**, the mean of 200 frames queued one after another
(the best of three rounds; what a stream of frames costs), and **alone**, the median of 50 frames
each waited for before the next (one frame's latency, launches included). The tables once, on the
host clock. SDK 5.0.0, CUDA 13.2. Milliseconds.

Jetson Orin, JetPack 7.2, power mode MAXN, the board otherwise idle:

| Output | Blend | Tables | Frame, back to back | Frame, alone | Table memory |
| --- | --- | ---: | ---: | ---: | ---: |
| Bird's-eye 1024x1024 | feather | 31 | 0.56 | 0.56 | 39 MB |
| Bird's-eye 1024x1024 | hard seam | 28 | 0.38 | 0.38 | 39 MB |
| Bird's-eye 1024x1024 | feather + exposure | 29 | 0.85 | 0.84 | 39 MB |
| Cylindrical 4096x1024 | feather | 136 | 1.55 | 1.56 | 129 MB |
| Cylindrical 4096x1024 | hard seam | 135 | 1.18 | 1.18 | 129 MB |
| Cylindrical 4096x1024 | feather + exposure | 136 | 2.09 | 2.10 | 129 MB |
| Spherical 4096x1024 | feather | 159 | 1.49 | 1.50 | 117 MB |

RTX 5090, idle, back to back: the bird's-eye view 0.024 (feather), 0.013 (hard seam) and 0.074
(with exposure), the cylindrical panorama 0.087, 0.050 and 0.133, the spherical one 0.081; tables
1.3 to 6.8 ms. Work of other processes on the same GPU slows both by several times: time on an
idle one.

RGB8 inputs or output change little. A frame costs about one memory pass over the output and the
camera pixels it samples; exposure compensation adds a sampling pass at a sixteenth of the pixels
and a one-thread solve. The tables are built in double precision, which the Jetson Orin's GPU runs
slowly; they are built once per calibration. The `stitching_benchmark` tool in the tests prints
this table on any machine (see [DEVELOPER.md](DEVELOPER.md)).

## Requirements

- Holoscan SDK 5.0 (Core and Sensor I/O; Testing for the tests), CUDA 13, CMake 4.0 or later and a
  C++20 compiler.
- [holoscan-messages](../messages/) 1.x, installed, or added to the same build first (the
  repository root does that).

## Build, test and install

From the repository root, in the repository's CI image or with the requirements above, after
installing holoscan-messages (see [its README](../messages/README.md#build-test-and-install)):

```bash
cmake -S modules/stitching -B build/stitching -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DCMAKE_PREFIX_PATH="$PWD/install"
cmake --build build/stitching
ctest --test-dir build/stitching --output-on-failure
cmake --install build/stitching --prefix "$PWD/install"
```

| CMake option | Default | |
| --- | --- | --- |
| `BUILD_TESTING` | `OFF` | Build the tests, when the module is built on its own |
| `HOLOSCAN_STITCHING_BUILD_TESTING` | `OFF` | Build the tests inside another project |
| `CMAKE_CUDA_ARCHITECTURES` | `native` (`all-major` for the package) | GPUs to compile for, unless already set |

Another project can also build the module as part of its own build, with `add_subdirectory()` or
`FetchContent`, and link `holoscan::stitching` without `find_package()`; it then builds a shared
library and none of its tests unless `HOLOSCAN_STITCHING_BUILD_TESTING` is `ON`.

The tests, every one but `stitching_real_data` on synthetic data made at test time:

| Test | Needs | Checks |
| --- | --- | --- |
| `stitching_stitcher` | GPU | Bird's-eye views and panoramas (cylindrical, spherical, at infinity, a partial 120-degree one) against the truth: PSNR, coverage, brightness and corners; the tables and the blend against the CPU reference; exposure compensation recovering known exposures and the seam visibility of each blend; smoothing; determinism; every input format with padded rows; a camera that sees nothing; two cameras that see the same; bad input |
| `stitching_operator` | GPU | `StitchOp` through the SDK's operator test harness, the calibration on its `camera_info_<i>` inputs: device, pinned host and pageable host images, the output equal to the `Stitcher`'s, its descriptor and provenance, capture times within `sync_window` and a window too wide refused, a one-camera rig, camera info with every frame set, frame sets before every camera's camera info skipped, a calibration that changes, and bad images, camera info and configurations |
| `stitching_real_data` | GPU, the local recordings | Reads `$HOLOSCAN_TEST_DATA/stereo-camera/<clip>/` and is skipped (exit 77) without it: every `camera_info` file of each clip taken; the stereo eyes, with the extrinsics from their two files, stitched: coverage, overlap, the stitch and gains equal to the CPU reference's, gains near 1, and, with the projector on, the eyes' correlation in the overlap peaking at the scene's distance and well above that at infinity or with the baseline turned (their PSNR there, about 28 dB, is set by the sensor noise and moves by under 1 dB between right and wrong extrinsics, so it is reported, not checked) |
| `stitching_calibration` | Nothing | `camera_info` files of every lens model read with the messages module and taken, cameras refused (skew, a region of interest, binning, an oversized image, an invalid calibration, poses that are not rotations), mounting poses, projection round trips for every model, the lens' valid range; also built on its own from `tests/host/` by the hosted CI job |
| `stitching_reference` | Nothing | Output points and their inverse, tables against hand computation (a region across a panorama's edge, a camera that sees nothing), feathering, hard seams, background and gains, and the gain solve; also built from `tests/host/` |

[`tests/installed_consumer/`](tests/installed_consumer/) is a separate project that uses the
installed packages: the four surround cameras' calibration written as `camera_info` files and sent
by `CameraInfoSourceOp`s, their rendered images through `StitchOp` to a bird's-eye view, each frame
checked against the truth, on one worker (it checks that the plan has one).

## Debian package

Inside the repository's CI image, with the module's folder as the Holoscan CLI project root:

```bash
holoscan --project-root modules/stitching package holoscan-stitching --language cpp \
  --pkg-generator DEB --local
```

This builds for every major GPU architecture and writes `holoscan-stitching_1.0.0_<arch>.deb` to the
current directory. It installs to `/usr` and depends on `holoscan (>= 5.0.0)` and
`holoscan-messages (>= 1.0.0, << 2)`, so install the messages package with it.

## Licenses

Apache-2.0. The module links the Holoscan SDK (Apache-2.0), which it does not redistribute, and
holoscan-messages (Apache-2.0). It uses no third-party code. See [NOTICE](NOTICE).
