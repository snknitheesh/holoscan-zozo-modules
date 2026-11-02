# holoscan-stitching

A Holoscan SDK 5 Module that stitches the images of several calibrated cameras into one image on
the GPU: a **cylindrical or spherical panorama** seen from a point of the rig, or a **bird's-eye
surround view** of the ground, as cars and mobile robots show it. `StitchOp` takes one image per
camera and emits the stitched image. The remap tables are built once per calibration, on the GPU;
each frame is then sampled bilinearly and blended, with distance-weighted feathering or hard seams
and optional exposure compensation. The [Stitching demo](../../applications/stitching/) runs it end
to end on four synthetic fisheye cameras.

| | |
| --- | --- |
| CMake package | `find_package(holoscan-stitching)` |
| Target | `holoscan::stitching` |
| Headers | `<holoscan/stitching/...>` |
| C++ namespace | `holoscan::stitching` |
| Debian package | `holoscan-stitching` |
| Platforms | x86_64; aarch64 (Jetson Thor and Orin, JetPack 7) |

## `StitchOp`

| Port | Type | |
| --- | --- | --- |
| `image_0` ... `image_<N-1>` (inputs) | `holoscan::schema::ImageT` | One per camera of the rig, in its order: MONO8, RGB8, BGR8, RGBA8 or BGRA8 of the camera's calibrated size, `uint8` `[H, W, C]` (or `[H, W]` for MONO8), rows possibly padded, in host or CUDA device memory |
| `stitched` (output) | `holoscan::schema::ImageT` | RGB8 or RGBA8 `[H, W, C]` of the output's size, in CUDA device memory |

A rig has 1 to 8 cameras (`kMaxCameras`). The SDK has no variable-length port list, so the operator
holds a fixed array of eight inputs, `images`, and declares only the rig's first N, named
`image_<i>` (as `HolovizOp` does for its layers); connect `camera->image` to `stitch->images[i]`.
With two or more cameras, the operator runs once per frame set: when an image with the same capture
time has arrived on every input, or within `sync_window` of each other. The output carries the
earliest capture time of its set and the set's frame id when they share one; its header's
`frame_id` is the configuration's (`"rig"`). With RGBA8 output, alpha is 255 where a camera sees the
pixel and 0 elsewhere.

`StitchConfig`:

| Field | Default | |
| --- | --- | --- |
| `rig` | none | The cameras: `CameraRig`, below |
| `output` | bird's-eye, 1024 x 1024, 20 m x 20 m | `OutputGeometry`: the projection, the size and its parameters, below |
| `blend` | feathering, RGBA8 | `BlendSettings`, below |
| `sync_window` | `0` | The largest spread of one frame set's capture times |
| `frame_id` | `"rig"` | The output header's frame |

`start()` builds the tables and logs their size, and warns about a camera that sees none of the
output (its images are received and ignored). Host images are copied to the GPU every frame, and
the operator says so once:

```text
StitchOp: images arrive in host memory and the stitcher reads CUDA device memory, so each frame set is copied host to device, 11059200 bytes (11.1 MB) per frame; to avoid the copy, send CUDA device images
```

## Calibration

`CameraRig` (`<holoscan/stitching/camera_rig.hpp>`) holds the cameras. Each `Camera` has a name, its
image size, pinhole intrinsics `fx`, `fy`, `cx`, `cy` (pixel centres at integers, as OpenCV and ROS
`CameraInfo` have them), a distortion model and its coefficients, and its pose `camera_from_rig`
(`p_camera = R p_rig + t`, row-major, metres). The rig frame is x forward, y left, z up (ROS REP
103); camera frames are optical (x right, y down, z forward). `mount(position, yaw, pitch, roll)`
makes the pose of a camera at `position`, turned left by `yaw`, tilted up by `pitch` (negative looks
down) and rolled by `roll`, in degrees.

| Distortion model | Coefficients | Also read as |
| --- | --- | --- |
| `none` | | |
| `brown_conrady` | k1, k2, p1, p2, k3 (4 means k3 = 0) | ROS `plumb_bob` |
| `rational_polynomial` | k1, k2, p1, p2, k3, k4, k5, k6 | |
| `kannala_brandt4` | k1, k2, k3, k4: equidistant fisheye, beyond 90 degrees | ROS `equidistant`, `fisheye` |

A rig file is the calibration.json style the repository's camera recordings use: the same camera
objects, distortion models and `{rotation, translation}` transforms, with a list of cameras:

```json
{
  "version": 1,
  "cameras": [
    {
      "name": "front", "width": 1280, "height": 720,
      "fx": 330.0, "fy": 330.0, "cx": 639.5, "cy": 359.5,
      "distortion": {"model": "kannala_brandt4", "coefficients": [0.02, -0.005, 0.001, -0.0001]},
      "camera_from_rig": {
        "rotation": [0, -1, 0, -0.5, 0, -0.8660254037844386, 0.8660254037844386, 0, -0.5],
        "translation": [0, 1.692820323027551, -1.3320508075688773]
      }
    }
  ]
}
```

A camera takes `camera_from_rig` or its inverse, `rig_from_camera`. `read_rig()` reads it, and also
reads a recording's own calibration.json: its `left`, `right` and `rgb` cameras, in the left camera's
frame. `to_json()` writes a rig file. A file is validated as it is read: sizes, finite positive focal
lengths, the coefficient count, rotations that are rotations (to 1e-6), distinct names.

Where a lens' distortion polynomial folds back (strong barrel distortion near the corners of a wide
lens), two rays would land on one pixel. `max_valid_angle()` finds the angle from the axis where the
projection stops moving outwards, and the stitcher ignores rays beyond it, as it ignores rays
behind a pinhole camera.

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
| `kFeather` (default) | Every camera that sees the pixel contributes, weighted by its margin, so the seam fades over the whole overlap (distance-weighted feathering). `feather_width` caps the weight, as OpenCV's `FeatherBlender`'s sharpness does: a camera reaches full weight that many pixels from its edge, and between those ramps the cameras count equally. |
| `kHardSeam` | Each pixel comes from the camera with the largest margin: a seam halfway through each overlap. |

**Exposure compensation** (`exposure_compensation`) scales each camera by a gain, solved on the GPU
every frame from the overlaps' mean brightness (gain compensation after Brown and Lowe, 2007): at
every 4th output pixel each way, every pair of cameras that both see it adds its luminance to that
pair's sum; one thread then solves the normal equations of

```text
E = sum over pairs (i, j) of N_ij * [ (g_i I_ij - g_j I_ji)^2 / sigma_N^2 + (1 - g_i)^2 / sigma_g^2 ]
```

for the gains `g`, where `I_ij` is camera i's mean over its overlap with camera j and `N_ij` the
overlap's samples. `exposure_sigma_noise` (`sigma_N`, 10 levels) and `exposure_sigma_gain`
(`sigma_g`) weigh matching the overlaps against keeping the gains near 1. Brown and Lowe use
`sigma_g = 0.1`, which on overlaps of mean brightness near 100 leaves about a third of an exposure
difference; the default 0.5 leaves about 2%. `exposure_smoothing` blends each frame's gains with the
previous frame's, against flicker. The sums are integers, so the gains do not depend on the order
of the GPU's additions. A camera with no overlap keeps gain 1. `Stitcher::gains()` returns them.

`background` is the colour where no camera sees; `output_format` RGB8 or RGBA8.

## Use it in your graph

```cmake
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core sensor_io)
find_package(holoscan-stitching 1.0 REQUIRED CONFIG)
target_link_libraries(my_app PRIVATE holoscan::stitching)
```

```cpp
#include <holoscan/stitching/stitch_op.hpp>

namespace st = holoscan::stitching;

const st::CameraRig rig = st::read_rig("rig.json");  // Four surround cameras.
st::StitchConfig surround{.rig = rig,
                          .output = st::birds_eye_view({-8.0, 8.0, -6.0, 6.0}, 0.02)};
surround.blend.exposure_compensation = true;
auto bev = graph.op<st::StitchOp>("bev", surround);

st::OutputGeometry pano{.projection = st::Projection::kCylindrical, .width = 4096, .height = 1024};
pano.panorama.elevation_min_deg = -40.0;
pano.panorama.elevation_max_deg = 40.0;
pano.panorama.center = {0.0, 0.0, 1.0};
auto panorama = graph.op<st::StitchOp>("panorama", st::StitchConfig{.rig = rig, .output = pano});

for (std::size_t i = 0; i < rig.cameras.size(); ++i) {
  graph.add_flow(cameras[i]->image, bev->images[i]);       // Output<holoscan::schema::ImageT>
  graph.add_flow(cameras[i]->image, panorama->images[i]);
}
```

Bind the outputs (`bev/stitched`, `panorama/stitched`) to a CUDA device in the `CompileOptions`, as
for any CUDA output. For live cameras, connect with `QueuePolicy::kDropOldest` and depth 1, and give
`sync_window` the cameras' trigger skew.

Without a graph, `Stitcher` (`<holoscan/stitching/stitcher.hpp>`) stitches device images on a
CUDA stream, and `reference_stitch()` with `build_remap_tables()` (`<holoscan/stitching/remap.hpp>`)
on the CPU. `<holoscan/stitching/synthetic_world.hpp>` has the tests' world: the surround and ring
rigs, GPU renders of their cameras and of the ground truth, the checkerboards' corners and
`compare_with_truth()`.

## Accuracy

On the synthetic world (a checkered ground plane and a checkered cylindrical wall around the rig),
rendered by four 1280x720 fisheye cameras (`surround_rig()`, about 215 degrees across) and by rings
of pinhole cameras, against the truth rendered directly from the world; PSNR after scaling the truth
by the ratio of the means; corners refined in both images as `cornerSubPix` does, where the
cameras resolve at least a quarter of a camera pixel per output pixel (`resolution_map()`). Three
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

One frame of four 1280x720 cameras, from submission until the output is in device memory, timed
with CUDA events over 200 frames queued back to back, the best of three rounds; the tables once,
on the host clock. x86_64: GeForce RTX 5090, driver 595. Orin: Jetson AGX Orin Developer Kit
(64 GB), JetPack 7.2 (R39.2), MAXN. SDK 5.0.0, CUDA 13.2. Milliseconds.

| Output | Blend | Tables, x86_64 | Frame, x86_64 | Tables, Orin | Frame, Orin | Table memory |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Bird's-eye 1024x1024 | feather | 1.6 | 0.024 | 33 | 0.57 | 39 MB |
| Bird's-eye 1024x1024 | hard seam | 1.3 | 0.013 | 29 | 0.37 | 39 MB |
| Bird's-eye 1024x1024 | feather + exposure | 1.3 | 0.074 | 29 | 0.81 | 39 MB |
| Cylindrical 4096x1024 | feather | 5.7 | 0.087 | 134 | 1.54 | 129 MB |
| Cylindrical 4096x1024 | hard seam | 5.8 | 0.050 | 133 | 1.17 | 129 MB |
| Cylindrical 4096x1024 | feather + exposure | 5.8 | 0.133 | 134 | 2.08 | 129 MB |
| Spherical 4096x1024 | feather | 6.8 | 0.081 | 158 | 1.48 | 117 MB |

The inputs are RGBA8; RGB8 inputs or output change little. A frame costs about one memory pass
over the output and the camera pixels it samples; exposure compensation adds a sampling pass at a
sixteenth of the pixels and a one-thread solve. The tables are built in double precision, which
the Orin's GPU runs slowly; they are built once, at `start()`. The `stitching_benchmark` tool in the
tests prints this table on any machine (see [DEVELOPER.md](DEVELOPER.md)).

## Requirements

- Holoscan SDK 5.0 (Core and Sensor I/O; Testing for the tests), CUDA 13, CMake 4.0 or later and a
  C++20 compiler.

## Build, test and install

From the repository root, in the repository's CI image or with the requirements above:

```bash
cmake -S modules/stitching -B build/stitching -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
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

The tests, every one on synthetic data made at test time:

| Test | Needs | Checks |
| --- | --- | --- |
| `stitching_stitcher` | GPU | Bird's-eye views and panoramas (cylindrical, spherical, at infinity, a partial 120-degree one) against the truth: PSNR, coverage, brightness and corners; the tables and the blend against the CPU reference; exposure compensation recovering known exposures and the seam visibility of each blend; smoothing; determinism; every input format with padded rows; a camera that sees nothing; two cameras that see the same; bad input |
| `stitching_operator` | GPU | `StitchOp` through the SDK's operator test harness: device and host images, the output equal to the `Stitcher`'s, its descriptor and provenance, capture times within `sync_window`, a one-camera rig, and bad images and configurations |
| `stitching_calibration` | Nothing | Distortion models, mounting poses, projection round trips for every model, the lens' valid range, rig files (both pose forms, every error) and a recording's calibration.json; also built on its own from `tests/host/` by the hosted CI job |
| `stitching_reference` | Nothing | Output points and their inverse, tables against hand computation (a region across a panorama's edge, a camera that sees nothing), feathering, hard seams, background and gains, and the gain solve; also built from `tests/host/` |

[`tests/installed_consumer/`](tests/installed_consumer/) is a separate project that uses the
installed package: a graph from four rendered surround cameras through `StitchOp` to a bird's-eye
view, each frame checked against the truth.

## Debian package

Inside the repository's CI image, with the module's folder as the Holoscan CLI project root:

```bash
holoscan --project-root modules/stitching package holoscan-stitching --language cpp \
  --pkg-generator DEB --local
```

This builds for every major GPU architecture and writes `holoscan-stitching_1.0.0_<arch>.deb` to the
current directory. It installs to `/usr` and depends on `holoscan (>= 5.0.0)`.

## Licenses

Apache-2.0. The module links the Holoscan SDK (Apache-2.0), which it does not redistribute. The gain
compensation follows M. Brown and D. G. Lowe, "Automatic Panoramic Image Stitching using Invariant
Features", IJCV 74(1), 2007; no code of theirs is used. See [NOTICE](NOTICE).
