# holoscan-stereo-camera

A Holoscan SDK 5 Module that makes a stereo camera a Holoscan source. `StereoCameraOp` sends each
frame as a left and a right image, optionally an RGB image and a depth image, and the camera's
calibration, all stamped with the frame's capture time, so that stereo matching, depth, point
clouds and detection downstream can take them together. It is generic: the frames come from a
backend, and any camera that gives a left and a right image (and optionally RGB) fits one.

- `realsense`: an Intel RealSense through librealsense, with its factory calibration. Built when
  librealsense is installed.
- `v4l2-sbs`: a V4L2 camera that sends both eyes side by side in one frame, as a ZED does, with
  the calibration from a file.
- `playback`: a recording (below), which needs no camera and no librealsense. Always built.
- Your own: implement `StereoSource` and pass a factory.

The [Stereo Camera demo](../../applications/stereo_camera/) shows the images in HoloViz and checks
them, from each backend.

| | |
| --- | --- |
| CMake package | `find_package(holoscan-stereo-camera)` |
| Targets | `holoscan::stereo_camera` |
| Headers | `<holoscan/stereo_camera/...>` |
| C++ namespace | `holoscan::stereo_camera` |
| Debian package | `holoscan-stereo-camera` |
| Tool | `stereo_camera_record`, which records a camera |
| Platforms | x86_64; aarch64 (Jetson Thor with JetPack 7, Jetson Orin with JetPack 7.2) |

## `StereoCameraOp`

| Port | Type | |
| --- | --- | --- |
| `left` (output) | `holoscan::schema::ImageT` | `MONO8` `[H, W]`, or `RGB8` `[H, W, 3]` for a color pair |
| `right` (output) | `holoscan::schema::ImageT` | As `left`, the same size |
| `rgb` (output, with an RGB image) | `holoscan::schema::ImageT` | `RGB8` `[H, W, 3]` |
| `depth` (output, with a depth image) | `holoscan::schema::ImageT` | `MONO16` `[H, W]`, in the calibration's `depth_scale` metres per unit; 0 is no depth |
| `calibration` (output, unless turned off) | `holoscan::stereo_camera::schema::StereoCalibrationT` | The calibration, with the source's frame number, timestamp and device sequence |

Images are in host memory, or in CUDA device memory with `OutputMemory::kDevice`, rows packed.
Every port of a frame carries the same capture time (on the graph's clock: the time the frame was
taken, when the backend knows how old it is, otherwise when the operator took it) and frame id (1,
2, 3, ... in the run), so a consumer takes them together with `OnAligned`. Each image's header has
that capture time, the frame id `<frame_id>_left`, `_right`, `_rgb` or `_depth`, and the camera's
own sample number as `device_sequence`. The `rgb` and `depth` ports exist only when the backend
gives the stream, and `calibration` unless `calibration` is false: an emit into an unconnected
output fails, so a graph that does not use the calibration turns its port off.

A live camera's frames are not worth waiting for. Connect it with
`holoscan::QueuePolicy::kDropOldest` edges, so that a consumer that is still busy finds the newest
frame waiting when it is done; on a blocking edge, the rest of a frame a consumer has no room for
is dropped. A recording plays losslessly: on a blocking edge, what a consumer has no room for goes
again on the next tick, port by port, so every frame arrives, in order.

The device is opened in `start()` and closed in `stop()`, never in the constructor: the SDK may
construct an operator more than once, and a camera serves one owner at a time. A device that does
not open fails the run, with the reason.

`StereoCameraConfig`, passed to the constructor:

| Setting | Default | Meaning |
| --- | --- | --- |
| `source` | playback | The backend and its settings: `SourceConfig`, below |
| `make_source` | none | A factory for a `StereoSource` of your own, used instead of `source` |
| `count` | `0` | Frames to send, after which the operator idles; 0 sends until a recording ends, or forever |
| `period` | `0` | A recording's pace, 1 ms to 1 s; 0 takes the recorded rate. A live camera is polled every 2 ms |
| `memory` | `OutputMemory::kHost` | `kHost`, or `kDevice` (bind the ports' GPU when compiling) |
| `calibration` | `true` | Declare the `calibration` output |
| `frame_id` | `stereo_camera` | The images' header frame ids start with it |
| `stats` | none | A `StereoCameraStats` to count the frames sent and dropped |

`SourceConfig`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `backend` | `Backend::kPlayback` | `kRealSense`, `kV4l2SideBySide` or `kPlayback` |
| `rgb`, `depth` | `false` | Also send the RGB and the depth image: a RealSense turns those streams on, a recording must have them, a side-by-side camera has neither |
| `realsense` | | `serial` (empty: the first camera), `width` `1280`, `height` `720`, `fps` `30`, `rgb_width` `1280`, `rgb_height` `720`, `emitter_enabled` `true` |
| `v4l2` | | `device` `/dev/video0`, `width` `2560` and `height` `720` (the whole side-by-side frame), `fps` `30`, `format` `kYuyv` or `kGray`, `output` `kMono8` or `kRgb8`, `calibration_file` (required) |
| `playback` | | `directory` (required), `loop` `false` |

## Backends

### realsense

The camera's two infrared cameras, Y8, are the eyes; the camera rectifies them, so `rectified` is
true and they have no distortion. With `rgb`, its color camera, RGB8; with `depth`, its depth,
Z16, which a RealSense computes in its left infrared camera's frame. The calibration is the
camera's factory calibration, read at `start()`: each stream's intrinsics
(`rs2::video_stream_profile::get_intrinsics` of the stream profile started, in the format
asked for), each one's pose relative to the left infrared camera (`get_extrinsics_to`), and the
depth sensor's depth scale. librealsense's pipeline gives frame sets whose streams it matched; the
two infrared images are taken together. `rs-enumerate-devices -c` prints one set of intrinsics for
all of a resolution's color formats, the first format's; a camera may report slightly different
ones for RGB8, as a D585 prototype does.

`emitter_enabled` turns the infrared projector on or off (`RS2_OPTION_EMITTER_ENABLED`). On, the
eyes see its dot pattern, which gives the camera's depth, and any stereo matcher, texture on plain
surfaces; off, they see only the scene, as in daylight. A camera without a projector refuses
`emitter_enabled = true`.

A frame's timestamp is librealsense's, in global time (the camera's hardware clock mapped to the
host's), when the camera gives hardware timestamps. A camera or kernel that does not (librealsense
needs its kernel patches or its metadata support for them) gets the time the host received the
frame instead, in milliseconds of the system clock.

The backend is built when CMake finds `realsense2` 2.50 or later; `HOLOSCAN_STEREO_CAMERA_WITH_REALSENSE=OFF`
leaves it out. `holoscan::stereo_camera::realsense_available()` and `holoscan-stereo-camera_WITH_REALSENSE`
say whether a build has it. Install librealsense from Intel's repository
(<https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md>):
`librealsense2-dev`, and `librealsense2-utils` for `rs-enumerate-devices`.

### v4l2-sbs

A camera that puts both eyes side by side in one V4L2 frame, YUYV or 8-bit gray: a ZED, or another
USB stereo camera. The left half of each row is the left eye. The eyes go out gray (the luma) or
RGB (BT.601, limited range). Only the newest frame is used. The calibration comes from a file in
the format of a recording's `calibration.json`, whose eyes must be half the frame's width.

### playback

A recording's frames, in order, once or in a loop; their capture time is when the operator sends
them, and the calibration message carries the recorded frame number and timestamp. The recording
is checked when the operator is constructed, and every image as it is read: a missing or broken
file fails the run with its name.

## Recordings

A recording is a directory that plays back with this library alone, without the camera or its
driver:

```text
calibration.json      The calibration (below)
index.csv             One line per frame
left/000000.pgm       8-bit PGM (P5); 8-bit PPM (P6) for a color pair
right/000000.pgm      Like left
rgb/000000.ppm        8-bit PPM, when recorded
depth/000000.pgm      16-bit PGM (big-endian, as the format says), when recorded
```

`index.csv` has a header line, then one line per frame:

```text
frame,timestamp_ns,device_sequence,left,right,rgb,depth
0,1793221326194000000,31,left/000000.pgm,right/000000.pgm,rgb/000000.ppm,depth/000000.pgm
```

The `rgb` and `depth` columns are there when those images are. `frame` counts 0, 1, 2, ...;
`timestamp_ns` is the camera's time for the frame and increases; `device_sequence` is the camera's
frame number, or 0. The paths are relative to the directory and stay inside it.

`calibration.json` holds `version` (1); `camera` (`name`, `serial`, `firmware`, and the `settings`
it ran with, as text); `rectified`; `left` and `right`, each with `width`, `height`, `fx`, `fy`,
`cx`, `cy` (pixels, pixel centers at integers) and `distortion` (`model`, `coefficients`);
`right_from_left`, a `rotation` (9 numbers, row-major) and a `translation` (3, in metres) from the
left eye's frame to the right's; and optionally `rgb` and `depth`, each a camera with `from_left`,
and `depth` with its `scale` in metres per unit. Distortion models are `none`, `brown_conrady`,
`modified_brown_conrady`, `inverse_brown_conrady`, `kannala_brandt4`, `ftheta` and
`rational_polynomial`. The reader checks it all and ignores keys it does not know.

`stereo_camera_record` writes one from a camera, through the same backends, so a recording holds
what `StereoCameraOp` would send. It needs no Holoscan SDK, so it also builds on its own, on a
machine with the camera:

```bash
cmake -S modules/stereo-camera/tools -B build/recorder -DCMAKE_BUILD_TYPE=Release
cmake --build build/recorder
build/recorder/stereo_camera_record --output d585_720p --frames 60 --emitter true
build/recorder/stereo_camera_record --source v4l2-sbs --device /dev/video0 --width 2560 \
  --height 720 --calibration zed.json --output zed --frames 60
```

`--help` lists its options. It skips 30 frames first, for the camera's exposure, and reports gaps
in the camera's frame numbers.

## Use it in your graph

```cmake
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core sensor_io)
find_package(holoscan-stereo-camera 1.0 REQUIRED CONFIG)
target_link_libraries(my_app PRIVATE holoscan::stereo_camera)
```

```cpp
#include <holoscan/stereo_camera/stereo_camera_op.hpp>

namespace sc = holoscan::stereo_camera;

sc::StereoCameraConfig config;
config.source.backend = sc::Backend::kRealSense;
config.source.rgb = true;
config.source.realsense.emitter_enabled = false;
auto camera = graph.op<sc::StereoCameraOp>("camera", config);
const holoscan::ConnectionOptions newest{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                         .queue_depth = 1U};
graph.add_flow(camera->left, matcher->left, newest);    // Output<holoscan::schema::ImageT>
graph.add_flow(camera->right, matcher->right, newest);
graph.add_flow(camera->calibration, matcher->calibration, newest);
```

With `OutputMemory::kDevice`, bind the image ports' GPU when compiling, as for any output in
device memory:

```cpp
holoscan::CompileOptions options;
for (const char* port : {"left", "right", "rgb"}) {
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "camera", .output_port = port, .device = holoscan::DeviceId{0}});
}
```

`from_message()` in `<holoscan/stereo_camera/calibration_message.hpp>` turns the calibration message
back into a `StereoCalibration`; its `depth_from_disparity()` is `fx * baseline`, so that depth in
metres is that over a disparity in pixels. Its `fx`, `fy`, `cx` and `cy` are those of
`holoscan::point_cloud::CameraIntrinsics`, for `DepthToPointCloudOp`.

The rest works without a Holoscan graph: the backends (`<holoscan/stereo_camera/source.hpp>`),
recordings (`recording.hpp`), the calibration file (`calibration.hpp`), Netpbm images
(`netpbm.hpp`), side-by-side splitting (`side_by_side.hpp`), and a synthetic rectified camera with
known disparity for tests (`synthetic.hpp`), which writes recordings too.

## Requirements

- Holoscan SDK 5.0 (Core and Sensor I/O; Testing for the tests), the CUDA runtime, CMake 4.0 or
  later and a C++20 compiler.
- For the realsense backend: librealsense 2.50 or later. Without it, the module builds without the
  backend.

## Build, test and install

From the repository root, in the repository's CI image or with the requirements above:

```bash
cmake -S modules/stereo-camera -B build/stereo-camera -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build/stereo-camera
ctest --test-dir build/stereo-camera --output-on-failure
cmake --install build/stereo-camera --prefix "$PWD/install"
```

| CMake option | Default | |
| --- | --- | --- |
| `BUILD_TESTING` | `OFF` | Build the tests, when the module is the top-level project |
| `HOLOSCAN_STEREO_CAMERA_BUILD_TESTING` | `OFF` | Build the tests when another project includes the module (`add_subdirectory()`, `FetchContent`); its `BUILD_TESTING` and `include(CTest)` do not |
| `HOLOSCAN_STEREO_CAMERA_WITH_REALSENSE` | `ON` | Build the realsense backend when librealsense is found |
| `HOLOSCAN_STEREO_CAMERA_BUILD_RECORDER` | `ON` at the top level | Build and install `stereo_camera_record` |

The tests:

| Test | Needs | Checks |
| --- | --- | --- |
| `stereo_camera_calibration` | | The calibration file: a RealSense's and the synthetic camera's calibration written and read back exactly, minimal files and unknown keys, every field the reader requires and every check `validate()` makes, and malformed JSON with its line and column |
| `stereo_camera_recording` | | Netpbm images in every format, padded rows, 16-bit byte order and comments; `index.csv` and its errors; the synthetic camera's disparity; a synthetic recording written and read back pixel for pixel; the writer's rules; and malformed recordings: missing, broken, truncated and resized files, wrong formats and timestamps that do not increase |
| `stereo_camera_side_by_side` | | Gray and YUYV side-by-side frames split into gray and RGB eyes, padded rows, BT.601 at known colors, frames that cannot be split |
| `stereo_camera_source` | | The playback backend: every frame in order, once and in a loop, with and without rgb and depth; backend names; configurations each backend refuses, and a V4L2 device that does not open |
| `stereo_camera_op` | GPU | `StereoCameraOp` in small graphs: synthetic recordings in host and device memory, every image pixel for pixel, the capture times, frame ids, headers and calibration messages; a count and a loop; a slow consumer that still gets every recorded frame; a live source of the test's own, opened once on drop-oldest edges; refused configurations, and a source that fails to open failing the run with its reason |
| `stereo_camera_real_data` | Data | Every recording in `$HOLOSCAN_TEST_DATA/stereo-camera/`: the calibration reads, every frame decodes, the eyes have the same size as each other and as the calibration, and the timestamps and frame numbers increase |
| `stereo_camera_real_data_op` | GPU, data | The same recordings through `StereoCameraOp` into device memory, every image against the recording |

The first four need no SDK and also run on their own (`tests/host/`). The two real-data tests are
skipped, with CTest's skip status, unless `HOLOSCAN_TEST_DATA` names a folder with a
`stereo-camera/` folder of recordings. The recordings are not in this repository.

[`tests/installed_consumer/`](tests/installed_consumer/) is a separate project that uses the
installed package: it writes a synthetic recording, plays it through `StereoCameraOp` into device
memory and checks every frame. Build it with the install prefix on `CMAKE_PREFIX_PATH` and run
`ctest` there.

## Debian package

Inside the repository's CI image, with the module's folder as the Holoscan CLI project root:

```bash
holoscan --project-root modules/stereo-camera package holoscan-stereo-camera --language cpp \
  --pkg-generator DEB --local
```

This writes `holoscan-stereo-camera_1.0.0_<arch>.deb`, which installs to `/usr` and depends on
`holoscan (>= 5.0.0)`, and on `librealsense2` when built with the realsense backend. The CI image
has no librealsense, so its package has no realsense backend.

## Licenses

Apache-2.0. See [NOTICE](NOTICE).
