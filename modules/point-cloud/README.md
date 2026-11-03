# holoscan-point-cloud

A Holoscan SDK 5 Module for point clouds on the GPU, in both directions, on the messages of
[holoscan-messages](../messages/): `CameraInfo` for the camera and `PointCloud` for the points.

- `DepthToPointCloudOp` turns a depth image from a depth camera, a stereo matcher or a monocular
  depth network, with the `CameraInfo` of its camera, into an organized `PointCloud`, optionally
  colored from an aligned color image. It is the step from 2D perception to 3D: mapping, ground
  and obstacle estimation, registration and grasp planning all start from it. The
  [Depth to Point Cloud demo](../../applications/depth_to_point_cloud/) runs it end to end.
- `PointCloudToDepthOp` projects a `PointCloud` from a lidar, a depth camera or a map into the
  camera a `CameraInfo` describes, at a given pose, and makes the depth image that camera would
  see, optionally with the points' colors. It is the step from 3D back to 2D: lidar depth for image
  networks, depth completion and fusion with camera images, and virtual cameras over a map. The
  [Point Cloud to Depth demo](../../applications/point_cloud_to_depth/) runs it end to end.

Deprojecting a depth image and projecting the points back into the same camera gives the image
back exactly.

| | |
| --- | --- |
| CMake package | `find_package(holoscan-point-cloud)` |
| Targets | `holoscan::point_cloud` |
| Headers | `<holoscan/point_cloud/...>` |
| C++ namespace | `holoscan::point_cloud` |
| Debian package | `holoscan-point-cloud` |
| Depends on | [holoscan-messages](../messages/) 1.0 |
| Platforms | x86_64; aarch64 (Jetson Thor with JetPack 7, Jetson Orin with JetPack 7.2) |

## The camera

Calibration has one input, a ROS `camera_info` YAML file. The first operator of a chain reads it
with the messages module's `CameraInfoSourceOp`, and both operators take the `CameraInfo` on a
mandatory `camera_info` input, with the messages module's contract: at the camera info's own rate
(`CameraInfoTiming::kLatest`, the default), each frame uses the newest camera info received before
it, and frames before the first one give no output; with camera info sent with every frame
(`kEveryFrame`), each frame waits for the one with its capture time. See
[Take camera info in an operator](../messages/README.md#take-camera-info-in-an-operator). Connect a
`CameraInfoSourceOp` with `latest_camera_info_connection()`.

The kernels model a pinhole camera: `CameraState::kUndistorted` or `kRectified`, without skew. A
`kRaw` camera, whose images still have their lens distortion (a distortion coefficient that is not
zero), fails the frame with an error that names the camera and says to undistort or rectify the
images first, as does a skew that is not zero. A distortion model whose coefficients are all zero
is undistorted and fine. The camera's `fx`, `fy`, `cx` and `cy` are those of the images as they
arrive, whatever `roi` and binning say about the sensor; a camera whose calibration changes is
picked up from the next frame.

## Operators

### `DepthToPointCloudOp`

For each pixel `(u, v)` with metric depth `Z = raw_depth * depth_scale`, the point in the camera
optical frame (x right, y down, z forward) is:

```text
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = Z
```

Pixel centers are at integer coordinates, as in OpenCV and ROS `CameraInfo`. Zero, negative and
non-finite depths, and depths outside `[depth_min, depth_max]`, become `(NaN, NaN, NaN)`, the
`PointCloud` message's invalid point. The cloud stays organized, `H x W x 3`, so pixel neighbors
stay neighbors for normals and segmentation. One CUDA thread handles one pixel. For a body frame
such as ROS's (x forward, y left, z up), rotate the cloud downstream.

| Port | Type | |
| --- | --- | --- |
| `depth` (input) | `holoscan::schema::ImageT` | `MONO16` (raw units, scaled by `depth_scale`) or `MONO32F`, `[H, W]` or `[H, W, 1]`, in host or CUDA device memory |
| `camera_info` (input, mandatory) | `holoscan::messages::schema::CameraInfoT` | The depth camera, for images of the depth image's size |
| `color` (input, with `color`) | `holoscan::schema::ImageT` | `RGB8`, `BGR8`, `RGBA8` or `BGRA8`, the depth image's size, with its capture time |
| `point_cloud` (output) | `holoscan::messages::schema::PointCloudT` | Organized: `points` float32 `[H, W, 3]` meters; `colors` uint8 `[H, W, 3]` RGB with `color`, else absent; no normals or intensities; `valid_count` -1. In device memory |

The cloud's header has the camera info's frame id and the depth image's capture time (0 when it
has none) and device sequence; the sample keeps the depth image's capture time and frame id.
`valid_count` is -1, not counted: counting would make the host wait for the kernel.

`DepthToPointCloudConfig`, passed to the constructor:

| Setting | Default | Meaning |
| --- | --- | --- |
| `depth_scale` | `0.001` | Raw depth to meters: `0.001` for uint16 millimeters, `1.0` for float32 meters |
| `depth_min`, `depth_max` | `0`, `100` | Valid depth range in meters |
| `color` | `false` | Declare the `color` input, and send colors in the cloud |
| `camera_info` | `kLatest`, 10 s | How the camera info arrives (`timing`) and, for `kLatest`, the oldest one a frame takes (`max_age`) |
| `max_width`, `max_height` | `1920`, `1080` | Largest image the output is sized for |

Rows must be packed, and data in device memory aligned to its element size (4 bytes for
four-channel color). The operator rejects an unsupported encoding, a color image of another size,
an image larger than its maximum, a camera info for another image size, and a camera it cannot
model, with an error that says so.

### `PointCloudToDepthOp`

Each point `p` is first moved into the camera optical frame (x right, y down, z forward) by the
configured extrinsics, a 3x4 row-major `[R | t]`: `(x, y, z) = R * p + t`. Then it projects to

```text
u = fx * x / z + cx
v = fy * y / z + cy
```

and lands on the pixel nearest `(u, v)`, `(lroundf(u), lroundf(v))`, with pixel centers at integer
coordinates. Each pixel keeps the nearest point: a z-buffer of one 64-bit key per pixel, the point's
depth above its index, written with `atomicMin`. Among equally near points the lower index wins, so
the image does not depend on thread scheduling. A second kernel turns the winning keys into depth
and color.

Points are dropped when a coordinate is not finite, when `z` is not positive (behind the camera) or
lies outside `[depth_min, depth_max]`, and when their pixel is outside the image. With `point_size`
3 or 5, a point covers the square of that many pixels a side around its pixel, clipped at the image
border; a point whose own pixel is outside the image draws nothing. Pixels no point reaches are 0,
and black in the color image.

The depth image is float32 meters, or uint16 counts of `depth_scale` meters, `round(z /
depth_scale)` rounded half away from zero. A uint16 depth cannot hold a point whose count would be
0, which marks an empty pixel, or above 65535: such a point is dropped rather than saturated, so a
pixel never shows a depth no point has, and the next point behind it can show through instead.

| Port | Type | |
| --- | --- | --- |
| `point_cloud` (input) | `holoscan::messages::schema::PointCloudT` | `points` `[N, 3]`, or organized `[H, W, 3]` taken as `H * W` points; `colors`, required with `color`; in host, pinned or device memory |
| `camera_info` (input, mandatory) | `holoscan::messages::schema::CameraInfoT` | The camera to project into, which sets the images' size |
| `depth` (output) | `holoscan::schema::ImageT` | `MONO32F` meters or `MONO16` counts, the camera's `[height, width]`, in device memory |
| `color` (output, with `color`) | `holoscan::schema::ImageT` | `RGB8`, the camera's `[height, width, 3]`, in device memory |

Outputs carry the cloud's capture time and frame id, and each image's header holds the capture time
and the camera info's frame id.

The depth image of a frame goes out before its color image. Should the color then fail to go out,
the depth image has been delivered alone and the frame fails. Pair the two by capture time, as the
demo's reporter does, and connect a consumer that only displays them with
`QueuePolicy::kDropOldest`, so that it never pushes back.

`PointCloudToDepthConfig`, passed to the constructor:

| Setting | Default | Meaning |
| --- | --- | --- |
| `max_width`, `max_height` | `1920`, `1080` | Largest camera the outputs are sized for, 1..16384; a larger one fails the frame |
| `extrinsics` | none (identity) | 3x4 row-major `[R \| t]` from the points' frame into the camera optical frame |
| `depth_min`, `depth_max` | `0`, `100` | Valid depth range in meters |
| `depth_format` | `DepthFormat::kFloat32` | `kFloat32` meters, or `kUint16` counts of `depth_scale` meters |
| `depth_scale` | `0.001` | Meters per uint16 count: millimeters by default |
| `point_size` | `1` | Pixels a side each point covers: 1, 3 or 5 |
| `color` | `false` | Declare the `color` output, from the clouds' colors |
| `camera_info` | `kLatest`, 10 s | How the camera info arrives (`timing`) and, for `kLatest`, the oldest one a cloud takes (`max_age`) |
| `max_points` | `16777216` | Largest cloud accepted, up to 2^30 points; a larger one fails the frame |

Points must be packed, and in device memory aligned to 4 bytes and on the operator's GPU. The
operator rejects a cloud that breaks the `PointCloud` layout (`check_point_cloud()`), a cloud above
`max_points`, a cloud without colors when `color` is set, a camera larger than its maximum, and a
camera it cannot model, with an error that says so. Clouds in host memory are copied to the GPU
through a staging buffer that grows with the largest cloud seen.

## Upgrading from 1.x

2.0 takes the camera as a `CameraInfo` message and sends and receives point clouds as `PointCloud`
messages, both from [holoscan-messages](../messages/), so that the module chains with the
collection's other modules. A 1.x graph changes as follows:

| | 1.x | 2.0 |
| --- | --- | --- |
| Intrinsics | `intrinsics` in the configs, or an `intrinsics` tensor input with `intrinsics_input` | The mandatory `camera_info` input, from `CameraInfoSourceOp` and a ROS `camera_info` file, or sent with every frame (`camera_info.timing = kEveryFrame`) |
| `DepthToPointCloudOp` outputs | `point_cloud` (`Tensor` `[H, W, 3]`) and `colors` (`Tensor`) | One `point_cloud` (`PointCloudT`) with `points` and, with `color`, `colors` |
| `DepthToPointCloudConfig::invalid_value` | Any value for invalid points | Gone: invalid points are NaN, as the message defines |
| `PointCloudToDepthOp` inputs | `points` and `colors` (`Tensor`), `intrinsics` | One `point_cloud` (`PointCloudT`) and `camera_info` |
| `PointCloudToDepthConfig` `width`, `height`, `frame_id` | The image size and the images' frame | From the camera info; `max_width` and `max_height` size the outputs |
| Raw (distorted) cameras | Not modeled, not detected | Refused with an error: undistort or rectify first |
| Package | `find_package(holoscan-point-cloud 1.1)` | `find_package(holoscan-point-cloud 2.0)`, which finds holoscan-messages; the Debian package depends on `holoscan-messages` |

Before:

```cpp
auto cloud = graph.op<pc::DepthToPointCloudOp>(
    "deproject", pc::DepthToPointCloudConfig{
                     .intrinsics = pc::CameraIntrinsics{615.0F, 615.0F, 319.5F, 239.5F},
                     .max_width = 640,
                     .max_height = 480});
graph.add_flow(camera->depth, cloud->depth);
graph.add_flow(cloud->point_cloud, consumer->points);  // Input<holoscan::Tensor>
```

After:

```cpp
auto info = graph.op<hm::CameraInfoSourceOp>(
    "camera_info", hm::CameraInfoSourceConfig{.camera_info_path = "depth.yaml"});
auto cloud = graph.op<pc::DepthToPointCloudOp>(
    "deproject", pc::DepthToPointCloudConfig{.max_width = 640, .max_height = 480});
graph.add_flow(camera->depth, cloud->depth);
graph.add_flow(info->camera_info, cloud->camera_info, hm::latest_camera_info_connection());
graph.add_flow(cloud->point_cloud, consumer->cloud);  // Input<hm::schema::PointCloudT>
```

where `depth.yaml` is the camera's `camera_info` file, with `camera_matrix` holding `fx`, `fy`,
`cx` and `cy`. A consumer reads the points from `sample->data.points` and the colors from
`sample->data.colors`, and checks the message with `hm::check_point_cloud()`.

## Use it in your graph

```cmake
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core sensor_io)
find_package(holoscan-point-cloud 2.0 REQUIRED CONFIG)
target_link_libraries(my_app PRIVATE holoscan::point_cloud)
```

```cpp
#include <holoscan/messages/camera_info_source_op.hpp>
#include <holoscan/point_cloud/depth_to_point_cloud_op.hpp>
#include <holoscan/point_cloud/point_cloud_to_depth_op.hpp>

namespace hm = holoscan::messages;
namespace pc = holoscan::point_cloud;

auto info = graph.op<hm::CameraInfoSourceOp>(
    "camera_info", hm::CameraInfoSourceConfig{.camera_info_path = "depth.yaml"});
auto cloud = graph.op<pc::DepthToPointCloudOp>(
    "deproject", pc::DepthToPointCloudConfig{.depth_scale = 0.001F,  // uint16 millimeters
                                             .depth_min = 0.1F,
                                             .depth_max = 10.0F,
                                             .max_width = 640,
                                             .max_height = 480});
graph.add_flow(camera->depth, cloud->depth);  // Output<holoscan::schema::ImageT>
graph.add_flow(info->camera_info, cloud->camera_info, hm::latest_camera_info_connection());

// Lidar points into the same camera: x forward, y left, z up into the camera
// optical frame, the lidar 10 cm above the camera.
auto project = graph.op<pc::PointCloudToDepthOp>(
    "project", pc::PointCloudToDepthConfig{
                   .max_width = 640,
                   .max_height = 480,
                   .extrinsics = pc::Extrinsics{{0.0F, -1.0F, 0.0F, 0.0F,    //
                                                 0.0F, 0.0F, -1.0F, -0.1F,   //
                                                 1.0F, 0.0F, 0.0F, 0.0F}},
                   .depth_min = 0.3F,
                   .depth_max = 50.0F,
                   .depth_format = pc::DepthFormat::kUint16,  // millimeters
                   .point_size = 3});
graph.add_flow(lidar->cloud, project->point_cloud);  // Output<hm::schema::PointCloudT>
graph.add_flow(info->camera_info, project->camera_info, hm::latest_camera_info_connection());
graph.add_flow(project->depth, consumer->depth);  // Input<holoscan::schema::ImageT>
```

The outputs of both operators are in device memory, so bind their GPU when compiling:

```cpp
holoscan::CompileOptions options;
options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
    .operator_path = "deproject", .output_port = "point_cloud", .device = holoscan::DeviceId{0}});
const auto plan = holoscan::compile(graph, std::move(options));
```

The kernels need no Holoscan graph: `holoscan::point_cloud::launch_deproject()` in
`<holoscan/point_cloud/deproject.hpp>` and `launch_project()` in
`<holoscan/point_cloud/project.hpp>`. For tests, `<holoscan/point_cloud/synthetic_depth.hpp>` is a
synthetic depth camera that renders frames on the GPU, evaluates the same depths on the host and
gives its `CameraInfo` (`synthetic_camera_info()`), `<holoscan/point_cloud/synthetic_scene.hpp>`
makes shuffled clouds in a body frame from it, with points mixed in that must not reach the image,
and `<holoscan/point_cloud/depth_truth.hpp>` checks a depth image made from such a cloud against the
camera. Host code that compares against them exactly must be compiled with `-ffp-contract=off`.

## Requirements

- Holoscan SDK 5.0 (Core and Sensor I/O; Testing for the tests), CUDA 13, CMake 4.0 or later and a
  C++20 compiler.
- [holoscan-messages](../messages/) 1.0, installed, or added to the same build before this module
  (the repository's root `CMakeLists.txt` does).

The module downloads nothing and needs no models. The real-data test reads local recordings when
`HOLOSCAN_TEST_DATA` is set.

## Build, test and install

From the repository root, in the repository's CI image or with the requirements above, after
installing holoscan-messages into `install/`:

```bash
cmake -S modules/point-cloud -B build/point-cloud -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DCMAKE_PREFIX_PATH="$PWD/install"
cmake --build build/point-cloud
ctest --test-dir build/point-cloud --output-on-failure
cmake --install build/point-cloud --prefix "$PWD/install"
```

| CMake option | Default | |
| --- | --- | --- |
| `BUILD_TESTING` | `OFF` | Build the tests, when the module is the top-level project |
| `HOLOSCAN_POINT_CLOUD_BUILD_TESTING` | `OFF` | Build the tests when another project includes the module (`add_subdirectory()`, `FetchContent`); its `BUILD_TESTING` and `include(CTest)` do not |
| `CMAKE_CUDA_ARCHITECTURES` | `native` (`all-major` for the package) | GPUs to compile for |

The tests:

| Test | Needs | Checks |
| --- | --- | --- |
| `point_cloud_deproject` | GPU | The kernel against a CPU reference: float32 and uint16 depth, holes, the valid range, non-finite depth, every color layout and the channel swap, image sizes that are not a multiple of the block size, intrinsics in device memory, usable or not, rejected color layouts, and launch error reporting |
| `point_cloud_depth_to_point_cloud_op` | GPU | `DepthToPointCloudOp` through the SDK's operator test harness and in small graphs: host and device input, color in every encoding, camera info at its own rate and with every frame, changing from frame to frame, every point and color against the synthetic camera, the cloud message's layout and header, the capture time and frame id of the cloud, frames before the first camera info and frames whose camera info carries another capture time, `CameraInfoSourceOp` reading a camera_info file, how host input is held until its copy is done, and the rejection of raw and skewed cameras, a camera of another size, misaligned data and unusable inputs and settings |
| `point_cloud_project` | GPU | The projection kernels against a CPU reference, pixel for pixel: random clouds through a rigid transform, every splat size, float32 and uint16 depth, colors, ties and the z-buffer's determinism, the valid range, non-finite points, points behind the camera, the image border, uint16 rounding and its limits, odd image sizes, intrinsics in device memory, and the round trip of a depth image through its points; the synthetic cloud's GPU and host versions, the rigid inverse, and launch error reporting |
| `point_cloud_point_cloud_to_depth_op` | GPU | `PointCloudToDepthOp` through the SDK's operator test harness and in small graphs: organized clouds that give their image back exactly, unorganized synthetic clouds with colors in a body frame against ground truth, camera info at its own rate and with every cloud, changing from cloud to cloud, images of the camera's size, the capture time and frame id of the outputs, clouds before the first camera info and clouds whose camera info carries another capture time, an empty cloud, the image headers, clouds from a producer's host, pinned and device pools, how host input is held, and the rejection of raw and skewed cameras, a camera larger than the maximum, clouds that break the layout or lack the colors asked for, and unusable settings |
| `point_cloud_real_data` | GPU, `HOLOSCAN_TEST_DATA`; skipped without it | The local recordings: each clip's first ten depth images with its `depth.yaml` through `CameraInfoSourceOp`, `DepthToPointCloudOp` and `PointCloudToDepthOp`; every point against the host's deprojection of its pixel, and every depth back exactly |

[`tests/installed_consumer/`](tests/installed_consumer/) is a separate project that uses the
installed package: a small graph that writes the synthetic camera's camera_info file, sends it with
`CameraInfoSourceOp`, deprojects synthetic uint16 depth images from host memory, checks every
point, and projects the clouds back into the camera, which must give every depth back exactly. Build it with the install prefix on `CMAKE_PREFIX_PATH` and run `ctest` there.

## Debian package

Inside the repository's CI image, with the module's folder as the Holoscan CLI project root:

```bash
holoscan --project-root modules/point-cloud package holoscan-point-cloud --language cpp \
  --pkg-generator DEB --local
```

`--language cpp` selects the module rather than its package entry, which has the same name. This
builds for every major GPU architecture and writes `holoscan-point-cloud_2.0.0_<arch>.deb` to the
current directory. It installs to `/usr` and depends on `holoscan (>= 5.0.0)` and
`holoscan-messages (>= 1.0.0)`, so install it with the messages module's package,
`apt install ./holoscan-messages_1.0.0_<arch>.deb ./holoscan-point-cloud_2.0.0_<arch>.deb`, which
needs NVIDIA's CUDA repository.

## Licenses

Apache-2.0. See [NOTICE](NOTICE).
