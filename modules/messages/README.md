# holoscan-messages

A Holoscan SDK 5 Module with the messages that every module of this collection uses for calibration
and point clouds. The SDK 5.0 ships sensor schemas for images and IMU samples only, so camera info
and point clouds have no SDK message; this module defines one set, in the SDK's own FlatBuffers
schema system, reusing its `Header`, `Matrix3d`, `Vector3d` and `Tensor`:

- `CameraInfo`: one camera's calibration for the images it goes with, and whether those images are
  raw, undistorted or rectified.
- `StereoCameraInfo`: a stereo pair's two cameras and their extrinsics in one message, so that they
  always come from the same calibration.
- `PointCloud`: points with optional colors, normals and intensities, as tensors in host or device
  memory.

Calibration has one file format, the ROS `camera_info` YAML file that ROS's `camera_calibration`
writes. The first operator of a chain reads it with `CameraInfoSourceOp`; every operator after it
takes `CameraInfo` or `StereoCameraInfo` on an input port, with the contract
[below](#take-camera-info-in-an-operator). The [Messages demo](../../applications/messages/) runs
all of it end to end.

| | |
| --- | --- |
| CMake package | `find_package(holoscan-messages)` |
| Targets | `holoscan::messages` |
| Headers | `<holoscan/messages/...>` |
| C++ namespace | `holoscan::messages`; generated messages in `holoscan::messages::schema` |
| FlatBuffers namespace | `holoscan.messages.schema` |
| Debian package | `holoscan-messages` |
| Platforms | x86_64; aarch64 (Jetson Thor with JetPack 7, Jetson Orin with JetPack 7.2) |

## Messages

Each message is a generated type that ports carry (`schema::CameraInfoT`,
`schema::StereoCameraInfoT`, `schema::PointCloudT`), from the schemas in [schemas/](schemas/).
`CameraInfo` and `StereoCameraInfo` also have plain C++ values in
[camera_info.hpp](include/holoscan/messages/camera_info.hpp), which need no SDK, with `validate()`
and the stereo helpers; [conversion.hpp](include/holoscan/messages/conversion.hpp) converts between
the two, validating both ways, so an invalid calibration is never sent or used.

Conventions, everywhere: pixel centers are at integer coordinates (as in ROS and OpenCV); matrices
are row-major; a camera frame has x right, y down and z forward; distances are metres.

### `CameraInfo`

| Field | Type | |
| --- | --- | --- |
| `header` | SDK `Header` | `capture_timestamp_ns`: the time the calibration is for; `frame_id`: the camera's optical frame |
| `width`, `height` | `uint32` | The images' size in pixels, > 0 |
| `state` | `Raw`, `Undistorted`, `Rectified` | What has been done to the images. `Raw`: a distortion coefficient is not zero. `Undistorted`: none is. `Rectified`: undistorted and in a stereo pair's rectified frame |
| `fx`, `fy`, `cx`, `cy`, `skew` | `double` | Pixels: a point `(x, y, z)` projects to `u = fx x/z + skew y/z + cx`, `v = fy y/z + cy`, before distortion |
| `distortion_model` | `None`, `PlumbBob`, `RationalPolynomial`, `Equidistant` | ROS's `plumb_bob` (k1 k2 p1 p2 k3), `rational_polynomial` (k1 k2 p1 p2 k3 k4 k5 k6) and `equidistant` (k1 k2 k3 k4) |
| `coefficients` | `[double]` | Exactly as many as the model has: 0, 5, 8 or 4 |
| `rectification` | `Matrix3d`, optional | Rotation from this camera's frame to its stereo pair's rectified frame; absent when there is none to apply, always for `Rectified` |
| `roi` | `x, y, width, height`, optional | Full-resolution sensor pixels the images hold; absent for the whole sensor |
| `binning_x`, `binning_y` | `uint32`, default 1 | Sensor pixels per image pixel; ROS's 0 reads as 1 |

### `StereoCameraInfo`

| Field | |
| --- | --- |
| `header` | `frame_id`: the left camera's frame |
| `left`, `right` | `CameraInfo` |
| `right_from_left` | `rotation` (`Matrix3d`) and `translation` (`Vector3d`, metres): `p_right = R p_left + t` |
| `baseline` | When both cameras are `Rectified`: the right camera's distance along +x of the rectified left frame, metres, > 0; else 0 |
| `doffs` | When both are `Rectified`: `cx_right - cx_left`, pixels; else 0 |

For a rectified pair, `depth_from_disparity(info, d)` is `fx * baseline / (d + doffs)` metres for a
disparity `d = x_left - x_right`, and `fx_baseline(info)` is `fx * baseline`. `validate()` checks
that a rectified pair has one `fx`, `fy` and `cy`, no skew, `R = I` and `t = (-baseline, 0, 0)`, and
that the cameras' rectifications, when present, agree with the extrinsics.

### `PointCloud`

| Field | Type | |
| --- | --- | --- |
| `header` | SDK `Header` | `capture_timestamp_ns`: when the points were measured; `frame_id`: the frame they are in |
| `points` | float32 `[N, 3]`, or `[H, W, 3]` when organized | Metres; an invalid point is NaN |
| `colors` | uint8, same leading shape, 3 | RGB; optional |
| `normals` | float32, same leading shape, 3 | Unit vectors, NaN where unknown; optional |
| `intensity` | float32, same leading shape | Optional |
| `organized` | `bool` | Whether `points` is `[H, W, 3]` |
| `valid_count` | `int64` | Valid points, or -1 when the producer did not count them (counting device data would cost a synchronization) |

The SDK carries several tensors in one message (a descriptor-backed schema), each in host or
device memory, but a message always carries all of them. An optional field that a cloud does not
have is therefore the empty rank-1 tensor `[0]`; `is_present()` tells them apart.
[point_cloud_layout.hpp](include/holoscan/messages/point_cloud_layout.hpp) states the rules without
the SDK, and `check_point_cloud()` checks a received cloud against them.

Frame ids are at most 256 bytes (`kMaxFrameIdBytes`), so that every message has a fixed largest
size.

## Calibration files

`read_camera_info_yaml(path)` reads a ROS `camera_info` file, and
`read_stereo_camera_info_yaml(left, right)` a stereo pair's two:

```yaml
image_width: 1280
image_height: 720
camera_name: left
camera_matrix: {rows: 3, cols: 3, data: [fx, skew, cx, 0, fy, cy, 0, 0, 1]}
distortion_model: plumb_bob
distortion_coefficients: {rows: 1, cols: 5, data: [k1, k2, p1, p2, k3]}
rectification_matrix: {rows: 3, cols: 3, data: [R, row-major]}
projection_matrix: {rows: 3, cols: 4, data: [fx', 0, cx', Tx, 0, fy', cy', Ty, 0, 0, 1, 0]}
```

Block or flow style alike. `image_width`, `image_height` and `camera_matrix` are required; any key
other than these eight is refused, so that a misspelt key cannot drop a matrix. Every refusal names
the file, the key and the line. ROS leaves implicit what the images are; reading makes it explicit:

- `camera_name` is the header's `frame_id`.
- K and D describe the camera: a camera with a coefficient that is not zero is `Raw`, else
  `Undistorted`.
- R, when present and not the identity, is the camera's `rectification`.
- In a stereo pair's files, as a ROS stereo calibration writes them (K and D of each raw camera, R
  rotating each into the rectified frame, P of each rectified camera with the right one's
  `Tx = -fx' * baseline`), the extrinsics come from R and P:
  `R = R_right^T R_left` and `t = R_right^T (Tx_r/fx'_r - Tx_l/fx'_l, Ty_r/fy'_r - Ty_l/fy'_l, 0)`,
  which is `-R_right^T (baseline, 0, 0)` for a usual file.
- A pair whose files describe rectified images (no distortion, R the identity, P's left 3x3 equal to
  K, one `fx`, `fy` and `cy`, `Ty = 0`) is a `Rectified` pair, with `baseline = -Tx_r / fx_r` and
  `doffs = cx_r - cx_l`.

`write_camera_info_yaml()` and `write_stereo_camera_info_yaml()` write the same layout, with the
fewest digits that read back as the same doubles, so writing and reading give the same values back.
A pair that is not rectified needs both cameras' rectifications to be written, since the format holds
the extrinsics only as R and P.

The reader takes the YAML that these files use (block and flow mappings and lists, quotes,
comments) and refuses anything else (anchors, tags, block scalars, several documents, tabs)
rather than misread it. The SDK ships no YAML library, and the module adds no dependency.

## Take camera info in an operator

Frames are aligned by capture time; the camera info may come with every frame or less often.
[camera_info_input.hpp](include/holoscan/messages/camera_info_input.hpp) makes that one line in
`contract()` and one in `compute()`:

```cpp
#include <holoscan/messages/camera_info_input.hpp>

namespace hm = holoscan::messages;

class MyOp : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(left, "left");
    spec.input(right, "right");
    spec.input(camera_info, "camera_info");  // Mandatory: an unwired one fails compile().
  }
  holoscan::Contract contract() const override {
    return hm::camera_info_contract(camera_info, left, right);
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    auto info = info_.update(camera_info);  // const StereoCameraInfo*, or an error
    if (!info) return holoscan::make_unexpected(info.error());
    if (*info == nullptr) return {};        // No camera info yet: skip this frame.
    if (info_.changed()) { /* recompute what depends on the calibration */ }
    // ... use (*info)->left.fx, hm::fx_baseline(**info), ...
    return {};
  }

  holoscan::Input<holoscan::schema::ImageT> left;
  holoscan::Input<holoscan::schema::ImageT> right;
  holoscan::Input<hm::schema::StereoCameraInfoT> camera_info;

 private:
  hm::LatestCameraInfo<hm::schema::StereoCameraInfoT> info_;
};
```

`camera_info_contract(camera_info, frames...)` aligns the frames exactly by capture time (`OnEach`
for one, `OnAligned` with `Exact` for several) and, by default (`CameraInfoTiming::kLatest`), selects
the camera info `LatestWithin{max_age}` (10 s) with a `SafeDefault`; `LatestCameraInfo` validates
each new one and keeps the newest. Each frame thus uses the newest camera info received before it:
its own when the producer sends it first, the last one when it comes less often. Frames before the
first camera info are skipped, and nothing ever waits for camera info. Connect the camera info with
`hm::latest_camera_info_connection()` (drop-oldest, depth 1), so that its producer never waits either.

`CameraInfoTiming::kEveryFrame` puts the camera info into the aligned set instead: each frame waits
for the camera info with its own capture time, so a producer that sends one with every frame is
matched exactly even when it sends it after the frame. A frame whose camera info never comes is
never processed, so use it only with such a producer, and give the camera info the frames' edge
options.

Why this contract: these are the SDK 5.0's alternatives, as a graph built to compare them measured
on an RTX 5090 (40 frames every 5 ms; camera info with every frame, every 50 ms, or once):

| Contract for the camera info | With every frame, sent first | With every frame, sent after | Every 50 ms, or once |
| --- | --- | --- | --- |
| `LatestWithin` + `SafeDefault`, kept by the operator (`kLatest`) | Exact | One frame late | Every frame, the newest |
| In `OnAligned` with the frames, `Exact` (`kEveryFrame`) | Exact | Exact | Stalls: the frames wait for camera info |
| `Exact` + `ReuseLast{max_age}` | Exact | Stalls after 4 frames (lossless edges) | No frame at all |
| `LatestWithin` + `ReuseLast{max_age}` | Exact | One frame late | No frame at all when a frame comes first, else every frame |
| `LatestWithin` + `Skip` | Exact | Exact | At most one frame, then a stall |

`LatestWithin` takes each camera info once, so the operator must keep it; and `ReuseLast` and `Skip`
hold back a frame for which no camera info exists yet, and every frame behind it, for good, which a
camera that starts before the camera info triggers.

Clock domains: the camera info's and the frames' capture times must be comparable. Camera info
without a capture time, or in another clock domain, makes `update()` return an error rather than
skip every frame silently.

## Operator: `CameraInfoSourceOp`

Reads one camera_info file (a single camera) or two (a stereo pair) at construction, validates them
(the SDK constructs the operator again when a run starts, which reads them again: keep them until
the run has started), and sends the calibration on a clock: `CameraInfo` on `camera_info`, or `StereoCameraInfo` on
`stereo_camera_info` (only the one in use is declared). Each message carries the time it is sent as
capture time, in its headers and its sample. The first goes out when the run starts.

| `CameraInfoSourceConfig` | Default | |
| --- | --- | --- |
| `camera_info_path` | required | A single camera's file, or a stereo pair's left one |
| `right_camera_info_path` | empty | A stereo pair's right file |
| `period` | 1 s | How often it sends, > 0; keep it below the consumers' `max_age` |
| `count` | 0 | How many it sends; 0 until the run ends |

A consumer without room keeps the calibration it has, and the operator sends again next time.

```cpp
#include <holoscan/messages/camera_info_source_op.hpp>

auto info = graph.op<hm::CameraInfoSourceOp>(
    "camera_info", hm::CameraInfoSourceConfig{.camera_info_path = "left.yaml",
                                              .right_camera_info_path = "right.yaml"});
graph.add_flow(info->stereo_camera_info, my_op->camera_info, hm::latest_camera_info_connection());
```

## Run a graph on one worker

The SDK 5.0 runs each partition of a graph on a worker of its own, and its automatic partitioning
splits a graph wherever it has another source, forks or joins. A `CameraInfoSourceOp` next to a
camera is a second source: the camera, the camera info, and the operator that takes both each get a
worker (the demo's graph, with three sources, would run on four). The compiled plan says how many:
`plan.execution_lanes().size()`.

Across workers, the SDK runs a producer even while its consumer's lossless (`kBlocking`) queue is
full, and refuses what it sends: an emit fails with `kBackpressured`, and an allocation with
`kResourceExhausted` while every slot of the output's pool is still held, by tensors the consumers
have not finished with or by a refused one, whose slot comes back only when the producer's own GPU
work on it is done. Under GPU load that happens often. On one worker it does not: a source cannot
tick again until its consumers have taken what it sent.

So keep a lossless chain on one worker, as the demo does:

```cpp
graph.partition("my_app").include_all();  // Every operator of the graph, on one worker.
```

A source that does run on a worker of its own can treat both errors as backpressure and send the
same frame again on its next tick (`CameraInfoSourceOp` sends again on a `kBackpressured`). An
operator in the middle of a chain has taken its input by the time it emits and cannot send it
again: it returns the error, which fails the run.

## Send and receive point clouds

[point_cloud.hpp](include/holoscan/messages/point_cloud.hpp): a producer declares the output, then
for each cloud allocates every tensor from the runtime's pool, fills the ones it carries and sends
it. A consumer declares a plain input and checks what it gets.

```cpp
#include <holoscan/messages/point_cloud.hpp>

// setup()
hm::declare_point_cloud_output(spec, cloud, "cloud", {.max_points = 640 * 480, .colors = true});

// compute()
const auto shape = hm::organized_cloud(480, 640);
auto loan = hm::allocate_point_cloud(cloud, shape, {.colors = true}, context.cuda_stream());
loan->write_and_commit(hm::PointCloudFields::points, context.cuda_stream(), fill_points);
loan->write_and_commit(hm::PointCloudFields::colors, context.cuda_stream(), fill_colors);
hm::emit_point_cloud(cloud, std::move(*loan), header, shape, /*valid_count=*/-1, emit_options);

// A consumer's compute()
auto sample = cloud_in.receive();
if (auto checked = hm::check_point_cloud(sample->data); !checked) return checked;
if (hm::is_present(sample->data.colors)) { /* ... */ }
```

The fields that `allocate_point_cloud()` leaves absent are `[0]` and already committed. Device
outputs need their GPU bound at compile time, as for any device tensor output
(`CompileOptions::deployment.bind_tensor_output_device()`).

For tests, [synthetic_calibration.hpp](include/holoscan/messages/synthetic_calibration.hpp) makes a
raw stereo pair with known intrinsics, distortion, extrinsics and rectifications, for any lens model.

## Requirements

- Holoscan SDK 5.0 (Core), CUDA 13, CMake 4.0 or later and a C++20 compiler.

The module downloads nothing and needs no models or data.

## Build, test and install

From the repository root, in the repository's CI image or with the requirements above:

```bash
cmake -S modules/messages -B build/messages -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build/messages
ctest --test-dir build/messages --output-on-failure
cmake --install build/messages --prefix "$PWD/install"
```

| CMake option | Default | |
| --- | --- | --- |
| `BUILD_TESTING` | `OFF` | Build the tests, when the module is the top-level project |
| `HOLOSCAN_MESSAGES_BUILD_TESTING` | `OFF` | Build the tests when another project includes the module (`add_subdirectory()`, `FetchContent`); its `BUILD_TESTING` and `include(CTest)` do not |

The tests:

| Test | Needs | Checks |
| --- | --- | --- |
| `messages_camera_info` | Nothing | `validate()` of cameras and pairs, each refusal naming its field; `depth_from_disparity()`, `same_calibration()`, the synthetic pairs |
| `messages_yaml` | Nothing | The YAML forms read and refused (with line numbers); a ROS calibration read; write and read round trips of a camera and a pair with every lens model, raw, undistorted and rectified; the extrinsics recovered from R and P against known values (1e-9 m, 1e-12 rotation); missing keys, wrong matrix sizes, wrong coefficient counts, R that is not a rotation, unknown models and keys, a projection without a baseline; files |
| `messages_point_cloud_layout` | Nothing | The point cloud layouts accepted and refused |
| `messages_message_graph` | GPU | `CameraInfo` and `StereoCameraInfo` sent through a graph with and without their optional fields and with the longest frame ids, field for field; `PointCloud` in device and host memory, organized, unorganized and empty, with optional tensors present and absent, value for value; the conversions' refusals |
| `messages_contract` | GPU | The consumer contract in running graphs: stereo and single-camera frames with `CameraInfoSourceOp` every 20 ms, frames before the first camera info, camera info with every frame before (`kLatest`) and after (`kEveryFrame`) the frames, a calibration that changes, an invalid camera info that fails the run, and an unwired camera info input that fails compilation |
| `messages_camera_info_source_op` | GPU | `CameraInfoSourceOp` for one file and a pair: the files' calibration, stamped on its period, a limited count and an unlimited one; the refusals at construction |

The SDK-free tests are also a standalone project, [tests/host/](tests/host/), which the hosted CI job
builds without the SDK. [`tests/installed_consumer/`](tests/installed_consumer/) is a separate project
that uses the installed package: it writes a camera_info file and runs `CameraInfoSourceOp` into an
operator that takes frames and the camera info with the contract, on one worker.

## Debian package

Inside the repository's CI image, with the module's folder as the Holoscan CLI project root:

```bash
holoscan --project-root modules/messages package holoscan-messages --language cpp \
  --pkg-generator DEB --local
```

This writes `holoscan-messages_1.0.0_<arch>.deb` to the current directory. It installs to `/usr`
and depends on `holoscan (>= 5.0.0)`, so `apt install ./holoscan-messages_1.0.0_<arch>.deb` needs
NVIDIA's CUDA repository.

## Licenses

Apache-2.0. See [NOTICE](NOTICE).
