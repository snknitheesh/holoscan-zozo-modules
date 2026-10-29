# holoscan-fiducials

A Holoscan SDK 5 Module that detects [ArUco](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)
markers and ChArUco boards. `ArucoDetectorOp` takes an image and emits the markers it contains,
with sub-pixel corners and, given the camera's intrinsics, each marker's 6-DoF pose.
`CharucoDetectorOp` emits one view of a ChArUco board: its chessboard corners with their ids,
sub-pixel image positions and board positions, the board's pose and a quality summary, which is
the observation a camera calibration takes. Both run on the CPU or the GPU (CUDA).
`FiducialOverlayOp` turns their messages into HoloViz layers. The
[Fiducial Detection demo](../../applications/fiducial_detection/) runs all three end to end.

| | |
| --- | --- |
| CMake package | `find_package(holoscan-fiducials [COMPONENTS overlay])` |
| Targets | `holoscan::fiducials`, `holoscan::fiducials_overlay` |
| Headers | `<holoscan/fiducials/...>` |
| C++ namespace | `holoscan::fiducials` |
| Debian package | `holoscan-fiducials` |
| Platforms | x86_64; aarch64 (Jetson Thor and Orin, JetPack 7) |

## Operators

### `ArucoDetectorOp`

| Port | Type | |
| --- | --- | --- |
| `image` (input) | `holoscan::schema::ImageT` | MONO8, RGB8, BGR8, RGBA8 or BGRA8, in host or CUDA device memory |
| `detections` (output) | `holoscan::fiducials::schema::FiducialDetectionsT` | One message per image, without a board |

`ArucoDetectorConfig` holds the backend, the `DetectorOptions` (the dictionary first) and, for
poses, the camera's `intrinsics` and the markers' `marker_length`, the side of the black border in
meters.

### `CharucoDetectorOp`

| Port | Type | |
| --- | --- | --- |
| `image` (input) | `holoscan::schema::ImageT` | As above |
| `detections` (output) | `holoscan::fiducials::schema::FiducialDetectionsT` | One message per image, with the board's view, even when none of it was found |

`CharucoDetectorConfig` holds the backend, the board (`CharucoBoardConfig`: `squares_x`,
`squares_y`, `square_length`, `marker_length`, `dictionary`, `legacy_pattern`,
`first_marker_id`), the `DetectorOptions` (whose dictionary is replaced by the board's),
`CharucoOptions::min_markers` and, for the board's pose, the camera's `intrinsics`.

Boards are laid out as OpenCV 4.6 and later lays them out (`cv::aruco::CharucoBoard`): square
`(x, y)` holds a marker when `x + y` is odd, the markers take the ids from `first_marker_id` up,
row by row, and `legacy_pattern` selects the layout before 4.6 for boards with an even number of
rows. A board printed by OpenCV with the same parameters is detected as is.

Each chessboard corner is predicted from the homographies of the markers in the squares that meet
there and refined to the saddle point of the image, by OpenCV `cornerSubPix`'s iteration with each
gradient weighted by its magnitude rather than its square (which pulls sharp corners toward pixel
centers by up to 0.08 px). A corner is reported when at least `min_markers` (1 or 2, default 2,
OpenCV's `minMarkers`) of its two markers were found, and only when the image around it turns from
dark to bright twice and its opposite sectors match, so the edge of something in front of the board
does not pass for a corner.

### `FiducialOverlayOp`

| Port | Type | |
| --- | --- | --- |
| `detections` (input) | `FiducialDetectionsT` | From either detector |
| `overlay` (output) | `holoscan::schema::ImageT` | A HoloViz RGBA8 image layer: markers' outlines in yellow, with the first corner marked in red; chessboard corners as magenta crosses; markers' and the board's X, Y and Z axes as red, green and blue arrows, when there are poses and intrinsics |
| `labels` (output) | `holoscan::Tensor` | A HoloViz text layer: `id N` per marker, and `corners found/total` for a board |

The overlay emits the labels first and then the drawing, both stamped with the frame's capture time,
so a HoloViz `FrameTrigger` on the overlay layer draws each frame's labels and drawing together over
the image they came from. Keep at least four images queued on the HoloViz image edge; the demo shows
the wiring.

## The message

Each `FiducialDetections` message ([schema](schemas/fiducial_detections.fbs)) holds the frame
number, the image header's `device_sequence`, the image size, the backend that ran, the dictionary,
the detection time, and per marker its id, corrected bits, four corners and, when known, its pose.
A message carries at most 256 markers: `dropped` counts the rest, and `truncated` says the detector
ran out of candidate capacity (4096 dark regions a frame), so markers may be missing; the operator
then also logs a warning, at most every five seconds. From `CharucoDetectorOp`, `board` holds the
board's parameters, the chessboard corners (`id`, `image` in pixels, `board` in meters), its pose,
and the quality: `corners_total`, `markers_found`, `markers_total`, and `coverage`, the convex hull
of the corners found as a fraction of the image. `markers_of()` and `board_of()` in
`<holoscan/fiducials/fiducial_messages.hpp>` read a message back into `Marker`s and a
`CharucoObservation`.

Chessboard corner `k` of a board is the inner corner at column `k % (squares_x - 1)` and row
`k / (squares_x - 1)`, at `((column + 1) * square_length, (row + 1) * square_length, 0)` in the
board frame, as in OpenCV.

### Conventions

- **Pixels.** Pixel centers are at integer coordinates, as in OpenCV and ROS `CameraInfo`. Give the
  intrinsics that way: `fx`, `fy`, `cx`, `cy` and OpenCV's distortion `k1, k2, p1, p2, k3`
  (`CameraIntrinsics::distortion`, zero for none). Corners are reported where they are in the
  image, distorted; poses account for the distortion.
- **Corners.** A marker's corners are the outer corners of its black border, in OpenCV's order:
  top-left, top-right, bottom-right, bottom-left as printed.
- **Poses.** The same convention as `holoscan-apriltag`: a pose places the target's frame in the
  camera frame (x right, y down, z forward), with the target's x right and y down as printed and
  z into the target. A marker's frame is centered on it, so its corners are at
  `(-l/2, -l/2, 0)`, `(l/2, -l/2, 0)`, `(l/2, l/2, 0)` and `(-l/2, l/2, 0)` for a side `l`. A board's
  frame has its origin at the board's top-left corner, x along `squares_x` and y along `squares_y`.
  OpenCV uses the same board frame; its single-marker poses have y up and z out of the marker.
- **Errors.** A pose's `reprojection_error` is the root mean square distance, in pixels, between
  the points it was fitted to and where it projects them.

Poses are fitted on the host, the same way for both backends: from the homography of the points,
refined by Levenberg-Marquardt on the reprojection error from both of the poses a plane seen from
afar can take (the planar ambiguity), keeping the better one.

## Dictionaries

`DetectorOptions::dictionary` and `CharucoBoardConfig::dictionary` take OpenCV's names:

| Dictionary | Cells | Markers | Bits corrected by default |
| --- | --- | --- | --- |
| `DICT_4X4_50`, `_100`, `_250`, `_1000` | 4 x 4 | 50 to 1000 | 0 |
| `DICT_5X5_50`, `_100`, `_250`, `_1000` | 5 x 5 | 50 to 1000 | 1 |
| `DICT_6X6_50`, `_100`, `_250`, `_1000` | 6 x 6 | 50 to 1000 | 3, 3, 3, 2 |
| `DICT_7X7_50`, `_100`, `_250`, `_1000` | 7 x 7 | 50 to 1000 | 5, 4, 4, 3 |
| `DICT_ARUCO_ORIGINAL` | 5 x 5 | 1024 | 0 |
| `DICT_ARUCO_MIP_36h12` | 6 x 6 | 250 | 7 |
| `DICT_APRILTAG_16h5`, `_25h9`, `_36h10`, `_36h11` | 4 x 4 to 6 x 6 | 30, 35, 2320, 587 | 1, 2, 2, 3 |

The codes are OpenCV 4.12's, so markers printed with OpenCV (or the ArUco library) keep their ids.
A dictionary corrects `floor(max_correction_bits * error_correction_rate)` bits, as OpenCV does,
with `error_correction_rate` 0.6 by default; `max_correction_bits` is OpenCV's for the ArUco
dictionaries and `(d - 1) / 2` for an AprilTag family of distance `d`, for which OpenCV stores 0.
The AprilTag tags are stored as the AprilTag library draws them, which OpenCV turns a half turn:
the ids agree with OpenCV's, and the corners and poses with `holoscan-apriltag`'s. Marker 1023 of
`DICT_ARUCO_ORIGINAL` looks the same turned a half turn, so its corner order is ambiguous.

## Backends

| Backend | Runs on | Notes |
| --- | --- | --- |
| `Backend::kCpu` | CPU | One thread. |
| `Backend::kCuda` | GPU | Thresholding, connected components, quads, edge refinement, decoding and chessboard corners on the GPU; only the markers and corners are copied to the host. |
| `Backend::kAuto` | One of the above | Chosen once, at start: `cuda` where a CUDA device is usable, else `cpu`. Opt-in; the default is `kCuda`. |

Both backends run the same code for every step ([`src/detection_core.hpp`](src/detection_core.hpp)):
the CUDA backend in kernels, a warp per candidate and per chessboard corner, the CPU backend in
loops that add a warp's lanes in the warp's order. They find the same markers and corners, to the
bit; the tests check it. The CPU path needs no OpenCV, so the module builds anywhere the SDK does.

The detector thresholds each pixel against the middle of the darkest and brightest values in the
3 x 3 tiles around it, joins dark pixels into 8-connected components, and takes each component's
convex corners as a quad: the pixel farthest from its centroid, the one farthest from that, and
the two farthest from the line between them. It reads the quad's cells, rejects it unless the
border is black and there is white around it, fits each edge to sub-pixel accuracy from where the
image rises across it, reads the cells again, and looks the code up in all four rotations.

With `kAuto`, the operator logs its choice once, at start, and `resolved_backend()` returns it
(`kAuto` before start); for example `ArucoDetectorOp: backend auto chose cuda: a CUDA device is
usable`. `holoscan::fiducials::resolve_auto()` is the rule itself, and `probe_platform()` and
`resolve_backend()` in `<holoscan/fiducials/detector_backend.hpp>` apply it.

| Where the image goes | Host image | CUDA device image |
| --- | --- | --- |
| `cpu` | Read in place (color is converted to gray on the CPU). | Gray copied to the host. |
| `cuda` | Gray copied to the GPU. | Read in place (color is converted to gray on the GPU); no copy. |

When the image is not where the backend reads it, the operator says so once, at info level, at the
first such frame: which backend, which direction, the bytes per frame, and how to avoid the copy.
For example:

```text
ArucoDetectorOp (cuda backend): the images are in host memory and the backend reads CUDA device memory, so each frame's gray image is copied host to device, 921600 bytes (0.9 MB) per frame; to avoid the copy, send CUDA device images or use the cpu backend
```

`DetectorOptions` also has `error_correction_rate` (0..1), `tile_size` (the threshold tiles, 2..64
pixels, default 8), `min_contrast` (gray levels, default 20), `min_marker_px` (the shortest side
reported, default 10) and `refine_corners` (default on).

## Accuracy and speed

On synthetic 1280x720 frames rendered with 8 x 8 supersampled edges and light noise (the module's
tests), both backends alike:

| | Corners (mean / max) | Pose (translation / rotation, max) |
| --- | --- | --- |
| Markers, about 100 to 140 px, tilted up to 40 degrees | 0.021 / 0.08 px | 0.05 % / 0.14 degrees |
| The same, Gaussian blur 1 px and noise 6 | 0.052 / 0.17 px | 0.07 % / 0.26 degrees |
| The same, blur 2 px and noise 4 | 0.078 / 0.19 px | 0.13 % / 0.46 degrees |
| ChArUco chessboard corners, 5x7 to 8x6 boards | 0.036 / 0.12 px | 0.03 % / 0.04 degrees |
| The same, blur 1 px and noise 5 | 0.082 / 0.19 px | 0.03 % / 0.05 degrees |

No marker was missed, and none reported in clutter, at the image's border or with the wrong
dictionary. Heavy blur next to the markers in a board (2 px against 6 px margins) degrades the
chessboard corners to about 0.2 px.

Detection time at 1280x720, from the gray image to the markers or chessboard corners on the host,
the mean over 20 frames (`fiducials_backends` prints it):

| | Six markers | 9x6 ChArUco board (27 markers, 40 corners) |
| --- | --- | --- |
| x86_64, RTX 5090, `cuda` | 1.3 ms | 1.7 ms |
| x86_64, `cpu` | 6.6 ms | 6.9 ms |
| Jetson Orin, `cuda` | ORIN_CUDA_MARKERS | ORIN_CUDA_BOARD |
| Jetson Orin, `cpu` | ORIN_CPU_MARKERS | ORIN_CPU_BOARD |

## Use it in your graph

```cmake
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core sensor_io)
find_package(holoscan-fiducials 1.0 REQUIRED CONFIG)
target_link_libraries(my_app PRIVATE holoscan::fiducials)   # add holoscan::fiducials_overlay for the overlay
```

```cpp
#include <holoscan/fiducials/charuco_detector_op.hpp>
#include <holoscan/fiducials/fiducial_messages.hpp>

namespace fd = holoscan::fiducials;

auto board = graph.op<fd::CharucoDetectorOp>(
    "charuco", fd::CharucoDetectorConfig{
                   .backend = fd::Backend::kCuda,
                   .board = {.squares_x = 7, .squares_y = 5, .square_length = 0.04,
                             .marker_length = 0.03, .dictionary = "DICT_5X5_100"},
                   .intrinsics = fd::CameraIntrinsics{910.5, 910.5, 639.5, 359.5, {}}});
graph.add_flow(camera->image, board->image);              // Output<holoscan::schema::ImageT>
graph.add_flow(board->detections, calibrator->views);     // Output<fd::schema::FiducialDetectionsT>
// In the consumer: fd::board_of(message)->corners, each {id, image, board}.
```

The detectors need no Holoscan graph: `create_backend()` in
`<holoscan/fiducials/detector_backend.hpp>` finds markers and refines chessboard corners,
`observe_charuco()` in `<holoscan/fiducials/charuco.hpp>` turns markers into a board's view, and
`<holoscan/fiducials/pose.hpp>` projects points and fits poses. `<holoscan/fiducials/synthetic_scene.hpp>`
renders scenes of markers and boards at known poses, with blur, noise, occluders and mirroring, and
checks detections against them, for tests.

## Requirements

- Holoscan SDK 5.0 (Core and Sensor I/O; Visualization for the overlay; Testing for the tests),
  CUDA 13, CMake 4.0 or later and a C++20 compiler. Nothing else: no OpenCV.

## Build, test and install

From the repository root, in the repository's CI image or with the requirements above:

```bash
cmake -S modules/fiducials -B build/fiducials -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build/fiducials
ctest --test-dir build/fiducials --output-on-failure
cmake --install build/fiducials --prefix "$PWD/install"
```

| CMake option | Default | |
| --- | --- | --- |
| `HOLOSCAN_FIDUCIALS_BUILD_OVERLAY` | `ON`; inside another project, `OFF` if `HOLOSCAN_ZOZO_MODULES_OPERATORS` is set without `FiducialOverlayOp` | Build `holoscan::fiducials_overlay` (needs Holoscan Visualization) |
| `BUILD_TESTING` | `OFF` | Build the tests, when the module is built on its own |
| `HOLOSCAN_FIDUCIALS_BUILD_TESTING` | `OFF` | Build the tests inside another project |
| `CMAKE_CUDA_ARCHITECTURES` | `native` (`all-major` for the package) | GPUs to compile for, unless already set |

Another project can also build the module as part of its own build, with `add_subdirectory()` or
`FetchContent`, and link `holoscan::fiducials` (and `holoscan::fiducials_overlay`) without
`find_package()`; the module then leaves the project's `BUILD_TESTING` and `BUILD_SHARED_LIBS`
alone.

The tests generate all their data, synthetic scenes with ground truth:

| Test | Needs | Checks |
| --- | --- | --- |
| `fiducials_backends` | GPU | The `cuda` backend against the `cpu` backend to the bit (markers, corners, chessboard corners) on clean, blurred, noisy, occluded and mirrored scenes; `cuda` against ground truth; candidate overflow; `auto`; the timings above |
| `fiducials_operators` | GPU | The three operators through the SDK's operator test harness: every encoding and backend, host and device images, size changes, ChArUco views against ground truth, `auto` and its log, the copy log (once when frames cross between host and device, never otherwise), provenance, bad input and configuration, and the overlay's layers |
| `fiducials_marker_detection` | Nothing | The `cpu` detector on ten dictionaries: every marker, corner and pose against ground truth; blur and noise; partial occlusion; no false positives on clutter; markers at and across the image's border; a mirrored image; the wrong dictionary; the options |
| `fiducials_charuco` | Nothing | Board geometry against OpenCV's; four boards (one legacy, one from id 30): every corner's id, board position and image position, poses and quality; other boards; blur and noise; occlusion and `min_markers`; a board cut by the image's edge; bad boards |
| `fiducials_pose` | Nothing | The pose convention, undistortion, exact poses with distortion, the planar ambiguity, noisy corners, degenerate input |
| `fiducials_dictionary` | Nothing | Every dictionary's size and correction, codes against OpenCV and the AprilTag library, prefixes, rotations and uniqueness |
| `fiducials_overlay_drawing` | Nothing | The overlay's drawing |
| `fiducials_backend_resolution` | Nothing | `auto`'s choice and the backend names |

The tests marked "Nothing" are also a project of their own, [`tests/host/`](tests/host/), which the
hosted CI job builds and runs. [`tests/installed_consumer/`](tests/installed_consumer/) is a
separate project that uses the installed package: a small graph that runs both detectors on every
backend.

## Debian package

Inside the repository's CI image, with the module's folder as the Holoscan CLI project root:

```bash
holoscan --project-root modules/fiducials package holoscan-fiducials --language cpp \
  --pkg-generator DEB --local
```

This builds for every major GPU architecture and writes `holoscan-fiducials_1.0.0_<arch>.deb`. It
installs to `/usr` and depends on `holoscan (>= 5.0.0)`.

## Licenses

Apache-2.0. The marker codes are generated from OpenCV's predefined dictionaries (Apache-2.0),
which carry the ArUco library's dictionaries and the AprilTag families (BSD-2-Clause); OpenCV is
neither linked nor distributed. See [NOTICE](NOTICE).
