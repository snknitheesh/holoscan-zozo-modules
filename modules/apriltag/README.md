# holoscan-apriltag

A Holoscan SDK 5 Module that detects [AprilTags](https://april.eecs.umich.edu/software/apriltag)
and estimates their 6-DoF poses. `AprilTagDetectorOp` takes an image and emits the tags it
contains, found on the CPU (AprilTag 3), the GPU (CUDA) or the PVA of Jetson Orin and Thor (NVIDIA
VPI 4). `AprilTagOverlayOp` turns those detections into HoloViz layers. The
[AprilTag Detection demo](../../applications/apriltag_detection/) runs both end to end.

| | |
| --- | --- |
| CMake package | `find_package(holoscan-apriltag [COMPONENTS overlay])` |
| Targets | `holoscan::apriltag`, `holoscan::apriltag_overlay` |
| Headers | `<holoscan/apriltag/...>` |
| C++ namespace | `holoscan::apriltag` |
| Debian package | `holoscan-apriltag` |
| Platforms | x86_64; aarch64 (Jetson Thor, JetPack 7) |

## Operators

### `AprilTagDetectorOp`

| Port | Type | |
| --- | --- | --- |
| `image` (input) | `holoscan::schema::ImageT` | MONO8, RGB8, BGR8, RGBA8 or BGRA8, in host or CUDA device memory |
| `detections` (output) | `holoscan::apriltag::schema::AprilTagDetectionsT` | One message per image |

`AprilTagDetectorConfig` selects the backend, the detector options and, for poses, the camera's
intrinsics and the tag's size:

| Backend | Detection runs on | Notes |
| --- | --- | --- |
| `Backend::kCpu` | CPU | The AprilTag 3 reference detector, with its default of one thread. |
| `Backend::kCuda` | GPU | Every stage on the GPU; only the final detections are copied to the host. |
| `Backend::kVpi` | PVA, or CPU | NVIDIA VPI 4's detector: on the PVA where there is one and the image suits it, otherwise on the CPU. x86_64 hosts have no PVA. |

Every backend accepts every AprilTag 3 family (`tag36h11`, `tag25h9`, `tag16h5`, `tag36h10`,
`tagStandard41h12`, `tagStandard52h13`, `tagCircle21h7`, `tagCircle49h12`, `tagCustom48h12`).
Poses are estimated on the host from the corners, the same way for every backend.
`DetectorOptions::decimate`, `refine_edges` and `decode_sharpening` tune `cpu` and `cuda`; VPI
ignores them. `max_hamming` 2 is refused for the 48-, 49- and 52-bit families, whose decode tables
would take gigabytes. The PVA takes image sides from 256 to 4096 pixels, except one past a multiple
of 128.

| Where the image goes | Host image | CUDA device image |
| --- | --- | --- |
| `cpu`, `vpi` on the CPU | Read in place (color is converted to gray on the CPU). | Gray copied to the host. |
| `cuda` | Gray copied to the GPU. | Read in place (color is converted to gray on the GPU); no copy. |
| `vpi` on the PVA | Gray copied into a VPI image. | Gray copied into a VPI image on the GPU. |

Each `AprilTagDetections` message ([schema](schemas/apriltag_detections.fbs)) holds the frame
number, the image header's `device_sequence`, the image size, the backend that ran (`cpu`, `cuda`,
`vpi-pva` or `vpi-cpu`), the family, the detection time, and per tag its id, corrected bits,
decision margin, center, corners (from the tag's bottom-left, counter-clockwise as printed) and,
with intrinsics, its pose. A message carries at most 64 tags: `dropped` counts the rest, and
`truncated` says the detector ran out of capacity before decoding, so tags may be missing; the
operator also logs a warning, at most every five seconds.

Poses follow AprilTag's convention: they place the tag in the camera frame (x right, y down, z
forward), with the tag's x right and y down as printed and z into the tag. Give the intrinsics with
pixel centers at integer coordinates, as in OpenCV and ROS `CameraInfo`, and the tag size as the
side of the square the detector finds: the black border for `tag36h11`.

### `AprilTagOverlayOp`

| Port | Type | |
| --- | --- | --- |
| `detections` (input) | `AprilTagDetectionsT` | From `AprilTagDetectorOp` |
| `labels` (output) | `holoscan::Tensor` | A HoloViz text layer: per tag, a dark blue `id N` label, with the distance when there is a pose |
| `axes` (output) | `holoscan::schema::ImageT` | A HoloViz RGBA8 image layer: per tag, its X, Y and Z axes as red, green and blue arrows on a transparent background (X and Y in the image plane without a pose) |

The overlay emits the labels first and then the axes, both stamped with the frame's capture time,
so a HoloViz `FrameTrigger` on the axes layer draws each frame's labels and axes together over the
image they came from. Keep at least four images queued on the HoloViz image edge so that the
frame's image is still there when its detections arrive; the demo shows the full wiring.

## Use it in your graph

```cmake
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core sensor_io)
find_package(holoscan-apriltag 1.0 REQUIRED CONFIG)
target_link_libraries(my_app PRIVATE holoscan::apriltag)   # add holoscan::apriltag_overlay for the overlay
```

```cpp
#include <holoscan/apriltag/apriltag_detector_op.hpp>

namespace at = holoscan::apriltag;

auto detector = graph.op<at::AprilTagDetectorOp>(
    "apriltag", at::AprilTagDetectorConfig{
                    .backend = at::Backend::kCuda,
                    .options = {.family = "tag36h11"},
                    .intrinsics = at::CameraIntrinsics{910.5, 910.5, 639.5, 359.5},
                    .tag_size = 0.08});
graph.add_flow(camera->image, detector->image);              // Output<holoscan::schema::ImageT>
graph.add_flow(detector->detections, consumer->detections);  // Output<at::schema::AprilTagDetectionsT>
```

`holoscan-apriltag_WITH_VPI` tells whether the installed module has the `vpi` backend, and
`holoscan::apriltag::available_backends()` lists the backends at run time. The backends themselves
need no Holoscan graph: `holoscan::apriltag::create_backend()` in
`<holoscan/apriltag/detector_backend.hpp>`. `<holoscan/apriltag/synthetic_scene.hpp>` renders scenes
of tags at known poses and checks detections against them, for tests.

## Requirements

- Holoscan SDK 5.0 (Core and Sensor I/O; Visualization for the overlay; Testing for the tests),
  CUDA 13, CMake 4.0 or later and a C++20 compiler.
- [AprilTag 3](https://github.com/AprilRobotics/apriltag) 3.4 or later
  ([BSD-2-Clause](https://github.com/AprilRobotics/apriltag/blob/master/LICENSE.md)): an installed
  copy, or release 3.4.5 downloaded at configure time, checked against its SHA-256 and linked into
  the library.
- Optional: NVIDIA VPI 4 (`vpi4-dev`) for the `vpi` backend. The repository's CI images install it
  from NVIDIA's Jetson package repository; see [`.github/docker/`](../../.github/docker/).

## Build, test and install

From the repository root, in the repository's CI image or with the requirements above:

```bash
cmake -S modules/apriltag -B build/apriltag -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build/apriltag
ctest --test-dir build/apriltag --output-on-failure
cmake --install build/apriltag --prefix "$PWD/install"
```

| CMake option | Default | |
| --- | --- | --- |
| `HOLOSCAN_APRILTAG_WITH_VPI` | `ON` | Build the `vpi` backend when VPI 4 is found |
| `HOLOSCAN_APRILTAG_BUILD_OVERLAY` | `ON` | Build `holoscan::apriltag_overlay` (needs Holoscan Visualization) |
| `BUILD_TESTING` | `OFF` | Build the tests |
| `CMAKE_CUDA_ARCHITECTURES` | `native` (`all-major` for the package) | GPUs to compile for |

The tests:

| Test | Needs | Checks |
| --- | --- | --- |
| `apriltag_backend_accuracy` | GPU | Every backend against ground truth, from host and GPU memory, for four families (among them `tagCircle21h7`) and two-bit correction; the CUDA backend against the CPU on tags cut by the image edge; that frames over a capacity report what they dropped; on the Jetson CI runner, that `vpi` runs on the PVA |
| `apriltag_operators` | GPU | Both operators through the SDK's operator test harness: every encoding and backend, size changes, options, provenance, CUDA streams, bad input and configuration, and the overlay's layers |
| `apriltag_tag_axes` | Nothing | The overlay's axis drawing; also built on its own from `tests/host/` by the hosted CI job |

[`tests/installed_consumer/`](tests/installed_consumer/) is a separate project that uses the
installed package: a small graph that runs `AprilTagDetectorOp` on every installed backend. Build
it with the install prefix on `CMAKE_PREFIX_PATH` and run `ctest` there.

## Debian package

Inside the repository's CI image, with the module's folder as the Holoscan CLI project root:

```bash
holoscan --project-root modules/apriltag package holoscan-apriltag --language cpp \
  --pkg-generator DEB --local
```

`--language cpp` selects the module rather than its package entry, which has the same name. This
builds for every major GPU architecture and writes `holoscan-apriltag_1.0.0_<arch>.deb` to the
current directory. It installs to `/usr` and depends on `holoscan (>= 5.0.0)` and, when built with
VPI, `libnvvpi4`, so `apt install ./holoscan-apriltag_1.0.0_<arch>.deb` needs NVIDIA's CUDA
repository and, for `libnvvpi4`, the Jetson repository that the CI images use.

## Licenses

Apache-2.0. AprilTag 3, when built into the library, is BSD-2-Clause; its notice is installed as
`share/doc/holoscan-apriltag/apriltag-LICENSE.md`. VPI is under NVIDIA's own terms, installed as
`/opt/nvidia/vpi4/doc/VPI_EULA.txt`. See [NOTICE](NOTICE).
