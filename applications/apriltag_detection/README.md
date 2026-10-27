# AprilTag Detection

The demo of the [holoscan-apriltag](../../modules/apriltag/) module. It renders synthetic scenes of
AprilTags at known poses, fans each frame out to an `AprilTagDetectorOp` per selected backend
(`cpu`, `cuda`, `vpi`), checks every detection against the scene's ground truth, and can show the
first backend's detections on a HoloViz display through `AprilTagOverlayOp`. It uses the installed
module the way any other project would: `find_package(holoscan-apriltag)`.

## Demo

- [Synthetic scene](tests/video/apriltag_synthetic.mp4) (12 s): four `tagStandard41h12` tags
  moving in a rendered scene, detected by the `cuda` backend. Each tag is drawn with its X, Y and Z
  axes in red, green and blue, and labelled with its ID and distance.
- [Real video](tests/video/apriltag_real.mp4) (22 s): a phone recording of two printed
  `tagStandard41h12` tags, each frame passed through `AprilTagDetectorOp` (`cuda`) and the overlay.
  Without the phone camera's intrinsics there is no pose, so the axes lie in the image plane.

Both are the frames HoloViz rendered, recorded one by one rather than captured from the screen.

## Requirements

- holoscan-apriltag, installed (see [its README](../../modules/apriltag/README.md)), and what it
  needs: Holoscan SDK 5.0 with Core, Sensor I/O and Visualization (Testing for the tests), CUDA 13,
  CMake 4.0 or later and a C++20 compiler.
- A display for the HoloViz window; without one, run with `--display false`.

## Build and run

From the repository root, after installing the module into `install/`:

```bash
cmake -S applications/apriltag_detection -B build/apriltag_detection -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PWD/install"
cmake --build build/apriltag_detection
build/apriltag_detection/apriltag_detection --backend all --count 300          # synthetic mode
build/apriltag_detection/apriltag_detection --backend all --count 30 --period-ms 10 --display false  # ci mode
```

`--help` lists the options, for example `--backend cuda,vpi`, `--family tagStandard41h12` and
`--device-input true`. `-DAPRILTAG_DETECTION_ENABLE_VISUALIZATION=OFF` builds it without HoloViz.
The modes in [metadata.json](metadata.json) are the two commands above: `synthetic`, displayed when
there is a display, and `ci`, a finite headless check that CI runs.

On a Jetson, the demo links the Vulkan loader first: once VPI has loaded Tegra's EGL driver,
HoloViz can no longer create its Vulkan instance. In a headless container there, leave `DISPLAY`
unset, even to an empty value, or VPI fails with `EGL_NOT_INITIALIZED`.

## Output

Each run prints a line per backend with its frame and tag counts, detection latency (mean, median,
95th percentile, maximum) and ground-truth errors, then `PASS`, `FAIL`, or `STOPPED` when an
interrupt or a closed window ends it early. It exits 0 on `PASS`, 2 for invalid options, else 1:

```text
apriltag_detection: backend=cpu frames=30/30 tags=120 ... truth: missed=0 extra=0 misplaced=0/120 ...
apriltag_detection: backend=cuda frames=30/30 tags=120 ... truth: missed=0 extra=0 misplaced=0/120 ...
apriltag_detection: backend=vpi-cpu frames=30/30 tags=120 ... truth: missed=0 extra=0 misplaced=0/120 ...
apriltag_detection: verdict=PASS
```

The display draws each tag's axes as red, green and blue arrows and a dark blue label with its id
and distance. HoloViz draws when the overlay's axes arrive, over the image they were detected in.

## Performance

Median time from receiving an image to having detections and poses, for 300 synthetic 1280x720
frames with four `tag36h11` tags, one backend at a time (SDK 5.0.0, CUDA 13.2, VPI 4.0.5) on an
RTX 5090 and a Ryzen 9 9950X3D. Every backend found all 1200 tags and nothing else.

```bash
build/apriltag_detection/apriltag_detection --backend cuda --count 300 --period-ms 1 --display false
```

| Backend | Host RGB input | GPU RGB input (`--device-input true`) | Largest corner error |
| --- | --- | --- | --- |
| `cpu` (one thread) | 21.3 ms | 21.6 ms | 0.22 px |
| `cuda` | 5.1 ms | 4.7 ms | 0.22 px |
| `vpi` (VPI's CPU path on x86_64, multi-threaded, so it varies with the host's load) | 6.6 ms | 8.5 ms | 0.40 px |

On a Jetson Thor (JetPack 7, SDK 5.0.0, CUDA 13.0), the same detector code had medians at 1280x720
of 37.9 ms (`cpu`), 14.2 ms (`cuda`) and 11.7 ms (`vpi` on the PVA).

## Tests

With `-DBUILD_TESTING=ON`, `ctest` checks the reporter through the SDK's operator test harness,
and the demo's runs: every backend against ground truth from host and GPU memory, the exit statuses
for invalid options, a failed run and a stopped one, a run without managed signal handling, and
that the display graph compiles.

## Licenses

Apache-2.0. The module's README lists the licenses of what it uses.
