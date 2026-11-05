# Stereo Disparity

The demo of the [holoscan-stereo-disparity](../../modules/stereo-disparity/) module. A stereo
source fans each frame out to a `StereoDisparityOp` per selected backend (`cuda`, `ofa`, `pva`, or
`auto`, which the module picks for the platform), and a reporter per backend checks its result:

- **Synthetic**: textured planes at known depths, rendered by the module's synthetic scene, so
  every disparity is checked against the truth.
- **Recording**: a recorded, rectified stereo clip, played from its folder, so the computed depth
  can be checked against the recording's own depth image.

Either way the pair's calibration comes from ROS camera_info files, read by holoscan-messages'
`CameraInfoSourceOp`: the synthetic scene's are written for the run, a recording has its own. The
source takes it, holds its first frame until it has come, and sends it with every frame, as a
rectifier does, so every matcher takes each frame's own (`CameraInfoTiming::kEveryFrame`). HoloViz shows the first backend's disparity, colorized from blue (far) to dark red
(near), black where it is invalid; for a color synthetic pair, beside the left image. The display
never holds back the matchers: its edges drop the oldest sample, keeping the newest disparity and,
the one exception to keeping one, the four newest left images, so that each disparity's own left
image is still there when HoloViz pairs them. The demo uses
the installed modules the way any other project would: `find_package(holoscan-stereo-disparity)`
and `find_package(holoscan-messages)`.

## Requirements

- holoscan-stereo-disparity and holoscan-messages, installed (see
  [its README](../../modules/stereo-disparity/README.md)), and what they need: Holoscan SDK 5.0 with Core, Sensor I/O and Visualization (Testing for the
  tests), CUDA 13, VPI 4.1, CMake 4.0 or later and a C++20 compiler.
- A display for the HoloViz window; without one, run with `--display false`.
- For the recording mode, a recording (below).

## Recordings

A recording is a folder with the pair's camera_info files `left.yaml` and `right.yaml`, `index.csv`,
8-bit PGM left and right images and, optionally, 16-bit (big-endian) PGM depth images with
`depth.yaml`. A small reader in [`src/recording.cpp`](src/recording.cpp) reads the index; the
camera_info files are read by holoscan-messages (`read_stereo_camera_info_yaml()`), and the operator
needs them to describe a rectified pair. The depth is compared when `depth.yaml` describes the left
camera (the same size and intrinsics) and `--camera-depth-scale` gives its unit, in meters.

`--recording DIR` plays a folder; `--clip NAME` plays `$HOLOSCAN_TEST_DATA/stereo-camera/NAME/`.
It plays one frame every `--period-ms`, and each frame's capture time is when it is played, not the
`timestamp_ns` of `index.csv`, so the latency it reports starts at playback.
The local test recordings are four clips of 60 frames of a depth camera's rectified infrared pair
looking at an office ceiling, at 1280x720 and 848x480, with its infrared projector on and off, and
its depth in units of 0.000452 m.

## Build and run

From the repository root, after installing the module into `install/`:

```bash
cmake -S applications/stereo_disparity -B build/stereo_disparity -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PWD/install"
cmake --build build/stereo_disparity
build/stereo_disparity/stereo_disparity --backend all,auto --count 300               # synthetic mode
build/stereo_disparity/stereo_disparity --source recording --recording "$STEREO_DISPARITY_RECORDING" \
  --backend all --loop true --count 600                                               # recording mode
build/stereo_disparity/stereo_disparity --backend all,auto --count 30 --period-ms 10 --display false  # ci mode
```

`--help` lists the options. `--backend` takes a comma-separated list of `cuda`, `ofa`, `pva`,
`auto` and `all` (every backend this platform has), each at most once; `auto` is reported as
`auto:<backend that ran>`. The matcher's parameters are `--max-disparity`, `--confidence-threshold`,
`--uniqueness`, `--p1`, `--p2`, `--window`, `--num-passes`, `--diagonals` and `--downscale`; unset,
each backend takes its defaults. `--confidence true` adds VPI's
confidence map and reports its mean (`cuda` and `pva`; `ofa` has none). `--camera-depth-scale X`
compares a recording's depth, with the depth's camera info. Synthetic frames are RGB8
in device memory unless `--color false` (MONO8) or `--device-input false` (host memory, which the
operator copies to the device and says so once). `-DSTEREO_DISPARITY_ENABLE_VISUALIZATION=OFF`
builds it without HoloViz.

The modes in [metadata.json](metadata.json) are the three commands above (the recording mode plays
the folder named by `STEREO_DISPARITY_RECORDING`); `ci` is the finite, headless check that CI runs. When `HOLOSCAN_CI_PLATFORM` is set, as on the CI runners, a run with
`auto` also checks its choice: `ofa` for `jetson`, `cuda` for `x86_64`.

On a Jetson, the demo links the Vulkan loader first: once VPI has loaded Tegra's EGL driver,
HoloViz can no longer create its Vulkan instance. In a headless container there, leave `DISPLAY`
unset, or VPI fails with `EGL_NOT_INITIALIZED`.

## Output and checks

Each run prints a line per backend: frames, the operator's compute time per frame (mean and
maximum), latency from capture to the reporter (median and 95th percentile), the share of pixels
with a disparity, and the checks. Then `PASS`, `FAIL`, or `STOPPED` when an interrupt or a closed
window ends it early. It exits 0 on `PASS`, 2 for invalid options, else 1. The source holds its
first frame until the camera info has come, so every matcher sees every frame, and a run passes
only with all of them.

**Synthetic**, over the pixels both cameras see: the share with a disparity (`valid`, at least
0.85), the share of those within 1 px of the truth (`within_1px`, at least 0.85), and their median
error (`median_px`, at most 0.5 px); the mean error (`mae_px`), and the median relative depth error
and the share within 5 % of the true depth, as for a recording, are reported too. Every line also
reports `whole_px`, the share of disparities without a fractional part.

**Recording**, with `--camera-depth-scale`, over the pixels where the camera has depth: the share
where the computed depth has one too (`coverage`), the median and 90th percentile of
`|computed - camera| / camera`, and the share within 5 %; and every depth camera info must be the
recording's left camera at the depth's size. They are checked only when the camera itself has depth
for at least `--min-camera-coverage` (0.2) of the pixels:

| Check | Pass | Measured, projector on |
| --- | --- | --- |
| `coverage` | at least 0.85 | 0.98 to 1.00 |
| `median_rel_err` | at most 0.03 | 0.0105 to 0.0115; 0.0155 to 0.0185 for `ofa` and `pva` on Thor |
| `within_5pct` | at least 0.75 | 0.84 to 0.89; 0.80 to 0.84 for `ofa` and `pva` on Thor |

The camera and VPI match the same infrared pair, both with semi-global matching, so they should
agree to a fraction of a pixel. At the ceiling's distance, about 2.5 m, the disparity is about 13 px
at 720p, so 3 % is 0.4 px; the 90th percentile, 5 to 8 %, is the edges and the textureless ducts,
where the two matchers fill in differently. On Jetson Thor, VPI 4.1.4's `ofa` and `pva` give
whole-pixel disparities (`whole_px=1.000`), up to half a pixel off, which costs them about half a
percent of median error here. With the projector off, the infrared images are nearly
black: the camera finds depth on only about 2 % of the pixels, so there is too little to compare,
and the run must complete and report its valid share, but its accuracy is only reported.

```text
stereo_disparity: recording .../720p_emitter_on 1280x720, 60 frames, rectified pair, fx=336.99 px, baseline=94.84 mm, with the camera's depth
stereo_disparity: backend=cuda frames=60/60 compute_ms(mean/max)=... valid=0.943 camera_depth: camera_valid=0.693 coverage=0.993 median_rel_err=0.0115 p90_rel_err=0.0555 within_5pct=0.885 (checked)
stereo_disparity: verdict=PASS
```

## Accuracy

Against the camera's depth, all 60 frames of each clip, at each backend's defaults (SDK 5.0.0,
VPI 4.1.4): `cuda` without the diagonal paths, `ofa` and `pva` with 1 pass on Jetson Orin and 3 on
Jetson Thor (see the module's [Defaults](../../modules/stereo-disparity/README.md#defaults)).
`cuda` gave the same numbers on x86_64 and Jetson Orin. The Jetson Thor rows were measured earlier
with the same settings.

| Clip | Backend | Valid | Camera has depth | Coverage | Median error | 90th percentile | Within 5 % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 720p, projector on | `cuda` | 0.943 | 0.693 | 0.993 | 1.15 % | 5.55 % | 0.885 |
| 720p, projector on | `ofa`, Orin | 0.995 | 0.693 | 1.000 | 1.15 % | 8.35 % | 0.842 |
| 720p, projector on | `ofa`, Thor | 0.995 | 0.693 | 1.000 | 1.55 % | 7.35 % | 0.836 |
| 720p, projector on | `pva`, Orin | 0.996 | 0.693 | 1.000 | 1.15 % | 7.65 % | 0.853 |
| 720p, projector on | `pva`, Thor | 0.997 | 0.693 | 1.000 | 1.55 % | 7.25 % | 0.841 |
| 480p, projector on | `cuda` | 0.937 | 0.806 | 0.978 | 1.15 % | 5.35 % | 0.890 |
| 480p, projector on | `ofa`, Orin | 0.995 | 0.806 | 0.999 | 1.15 % | 8.35 % | 0.843 |
| 480p, projector on | `ofa`, Thor | 0.996 | 0.806 | 1.000 | 1.85 % | 8.25 % | 0.805 |
| 480p, projector on | `pva`, Orin | 0.996 | 0.806 | 0.999 | 1.05 % | 7.65 % | 0.855 |
| 480p, projector on | `pva`, Thor | 0.996 | 0.806 | 1.000 | 1.75 % | 7.85 % | 0.814 |
| 720p, projector off | `cuda` | 0.873 | 0.018 | 0.958 | 6.65 % | 64 % | 0.439 |
| 720p, projector off | `ofa`, Orin | 0.980 | 0.018 | 0.991 | 7.95 % | 85 % | 0.405 |
| 720p, projector off | `ofa`, Thor | 0.987 | 0.018 | 0.996 | 8.05 % | 75 % | 0.400 |
| 720p, projector off | `pva`, Orin | 0.986 | 0.018 | 0.993 | 7.75 % | 84 % | 0.411 |
| 720p, projector off | `pva`, Thor | 0.996 | 0.018 | 0.999 | 8.05 % | 80 % | 0.394 |
| 480p, projector off | `cuda` | 0.906 | 0.021 | 0.972 | 6.65 % | 49 % | 0.428 |
| 480p, projector off | `ofa`, Orin | 0.971 | 0.021 | 0.994 | 7.55 % | 61 % | 0.402 |
| 480p, projector off | `ofa`, Thor | 0.977 | 0.021 | 0.992 | 7.55 % | 58 % | 0.394 |
| 480p, projector off | `pva`, Orin | 0.977 | 0.021 | 0.996 | 7.35 % | 59 % | 0.409 |
| 480p, projector off | `pva`, Thor | 0.990 | 0.021 | 0.997 | 7.45 % | 61 % | 0.399 |

On the synthetic scene (30 frames, 1280x720, MONO8), at the defaults:

| Backend | Valid | Within 1 px | Median error | Mean error | Median depth error | Depth within 5 % |
| --- | --- | --- | --- | --- | --- | --- |
| `cuda`, x86_64, Orin and Thor | 0.994 | 0.994 | 0.14 px | 0.23 px | 1.05 % | 0.993 |
| `ofa`, Orin | 1.000 | 0.993 | 0.08 px | 0.27 px | 0.35 % | 0.993 |
| `ofa`, Thor | 1.000 | 0.999 | 0.20 px | 0.22 px | 1.55 % | 0.999 |
| `pva`, Orin | 1.000 | 0.994 | 0.08 px | 0.26 px | 0.35 % | 0.993 |
| `pva`, Thor | 1.000 | 0.922 | 0.20 px | 3.2 px | 1.55 % | 0.922 |

On Thor, `pva` leaves 8 % of the pixels more than a pixel off, some of them far off.

## Performance

Mean time in `StereoDisparityOp::compute()` per frame at each backend's defaults, one backend at a
time, from receiving the pair to having the outputs queued (SDK 5.0.0, CUDA 13.2, VPI 4.1.4; driver
595.99.02 on x86_64, JetPack 7.2 on both Jetsons, with driver 595.78 on Jetson Orin), as
[`tools/sweep.py`](tools/sweep.py) measures it. The recordings are MONO8 in host memory, so they
include the copy to the GPU; the synthetic frames are MONO8 in device memory, so they
include waiting for the source's rendering when it is still running on the GPU.

```bash
build/stereo_disparity/stereo_disparity --source recording --recording "$STEREO_DISPARITY_RECORDING" \
  --backend ofa --period-ms 100 --display false
```

| Input | `cuda`, RTX 5090 | `cuda`, Orin | `ofa`, Orin | `pva`, Orin | `cuda`, Thor | `ofa`, Thor | `pva`, Thor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Recording, 1280x720 | 1.43 ms | 20.3 ms | 29.3 ms | 34.4 ms | | | |
| Recording, 848x480 | 1.09 ms | 20.4 ms | 20.4 ms | 20.5 ms | | | |
| Synthetic, 1280x720 | 1.36 ms | 20.8 ms | 23.0 ms | 35.0 ms | 11.8 ms | 18.9 ms | 15.1 ms |

The Jetson Thor is a CI runner shared with other work, which has no recordings, so its times are
upper bounds. On a Jetson, `ofa` and `pva` leave the GPU free for other work.

## Tests

With `-DBUILD_TESTING=ON`, `ctest` runs the demo: the synthetic scene through every backend and
`auto` (and, on a CI runner, `auto`'s choice), from RGB8 device and MONO8 host images, with the
confidence map; a recording of the synthetic scene, written by a test fixture with its camera_info
files and its true depth, through the reader, the recording source, `CameraInfoSourceOp`, the
camera-depth check and the depth's camera info, with and without depth, and looped; the local
recordings through every backend and `auto`, found by the endings of their folder names in
`$HOLOSCAN_TEST_DATA/stereo-camera/` and skipped (reported as skipped by CTest) when it is not set,
as in CI, where every test makes its own data; the exit statuses for invalid options; that the
display graph compiles; and a run without managed signal handling. `tests/host/` tests the
recording reader without the SDK. [`tools/sweep.py`](tools/sweep.py) runs the demo once per matcher
setting and prints a CSV of times and accuracy, which the tables above come from.

## Licenses

Apache-2.0. The module's README lists the licenses of what it uses.
