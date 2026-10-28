# Stereo Disparity

The demo of the [holoscan-stereo-disparity](../../modules/stereo-disparity/) module. A stereo
source fans each frame out to a `StereoDisparityOp` per selected backend (`cuda`, `ofa`, `pva`, or
`auto`, which the module picks for the platform), and a reporter per backend checks its result:

- **Synthetic**: textured planes at known depths, rendered by the module's synthetic scene, so
  every disparity is checked against the truth.
- **Recording**: a recorded stereo camera clip, played from its folder, so the computed depth is
  checked against the camera's own depth image.

HoloViz shows the first backend's disparity, colorized from blue (far) to dark red (near), black
where it is invalid; for a color synthetic pair, beside the left image. The demo uses the installed
module the way any other project would: `find_package(holoscan-stereo-disparity)`.

## Requirements

- holoscan-stereo-disparity, installed (see [its README](../../modules/stereo-disparity/README.md)),
  and what it needs: Holoscan SDK 5.0 with Core, Sensor I/O and Visualization (Testing for the
  tests), CUDA 13, VPI 4, CMake 4.0 or later and a C++20 compiler.
- A display for the HoloViz window; without one, run with `--display false`.
- For the recording mode, a recording (below).

## Recordings

A recording is a folder with `calibration.json`, `index.csv`, and 8-bit PGM left and right images
and, optionally, 16-bit (big-endian) PGM depth images in the left eye's frame, as the
stereo-camera test recordings are stored. A small reader in [`src/recording.cpp`](src/recording.cpp)
takes what the demo needs: the eyes' size, the left focal length and principal point, the right
principal point, the baseline (minus the x translation of `right_from_left`), and the depth's scale
when the depth is aligned with the left eye. It refuses a calibration whose `rectified` is not
`true`: the operator needs rectified input, and the eyes of a RealSense are.

`--recording DIR` plays a folder; `--clip NAME` plays `$HOLOSCAN_TEST_DATA/stereo-camera/NAME/`.
The four test clips are 60 frames each of a RealSense D585 looking at an office ceiling, at
1280x720 and 848x480, with its infrared projector on and off.

## Build and run

From the repository root, after installing the module into `install/`:

```bash
cmake -S applications/stereo_disparity -B build/stereo_disparity -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PWD/install"
cmake --build build/stereo_disparity
build/stereo_disparity/stereo_disparity --backend all,auto --count 300               # synthetic mode
build/stereo_disparity/stereo_disparity --source recording --clip d585_720p_emitter_on \
  --backend all --loop true --count 600                                               # recording mode
build/stereo_disparity/stereo_disparity --backend all,auto --count 30 --period-ms 10 --display false  # ci mode
```

`--help` lists the options. `--backend` takes a comma-separated list of `cuda`, `ofa`, `pva`,
`auto` and `all` (every backend this platform has), each at most once; `auto` is reported as
`auto:<backend that ran>`. The matcher's parameters are `--max-disparity`, `--confidence-threshold`,
`--uniqueness`, `--p1`, `--p2`, `--window` and `--num-passes`. `--confidence true` adds VPI's
confidence map and reports its mean (`cuda` and `pva`; `ofa` has none). `--geometry-input true`
sends the recording's geometry with every frame instead of configuring it. Synthetic frames are RGB8
in device memory unless `--color false` (MONO8) or `--device-input false` (host memory, which the
operator copies to the device and says so once). `-DSTEREO_DISPARITY_ENABLE_VISUALIZATION=OFF`
builds it without HoloViz.

The modes in [metadata.json](metadata.json) are the three commands above; `ci` is the finite,
headless check that CI runs. When `HOLOSCAN_CI_PLATFORM` is set, as on the CI runners, a run with
`auto` also checks its choice: `ofa` for `jetson`, `cuda` for `x86_64`.

On a Jetson, the demo links the Vulkan loader first: once VPI has loaded Tegra's EGL driver,
HoloViz can no longer create its Vulkan instance. In a headless container there, leave `DISPLAY`
unset, or VPI fails with `EGL_NOT_INITIALIZED`.

## Output and checks

Each run prints a line per backend: frames, the operator's compute time per frame (mean and
maximum), latency from capture to the reporter (median and 95th percentile), the share of pixels
with a disparity, and the checks. Then `PASS`, `FAIL`, or `STOPPED` when an interrupt or a closed
window ends it early. It exits 0 on `PASS`, 2 for invalid options, else 1.

**Synthetic**, over the pixels both cameras see: the share with a disparity (`valid`, at least
0.85), the share of those within 1 px of the truth (`within_1px`, at least 0.9), and their mean
error (`mae_px`, at most 0.5 px).

**Recording**, over the pixels where the camera has depth: the share where the computed depth has
one too (`coverage`), the median and 90th percentile of `|computed - camera| / camera`, and the
share within 5 %. They are checked only when the camera itself has depth for at least
`--min-camera-coverage` (0.2) of the pixels:

| Check | Pass | Measured, projector on |
| --- | --- | --- |
| `coverage` | at least 0.85 | 0.98 to 1.00 |
| `median_rel_err` | at most 0.03 | 0.0095 to 0.0105 |
| `within_5pct` | at least 0.75 | 0.87 to 0.90 |

The camera and VPI match the same infrared pair, both with semi-global matching, so they should
agree to a fraction of a pixel. At the ceiling's distance, about 2.5 m, the disparity is about 13 px
at 720p, so 3 % is 0.4 px; the 90th percentile, 5 to 7 %, is the edges and the textureless ducts,
where the two matchers fill in differently. With the projector off, the infrared images are nearly
black: the camera finds depth on only about 2 % of the pixels, so there is too little to compare,
and the run must complete and report its valid share, but its accuracy is only reported.

```text
stereo_disparity: recording .../d585_720p_emitter_on (RealSense D585 Prototype 541522320091) 1280x720, 60 frames, fx=336.99 px, baseline=94.84 mm, with the camera's depth
stereo_disparity: backend=cuda frames=60/60 compute_ms(mean/max)=... valid=0.929 camera_depth: camera_valid=0.693 coverage=0.994 median_rel_err=0.0105 p90_rel_err=0.0505 within_5pct=0.897 (checked)
stereo_disparity: verdict=PASS
```

## Accuracy

Against the camera's depth, all 60 frames of each clip, VPI 4.1.4 (`cuda` on x86_64 and Jetson
Orin gave the same numbers; `ofa` and `pva` on Jetson Orin):

| Clip | Backend | Valid | Camera has depth | Coverage | Median error | 90th percentile | Within 5 % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 720p, projector on | `cuda` | 0.929 | 0.693 | 0.994 | 1.05 % | 5.05 % | 0.897 |
| 720p, projector on | `ofa` | 0.995 | 0.693 | 1.000 | 1.05 % | 6.55 % | 0.866 |
| 720p, projector on | `pva` | 0.996 | 0.693 | 1.000 | 1.05 % | 6.05 % | 0.877 |
| 480p, projector on | `cuda` | 0.925 | 0.806 | 0.977 | 1.05 % | 4.95 % | 0.903 |
| 480p, projector on | `ofa` | 0.996 | 0.806 | 1.000 | 1.05 % | 6.55 % | 0.866 |
| 480p, projector on | `pva` | 0.996 | 0.806 | 1.000 | 0.95 % | 5.95 % | 0.879 |
| 720p, projector off | `cuda` | 0.902 | 0.018 | 0.964 | 6.65 % | 64 % | 0.440 |
| 720p, projector off | `ofa` | 0.987 | 0.018 | 0.996 | 7.85 % | 75 % | 0.407 |
| 720p, projector off | `pva` | 0.994 | 0.018 | 0.998 | 7.55 % | 73 % | 0.414 |
| 480p, projector off | `cuda` | 0.932 | 0.021 | 0.982 | 6.75 % | 49 % | 0.428 |
| 480p, projector off | `ofa` | 0.977 | 0.021 | 0.992 | 7.35 % | 57 % | 0.408 |
| 480p, projector off | `pva` | 0.986 | 0.021 | 0.995 | 7.05 % | 55 % | 0.417 |

On the synthetic scene (30 frames, 1280x720, the `ci` mode): `cuda` 0.994 valid, 0.994 within
1 px, 0.23 px mean error; `ofa` 1.000, 0.999, 0.094 px; `pva` 1.000, 0.999, 0.088 px.

## Tests

With `-DBUILD_TESTING=ON`, `ctest` runs the demo: the synthetic scene through every backend and
`auto` (and, on a CI runner, `auto`'s choice), from RGB8 device and MONO8 host images, with the
confidence map; a recording of the synthetic scene, written by a test fixture with its true depth,
through the reader, the recording source, configured and per-frame geometry and the camera-depth
check; the four real clips through every backend and per-frame geometry, skipped (reported as
skipped by CTest) when `HOLOSCAN_TEST_DATA` is not set; the exit statuses for invalid options; that
the display graph compiles; and a run without managed signal handling. `tests/host/` tests the
recording reader without the SDK.

## Licenses

Apache-2.0. The module's README lists the licenses of what it uses.
