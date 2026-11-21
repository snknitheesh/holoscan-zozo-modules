# Foundation Stereo

The demo of the [holoscan-foundation-stereo](../../modules/foundation-stereo/) module. A stereo
source and the pair's calibration feed `FoundationStereoOp`, which runs NVIDIA's
Fast-FoundationStereo through TensorRT, and a reporter checks its result. The calibration is always
a pair of ROS camera_info files, read by [holoscan-messages](../../modules/messages/)'
`CameraInfoSourceOp` and sent to the source, which sends it on with every frame:

- **Synthetic**: textured planes at known depths, rendered by the module's synthetic scene, so every
  disparity is checked against the truth. The scene's calibration is written as camera_info files
  into a temporary folder for the run, and removed after it.
- **Recording**: a rectified stereo recording, played from its folder, as a camera would deliver it,
  with its own camera_info files; with `--camera-depth-scale`, the computed depth and its camera
  info are compared with the camera's own depth image.
- **Stub** (`--model stub`, the `ci` mode): the module's stand-in model instead of the real one, on
  the synthetic scene, each frame checked against the CPU reference of the operator's whole chain.
  It needs no download.

HoloViz shows the left image beside the disparity, colorized from blue (far) to dark red (near),
black where it is invalid. `--display offscreen` renders the same into GPU memory instead, checks
that the left half shows the image and the right half the colorized disparity, and with
`--snapshot FILE` writes the newest frame as a PPM file. The demo uses the installed module the way
any other project would: `find_package(holoscan-foundation-stereo)`.

> **The model is NVIDIA's, under NVIDIA's licenses.** C-Fast-FoundationStereo, which the modes use,
> is governed by the NVIDIA Open Model Agreement; the upstream Fast-FoundationStereo code and research
> checkpoints are licensed for non-commercial research use only. Neither is part of this repository.
> See the [module's README](../../modules/foundation-stereo/README.md) and its NOTICE. The tests and
> the `ci` mode use the stub model.

## Requirements

- holoscan-foundation-stereo, installed (see [its README](../../modules/foundation-stereo/README.md)),
  and what it needs: Holoscan SDK 5.0 with Core, Sensor I/O, Inference and Visualization (Testing
  for the tests), CUDA 13 with TensorRT 10, CMake 4.0 or later and a C++20 compiler.
- A display for the HoloViz window; without one, run with `--display false` or `--display offscreen`.
- For every mode but `ci`, the model; for the recording mode, a recording (below).

## Prepare the model

From the repository root (Python 3.12 and the network; no PyTorch):

```bash
python3 modules/foundation-stereo/tools/prepare_model.py --output data/foundation_stereo
```

This writes `data/foundation_stereo/c_fast_foundationstereo_480x736.onnx`, where the modes and the
tests look for it. The first run builds its TensorRT engine beside it, which takes about 2 minutes on
an RTX 5090 and about 46 minutes on a Jetson Orin; later runs load it in seconds. An interrupt
(Ctrl-C) during that build takes effect only once the build is done. `--model stub` writes the stub
model, and caches its engine, in `$XDG_CACHE_HOME/holoscan-foundation-stereo` (by default
`~/.cache/holoscan-foundation-stereo`), never in the working directory.

## Recordings

A recording is a folder with the pair's ROS camera_info files `left.yaml` and `right.yaml`,
`index.csv`, and 8-bit PGM left and right images and, optionally, `depth.yaml` and 16-bit
(big-endian) PGM depth images, as the local test recordings are stored. A small reader in
[`src/recording.cpp`](src/recording.cpp) reads the camera_info files with holoscan-messages'
reader and the index; images named outside the folder are refused. The operator needs a rectified
pair, and the files must describe one (both cameras rectified, as a ROS stereo calibration writes
them for rectified images); `CameraInfoSourceOp` reads them again when the run starts. The depth
is compared with the camera's only when `depth.yaml` describes the left camera and
`--camera-depth-scale` gives the depth images' meters per unit.

`--recording DIR` plays a folder; `--clip NAME` plays `$HOLOSCAN_TEST_DATA/stereo-camera/NAME/`. The
four local test recordings are 60 frames each of an office ceiling, from a stereo camera whose
infrared pair is rectified, at 1280x720 and 848x480, with its projector on and off; their depth is
0.000452 m per unit. Played with `--realtime true`, as the recording mode does, the edges are
depth-1 drop-oldest, as for a live camera: the model always takes the newest pair, and frames it
cannot keep up with are dropped and counted.

## Build and run

From the repository root, after installing the module into `install/`:

```bash
cmake -S applications/foundation_stereo -B build/foundation_stereo -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PWD/install"
cmake --build build/foundation_stereo
model=data/foundation_stereo/c_fast_foundationstereo_480x736.onnx
build/foundation_stereo/foundation_stereo --model $model --count 300               # synthetic mode
build/foundation_stereo/foundation_stereo --model $model --source recording \
  --recording /path/to/recording --loop true --count 600 --realtime true           # recording mode
build/foundation_stereo/foundation_stereo --model stub --count 30 --width 640 --height 480 \
  --period-ms 50 --display offscreen                                               # ci mode
```

`--help` lists the options. Synthetic frames are RGB8 in device memory unless `--color false`
(MONO8) or `--device-input false` (host memory, which the operator copies to the device and says so
once). `--fp16 false` builds an FP32 engine, and `--cuda-graphs false` launches the model without a
CUDA graph. `-DFOUNDATION_STEREO_ENABLE_VISUALIZATION=OFF` builds it without HoloViz. The modes in
[metadata.json](metadata.json) are the three commands above (the data directory is the repository's
`data`; the recording mode's folder is `FOUNDATION_STEREO_RECORDING`); `ci` is the finite, headless
check that CI runs.

Every lossless run, the default, is one graph on one worker, as the messages module's README asks
of a lossless graph, so the reporter keeps up with every frame; `--compile-only true` prints how
many workers the graph has. Such a run goes at the model's speed, whatever `--period-ms` says: the
source is refused while the model is busy and sends the same frame again, so on a Jetson Orin a
33 ms period gives a frame about every 160 ms. `--realtime true` leaves the graph to the SDK's
automatic partitioning instead, with drop-oldest edges. In real time, an interrupt (Ctrl-C) ends the
run with `verdict=STOPPED`, not as a failure.

## Output and checks

Each run prints one line: the model's input size, the frames, the GPU time of the first frame (which
also captures the CUDA graph) and the mean and maximum of the others (resize, model, resize back),
the latency from capture until the disparity's GPU work is done, without the first two frames, the
drop in the GPU's free memory while the model loaded (approximate: other processes on the GPU
change it), and the share of valid pixels; then, by source:

- synthetic, real model: over the pixels both cameras see, the share with a disparity, the share
  within 1 px of the truth, the median and the mean error (EPE). It passes with at least 0.95, 0.95
  and a median of at most 0.25 px;
- synthetic, stub: the pixels compared with the CPU reference, those off by more than the
  tolerance (1 % of the value with FP16, 0.01 % with FP32), and the largest difference. It passes
  with none off;
- recording, with `--camera-depth-scale`: against the camera's depth, the camera's own coverage, the
  share of its pixels the computed depth covers, the median and 90th percentile relative error, and
  the share within 5 %. Every depth must come with the left camera's camera info. The accuracy is
  checked only where the camera has depth for at least `--min-camera-coverage` (0.2) of the pixels:
  coverage at least 0.85, median at most 3 %, within 5 % at least 0.60. With the projector off, the
  camera has depth on about 2 % of the pixels, and the comparison is only reported.

```text
foundation_stereo: recording .../720p_emitter_on 1280x720, 60 frames, rectified pair, fx=336.99 px, baseline=94.84 mm, with the camera's depth
foundation_stereo: model=736x480 frames=60/60 gpu_ms(first,mean/max)=20.10,10.88/14.87 latency_ms(p50/p95)=12.48/13.66 free_memory_drop_mib=774 valid=0.964 camera_depth: camera_valid=0.693 coverage=0.995 median_rel_err=0.0115 p90_rel_err=0.4775 within_5pct=0.776 (checked)
foundation_stereo: verdict=PASS
```

The module's README has the accuracy, latency and memory measured on every recording, on x86_64 and
the Jetson Orin, and how they compare with the classic matcher.

## Tests

With `-DBUILD_TESTING=ON`, `ctest` runs the demo. With the stub model, needing no download: the
synthetic scene from RGB8 device and MONO8 host images, with FP16 and FP32 engines, without CUDA
graphs, and in real time; the offscreen display, on the synthetic scene and on a recording; a
recording of the synthetic scene, written by a test fixture with its camera_info files and true
depth, through the reader, the recording source, the camera info with every frame, the depth and its
camera info and the camera-depth comparison, losslessly and in real time; that the lossless graphs,
the `ci` mode's and a recording's with the camera's depth, compile to one worker; the exit statuses
for invalid options; that the display graph compiles; and a run without managed signal handling.
With the real model, skipped (exit 77, reported as skipped) unless it is prepared: the synthetic
scene against its truth in color and gray, the synthetic recording against its true depth, and,
when `HOLOSCAN_TEST_DATA` is also set, the four local recordings against the camera's depth,
losslessly and one in real time. CI has neither the model nor the recordings, so it runs synthetic
data only. `tests/host/` tests the recording reader without the SDK.

## Licenses

Apache-2.0. The model is not covered by it; see the module's README and NOTICE.
