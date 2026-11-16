# Recorder

The demo of the [holoscan-recorder](../../modules/recorder/) module. It records synthetic streams
with `RecorderOp` and replays them with `ReplayerOp`, checking that every sample comes back byte for
byte; replays a recording, or a stereo camera clip, into a HoloViz window; and measures what the
disk takes from a live camera. It uses the installed module the way any other project would:
`find_package(holoscan-recorder)`.

## Requirements

- holoscan-recorder, installed (see [its README](../../modules/recorder/README.md)), and what it
  needs: Holoscan SDK 5.0 with Core, Sensor I/O and Visualization, CUDA 13, CMake 4.0 or later and
  a C++20 compiler.
- A display for the HoloViz window of the replay mode; without one, run it with `--display false`.
- For the replay of real data: the stereo camera clips in `$HOLOSCAN_TEST_DATA/stereo-camera/`.

## Build and run

From the repository root, after installing the module into `install/`:

```bash
cmake -S applications/recorder -B build/recorder_demo -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PWD/install"
cmake --build build/recorder_demo
build/recorder_demo/recorder --mode ci --recording /tmp/ci_recording
build/recorder_demo/recorder --mode replay --stream rgb --loop true \
  --recording "$HOLOSCAN_TEST_DATA/stereo-camera/d585_720p_emitter_on"
build/recorder_demo/recorder --mode bench --width 1280 --height 720 --encoding rgb8 --fps 60
```

`--help` lists the options. `-DRECORDER_ENABLE_VISUALIZATION=OFF` builds it without HoloViz.

The three modes, which [metadata.json](metadata.json) also names:

- **`ci`**, the finite headless check that CI runs. A source sends 30 frames (`--frames`) of five
  streams at 320x240 (`--width`, `--height`): RGB8 and MONO16 images from the GPU, MONO8 and NV12
  images from the host, and a float32 tensor, each frame a known byte pattern with a known capture
  time and frame id. `RecorderOp` records them through lossless edges. `ReplayerOp` then plays the
  recording as fast as possible, once into host memory and once into device memory, into a sink
  that checks every sample: its bytes, shape, element type, capture time, frame id and image
  descriptor. It passes only if every sample was recorded and comes back exactly.
- **`replay`**, one image stream of a recording (`--stream`; `rgb`, else the first) in real time
  (`--pacing realtime`) or as fast as possible (`--pacing fast`), optionally in a loop, into a
  HoloViz window and a counter, through `kDropOldest` edges. MONO8, MONO16, RGB8 and RGBA8 streams
  can be shown. Without a display (`--display false`), it counts the frames and stops after
  `--count` of them, or at the end. A stereo camera clip plays the same way: its streams are
  `left`, `right`, `rgb` and `depth`.
- **`bench`**, a synthetic camera of `--width` x `--height` RGB8 or MONO8 (`--encoding`) frames at
  `--fps`, in device or host memory (`--memory`), recorded through a `kDropOldest` edge, as a live
  camera would be. It reports the rate offered, the frames recorded and dropped, and the rate
  written including start-up and the final drain, and deletes the recording unless `--keep true`.

Every run that is not a replay with a display is stopped, and fails, after `--timeout-s` (120 s).

## Output

`ci` prints what it recorded and what each replay found, then `PASS` or `FAIL`; it exits 0 on
`PASS`, 2 for invalid options, else 1:

```text
recorder: recorded=150/150 dropped=0 bytes=17280720 to /tmp/ci_recording
recorder: replayed to host: samples=150/150 mismatches=0
recorder: replayed to device: samples=150/150 mismatches=0
recorder: verdict=PASS
```

`bench` prints one line:

```text
recorder: bench 1280x720 rgb8 device at 60.0 fps: frames=300/300 offered=158.2 MiB/s received=300 recorded=300 dropped=0 edge_dropped=0 written=152.5 MiB/s (with startup and drain) max_queued=1
```

## Tests

`ctest` in the build folder runs the `ci` mode, a 1280x720 one, real-time and looped replays of
what it recorded, a short benchmark, the HoloViz graph's compilation, and the rejection of invalid
options. Two more replay the `d585_480p_emitter_on` clip, fast and in real time, when
`HOLOSCAN_TEST_DATA` has it, and are skipped otherwise.
