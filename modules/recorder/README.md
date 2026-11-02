# holoscan-recorder

A Holoscan SDK 5 Module that records the streams of a graph to disk and plays them back:

- `RecorderOp` records named streams, each a stream of images (`schema::ImageT`, any pixel format)
  or of bare tensors (any element type and shape), in host or device memory, every sample with its
  capture time, frame id and image descriptor. It never holds the graph back: a writer thread of its
  own writes to disk, and when the disk falls behind, samples are dropped and counted, not waited
  for.
- `ReplayerOp` plays a recording back with the recorded capture times, byte for byte, into host or
  device memory: as fast as the consumers take it and lossless, for tests; in real time by the
  recorded timestamps; or in a loop; over a frame range and a subset of the streams. It also plays
  the stereo camera recordings used for real-data tests.

Other modules use it for tests on real data: record a camera once, then replay it in CI. The
[Recorder demo](../../applications/recorder/) records, replays and checks synthetic streams, replays a
real recording into a HoloViz window, and measures what the disk takes.

| | |
| --- | --- |
| CMake package | `find_package(holoscan-recorder)` |
| Targets | `holoscan::recorder` |
| Headers | `<holoscan/recorder/...>` |
| C++ namespace | `holoscan::recorder` |
| Debian package | `holoscan-recorder` |
| Platforms | x86_64; aarch64 (Jetson Thor with JetPack 7, Jetson Orin with JetPack 7.2) |

## Operators

### `RecorderOp`

`RecorderOp(RecorderConfig)` has one input per stream of `RecorderConfig::streams`, named after it:
`recorder->image("left")`, an `Input<schema::ImageT>`, for an image stream, and
`recorder->tensor("imu")`, an `Input<holoscan::Tensor>`, for a tensor stream. It runs when any input
has samples (`OnAnyInput`) and takes all of them (`DrainAll`).

For each sample, `compute()` copies the data into pinned host memory, from the GPU asynchronously on
the operator's CUDA stream, and queues it for its writer thread; it never touches the disk. The
queue is bounded by `queue_samples` (32) and `queue_bytes` (512 MiB of pinned memory). When it is
full, the sample is dropped and counted (`RecorderStats::dropped`, and per stream in the recording),
and a warning says so at most once a second. Packed tensors and tensors whose rows are padded are
recorded packed; other layouts fail the run.

Connect live sources with `QueuePolicy::kDropOldest` edges, as for any consumer that must not hold a
camera back. A lossless `kBlocking` edge records every sample the queue has room for, and is what a
test uses.

| `RecorderConfig` | Default | |
| --- | --- | --- |
| `directory` | | The recording's folder: new, empty, or with `overwrite` a recording |
| `streams` | | `{name, StreamKind::kImage or kTensor}`, one input each; names are 1 to 64 of `[A-Za-z0-9_.-]`, not starting with `.` |
| `overwrite` | `false` | Replace a recording in `directory`. A folder holding any other file is never written into |
| `calibration` | | Calibration blobs by name, each a JSON value, stored in `recording.json` |
| `metadata` | | Free-form text by name, stored in `recording.json` |
| `queue_samples`, `queue_bytes` | 32, 512 MiB | The writer's queue |
| `input_queue_depth` | 4 | Each input's queue, for an edge that does not set one |
| `max_write_bytes_per_second` | 0 (no limit) | Holds the writer to a rate: for a disk shared with other work, or to play a slow disk |
| `stats` | | `RecorderStats`: samples received, recorded, dropped, bytes written, the most queued at once |

The recording is created in `start()` (a folder it may not write into fails the run there) and
finished in `stop()`, which writes what is still queued and marks it complete. A write that fails,
such as on a full disk, fails the run at the next `compute()`.

### `ReplayerOp`

`ReplayerOp(ReplayerConfig)` opens and checks the recording in its constructor, and has one output
per stream played, named after it: `replayer->image("left")` or `replayer->tensor("imu")`. Each
sample goes out with its recorded capture time (on the graph's clock domain), frame id and image
descriptor, and its data byte for byte. The outputs declare each stream's element type and rank when
all of its samples share them, and the largest sample's size as their bound.

Samples go out in the order of their capture times, at most one per output each `period` (1 ms).
A sample a consumer refuses on a lossless edge is sent again on the next tick, so nothing is lost.

| `ReplayerConfig` | Default | |
| --- | --- | --- |
| `directory` | | A recording, or a stereo camera recording |
| `pacing` | `kAsFastAsPossible` | Or `kRealtime`: each sample when its time comes, measured from the first sample and the first `compute()` |
| `loop` | `false` | Start over at the end. Each later pass adds the recording's length plus one typical interval to the capture times, and the largest frame id to the frame ids, so both keep increasing |
| `first_frame`, `end_frame` | all | Play the samples whose sequence number (a stereo camera recording's frame number) is in `[first_frame, end_frame)` |
| `streams` | all | The streams to play, by name |
| `memory` | `kHost` | Or `kDevice`: the data is read into pinned memory and copied to the GPU on the operator's stream. Place the outputs with `replayer->bind_device(options, "replayer")` |
| `period` | 1 ms | How often it runs, 10 us to 1 s: this bounds the rate per output, and the real-time grain |
| `stats` | | `ReplayerStats`: samples emitted, passes completed, whether it has finished, the worst lateness in real time |

When the recording ends, the operator idles; a consumer that counts samples ends the run (or use a
`loop`). `recording()` gives the `RecordingReader`, with the calibration, and `stream_names()` the
outputs.

## Use it in your graph

```cmake
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core sensor_io)
find_package(holoscan-recorder 1.0 REQUIRED CONFIG)
target_link_libraries(my_app PRIVATE holoscan::recorder)
```

Record a camera's left image and its IMU:

```cpp
#include <holoscan/recorder/recorder_op.hpp>

namespace rec = holoscan::recorder;

auto recorder = graph.op<rec::RecorderOp>(
    "recorder", rec::RecorderConfig{
                    .directory = "/data/run_042",
                    .streams = {{"left", rec::StreamKind::kImage}, {"imu", rec::StreamKind::kTensor}},
                    .calibration = {{"left", calibration_json}}});
const holoscan::ConnectionOptions newest{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                         .queue_depth = 1U};
graph.add_flow(camera->left, recorder->image("left"), newest);
graph.add_flow(imu->samples, recorder->tensor("imu"), newest);
```

Replay it into the same consumers, in device memory, for a test:

```cpp
#include <holoscan/recorder/replayer_op.hpp>

auto replayer = graph.op<rec::ReplayerOp>(
    "replayer", rec::ReplayerConfig{.directory = "/data/run_042",
                                    .memory = rec::ReplayMemory::kDevice});
graph.add_flow(replayer->image("left"), detector->image);
holoscan::CompileOptions options;
replayer->bind_device(options, "replayer");
```

`RecordingWriter` and `RecordingReader` (`<holoscan/recorder/format.hpp>`) write and read
recordings without the SDK, for tools and tests.

## Recording format

A recording is a folder:

```text
recording.json   The streams, the calibration blobs and free-form metadata
index.csv        One line per sample, in the order they were written
<stream>.bin     Each stream's samples, back to back, packed
```

`recording.json`:

```json
{
  "format": "holoscan-recording",
  "version": 1,
  "complete": true,
  "streams": [
    {"name": "left", "kind": "image", "file": "left.bin", "records": 60, "bytes": 55296000, "dropped": 0},
    {"name": "imu", "kind": "tensor", "file": "imu.bin", "records": 600, "bytes": 14400, "dropped": 2}
  ],
  "calibration": {"left": {"fx": 615.0, "fy": 615.0, "cx": 319.5, "cy": 239.5}},
  "metadata": {"robot": "unit 7"}
}
```

`complete` is `false` until the recorder finishes. `records`, `bytes` and `dropped` are written when
it does; a reader takes the samples from the index, and only `dropped` from here.

`index.csv` has a header line and one line per sample, CSV as RFC 4180 has it:

| Column | |
| --- | --- |
| `stream` | The stream's name |
| `sequence` | The sample's number on its stream, counting every sample the recorder was given, so a dropped sample leaves a gap |
| `timestamp_ns` | The capture time, in nanoseconds of the clock that stamped it; a sample without one gets the time it was recorded |
| `file`, `offset`, `bytes` | Where the data is: the stream's file, the byte offset in it, and its size |
| `frame_id` | The sample's frame id, or 0 |
| `dtype` | The element type: `uint8`, `int16`, `float32`, `bfloat16`, `complex64`, `bool8`, with `x<lanes>` for vector lanes (`int16x4`) |
| `shape` | The extents, `720x1280x3`, or `scalar` |
| `encoding`, `width`, `height`, `color_space`, `significant_bits`, `roi_x`, `roi_y`, `planes`, `exposure_ns`, `gain` | An image's descriptor (`schema::ImageT`): the encoding and color space by their SDK names (`RGB8`, `BT709`), `planes` as `offset:row_stride:size` separated by `;`, `gain` with every digit a float needs. Empty for a tensor |
| `header_timestamp_ns`, `device_sequence`, `header_frame_id` | The image's header (`schema::HeaderT`), when it has one; `header_frame_id` is always quoted, so that an empty one is told apart from none. The header's `stamp` is not kept |

The data is the tensor's elements, packed, row-major, in the host's byte order. A sample's data is
written before its index line, and each line is flushed, so a recording cut short (a crash, a power
loss) reads up to its last full line. A recording marked complete must be whole.

The reader checks everything before the first sample plays: the header, every field, that sizes
match shapes and element types, that the data is in its file, and that sequence numbers increase; a
recording that fails any of it is refused with a `RecordingError` naming the file, the line and the
problem.

### Stereo camera recordings

`RecordingReader`, and so `ReplayerOp`, also reads the stereo camera recordings that the stereo
camera module writes and CI keeps in `$HOLOSCAN_TEST_DATA/stereo-camera/<clip>/`, read-only: a folder
with `calibration.json`, `index.csv` with the columns `frame,timestamp_ns,device_sequence,left,right`
and optionally `rgb` and `depth`, and one Netpbm file per image. Its streams are `left`, `right`,
`rgb` and `depth`, in the index's order: 8-bit PGM as `MONO8` `[H, W]`, 8-bit PPM as `RGB8`
`[H, W, 3]`, 16-bit PGM as `MONO16` `[H, W]` (big-endian in the file, host order when read).
Each sample's sequence is the `frame`, its frame id `frame + 1`, its capture time `timestamp_ns`,
and its header the stream's name, `timestamp_ns` and `device_sequence`. `calibration.json` is the
calibration blob `stereo_camera`.

It is an import, not the recorder's format: that layout fixes four image streams at one rate and one
file per image, while a recording holds any number of image and tensor streams at their own rates,
each in one file the writer appends to. The recorder never writes it.

## Requirements

- Holoscan SDK 5.0 (Core and Sensor I/O), the CUDA runtime, CMake 4.0 or later and a C++20 compiler.
- A disk fast enough for what is recorded; see [Throughput](#throughput).

Recordings are not compressed. No compression library is in the CI image, and video encoding
belongs to a codec module.

## Build, test and install

From the repository root, in the repository's CI image or with the requirements above:

```bash
cmake -S modules/recorder -B build/recorder -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build/recorder
ctest --test-dir build/recorder --output-on-failure
cmake --install build/recorder --prefix "$PWD/install"
```

| CMake option | Default | |
| --- | --- | --- |
| `BUILD_TESTING` | `OFF` | Build the tests, when the module is the top-level project |
| `HOLOSCAN_RECORDER_BUILD_TESTING` | `OFF` | Build the tests when another project includes the module (`add_subdirectory()`, `FetchContent`); its `BUILD_TESTING` and `include(CTest)` do not |

The tests:

| Test | Needs | Checks |
| --- | --- | --- |
| `recorder_round_trip` | GPU | Every pixel format of `schema::ImageEncoding` (with plane layouts for the planar ones) and tensors of other element types, ranks and lanes, recorded from host and from device memory and replayed into host and device memory: every byte, capture time, frame id and image descriptor back as sent; and 1280x720 RGB8 and MONO16 frames |
| `recorder_recorder_op` | GPU | A disk held to 10 MB/s under a 100 MB/s source: the source is never refused, the drops are counted, gaps in the sequence numbers show them, and every recorded frame is whole; padded rows from host and device memory recorded packed; rejected settings, foreign folders and existing recordings |
| `recorder_replayer_op` | GPU | Real-time pacing in host and device memory (every sample within 10 ms of its time, 20 ms on Jetson), as fast as possible, loops with increasing capture times and frame ids, a frame range of one stream, and rejected settings and recordings |
| `recorder_real_data` | GPU, `HOLOSCAN_TEST_DATA` | Every frame of the four stereo camera clips, replayed into device memory with its capture time; the pixels of three frames per stream against the files. Skipped without the data |
| `recorder_format`, `recorder_malformed`, `recorder_host_real_data` | Nothing (`tests/host/`) | The writer and reader without the SDK: every field back exactly, recordings cut short, folders; 45 malformed recordings of both formats, each refused with its file, line and problem; and every pixel of the four clips. Also run by the hosted `Lint` job |

[`tests/installed_consumer/`](tests/installed_consumer/) is a separate project that uses the
installed package: it records five images and replays them into device memory, checking every
pixel and capture time.

## Throughput

What `RecorderOp` records without dropping a frame from a live camera, measured with the demo's
`bench` mode: synthetic 1280x720 frames from device memory through a `kDropOldest` edge, for 10 s
(at most 3000 frames), with the default queue of 32 samples and 512 MiB, onto the machine's NVMe
SSD (ext4). Recording writes through the page cache, so a run much longer than the kernel's dirty
page budget sees the disk's own sustained rate.

| Machine | RGB8 (2.6 MiB a frame) | MONO8 (0.9 MiB a frame) |
| --- | --- | --- |
| x86_64 workstation, NVMe SSD | 528 MiB/s (200 fps) every time; 633 MiB/s (240 fps) in two runs of three | 879 MiB/s (1000 fps, the source's limit) |
| Jetson AGX Orin, JetPack 7.2, NVMe SSD | 633 MiB/s (240 fps); 1265 MiB/s (480 fps) drops 4 % | 879 MiB/s (1000 fps, the source's limit) |

Above that, frames are dropped and counted, and the source is never held back. Real-time replay of
the tests' 15-sample recording kept every sample within 0.11 ms of its time on both.

To measure a disk of your own: `recorder --mode bench --width 1280 --height 720 --encoding rgb8
--fps 240 --recording /path/on/that/disk`.

## Debian package

Inside the repository's CI image, with the module's folder as the Holoscan CLI project root:

```bash
holoscan --project-root modules/recorder package holoscan-recorder --language cpp \
  --pkg-generator DEB --local
```

This writes `holoscan-recorder_1.0.0_<arch>.deb` to the current directory. It installs to `/usr` and
depends on `holoscan (>= 5.0.0)`, so `apt install ./holoscan-recorder_1.0.0_<arch>.deb` needs
NVIDIA's CUDA repository.

## Licenses

Apache-2.0. See [NOTICE](NOTICE).
