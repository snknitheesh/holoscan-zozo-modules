# holoscan-codec

A Holoscan SDK 5 Module that encodes images to H.264 or H.265 and decodes them back, on the GPU's
hardware video engines (NVENC and NVDEC), for recording or streaming camera frames from inside a
graph. `VideoEncoderOp` takes images and emits one Annex-B access unit per image, and can write the
raw stream to a file; `VideoDecoderOp` takes access units and emits images in device memory. The
[Codec demo](../../applications/codec/) runs both end to end.

| | |
| --- | --- |
| CMake package | `find_package(holoscan-codec)` |
| Targets | `holoscan::codec` |
| Headers | `<holoscan/codec/...>` |
| C++ namespace | `holoscan::codec` |
| Debian package | `holoscan-codec` |
| Platforms | x86_64; aarch64 (Jetson Thor and Orin, JetPack 7.2) |

## Operators

### `VideoEncoderOp`

| Port | Type | |
| --- | --- | --- |
| `image` (input) | `holoscan::schema::ImageT` | RGB8, BGR8, RGBA8, BGRA8, or NV12 with its two plane layouts; in CUDA device memory, or host memory at the cost of a copy (logged once); at least 160 pixels on each side |
| `bitstream` (output) | `holoscan::codec::schema::EncodedVideoFrameT` | One access unit per image |

`VideoEncoderConfig` holds the `EncoderSettings` and the largest image size, which sizes the
output (`max_width` x `max_height` x 1.5 bytes per access unit), an optional `output_path` for the
raw stream, and `allow_resolution_change`:

| Setting | Default | |
| --- | --- | --- |
| `codec` | `kH264` | `kH264` or `kH265` (Main, 8-bit 4:2:0) |
| `backend` | `kAuto` | See [Backends](#backends) |
| `rate_control` | `kVbr` | `kCbr`, `kVbr` (peak `max_bitrate`, 0 for twice `bitrate`), or `kConstQp` |
| `bitrate` | 8 000 000 | Bits per second, for `kCbr` and `kVbr` |
| `qp` | 23 | The quantizer for `kConstQp`, 0 (best) to 51: the "CRF-like" quality setting, a fixed quality whose bitrate follows the content |
| `gop` | 30 | Frames from one keyframe to the next: frame *n* of a stream is an IDR keyframe exactly when *n* % `gop` == 0 |
| `preset` | `kFast` | `kFast`, `kMedium`, `kQuality`: NVENC's P1, P4, P7; the V4L2 encoder's UltraFast, Medium, Slow |
| `low_latency` | `true` | NVENC's ultra-low-latency tuning, a one-frame VBV with a bitrate, and one reference frame, so a low-latency decoder shows each frame at once. Off: NVENC's high-quality tuning |
| `frame_rate` | 30 | The rate the rate control plans for |

Every stream has I and P frames only (no B-frames), each frame comes out as soon as it is encoded,
and every keyframe starts with its parameter sets (VPS for H.265, SPS, PPS), so a decoder can join
the stream at any keyframe. RGB is converted to NV12 on the GPU with the BT.709 matrix in limited
range, which NVENC also writes into the stream's VUI.

Each `EncodedVideoFrame` ([schema](schemas/encoded_video_frame.fbs)) carries the image's header,
unchanged, the codec, the image's width and height, the `pts` (the header's
`capture_timestamp_ns`, else the sample's capture time, else the frame's index at `frame_rate`),
whether it is a keyframe, its index in the stream, and the access unit as a uint8 tensor in host
memory, start codes included. The sample keeps the image's capture time and frame id.

An odd side is encoded one pixel larger, the last row or column repeated, as 4:2:0 needs even
sides; the message keeps the image's size and `VideoDecoderOp` crops back to it. A change of image
size starts a new stream at the new size: the next access unit is a keyframe with new parameter
sets and the frame index starts again at 0. With `allow_resolution_change` off, the frame fails
instead.

### `VideoDecoderOp`

| Port | Type | |
| --- | --- | --- |
| `bitstream` (input) | `EncodedVideoFrameT` | One access unit per sample, as `VideoEncoderOp` emits them; in host memory, or device memory at the cost of a copy (logged once) |
| `image` (output) | `holoscan::schema::ImageT` | One image per decoded frame, in CUDA device memory |

`VideoDecoderConfig` selects the backend, the `output_format` (RGB8, BGR8, RGBA8, BGRA8, or NV12 as
one byte tensor with its plane layouts and `REC709` color space), the largest image, and
`low_latency`. The decoder is built for the first access unit's codec, and again when the codec or
the frame size changes. Access units before the first keyframe cannot be decoded and give nothing,
nor does an empty access unit; both are counted and logged at stop. Each image has the access
unit's width and height, its header, and its `pts` as `header.capture_timestamp_ns`, and the
sample keeps the access unit's capture time and frame id.

With `low_latency` (the default), each image comes from the `compute()` of its own access unit,
which needs a stream with one reference frame and no reordering, as `VideoEncoderOp` makes in its
low-latency mode; turn it off for other streams, whose images then come some access units later.
Jetson Orin's decoder gives the first images of a stream one or two access units late, then one per
access unit; the operator emits up to four images in one `compute()` to catch up.

The operators log their backend once, for example:

```text
VideoEncoderOp: h264 on the nvcodec backend (NVENC on NVIDIA GeForce RTX 5090), which auto chose as the first backend that works here; vbr 8000000 bit/s, gop 30, preset fast, low latency
VideoDecoderOp: h264 on the v4l2 backend (NVDEC through /dev/v4l2-nvdec, VIC through VPI), which auto chose as the first backend that works here
```

## Backends

| Backend | Platforms | Encoding | Decoding |
| --- | --- | --- | --- |
| `kNvcodec` | x86_64 with an NVIDIA GPU; Jetson Thor (JetPack 7.2) | NVENC through the Video Codec SDK (`libnvidia-encode`), reading NV12 straight from device memory | NVDEC through the Video Codec SDK (`libnvcuvid`); each frame mapped in device memory and copied out |
| `kV4l2` | Jetson Orin (JetPack 7.2) | NVENC through `/dev/v4l2-nvenc` and NVIDIA's libv4l2; each frame copied from device memory into the device's input buffer (pinned, so a DMA copy) | NVDEC through `/dev/v4l2-nvdec`; the block-linear frames laid out pitch-linear in device memory by the VIC, through VPI 4 |
| `kAuto` | Any | The first of the two that works here, probed once | The same |

Jetson Thor also has `/dev/v4l2-nvenc` and `/dev/v4l2-nvdec`, but there NVIDIA's V4L2 plugins are a
layer over the Video Codec SDK with other buffer semantics, which the `kV4l2` backend does not
drive: where `kNvcodec` works, `kV4l2` reports itself unavailable and Thor uses `kNvcodec`.

The module links none of these libraries: it loads them at run time, so it builds and runs
anywhere, and a backend whose libraries or devices are missing is reported as unavailable with the
reason, for example
`nvcodec: libnvidia-encode.so.1: libnvcuvid.so.1: cannot open shared object file` on Jetson Orin,
whose GPU driver has no `libnvcuvid`. `probe_encoder_backends()` and `probe_decoder_backends()`
list each backend's status and reason; an operator whose backend is unavailable fails its first
frame with that reason.

- **In a container**, the NVIDIA Container Toolkit mounts `libnvidia-encode` and `libnvcuvid` only
  when `NVIDIA_DRIVER_CAPABILITIES` includes `video` (or is `all`, as in the repository's CI
  image), and the Jetson devices with `--runtime nvidia` (the CI runs Jetson containers with
  `--privileged` as well).
- **Video Codec SDK headers**: the interface headers (MIT) of SDK 12.1, from FFmpeg's
  [nv-codec-headers](https://github.com/FFmpeg/nv-codec-headers) n12.1.14.1, downloaded at
  configure time and checked against their SHA-256; `HOLOSCAN_CODEC_NV_CODEC_HEADERS_DIR` points
  at a local copy instead. The driver must support NVENC API 12.1 (R530 or newer).
- **VPI**: the `kV4l2` decoder loads VPI 4 privately (`RTLD_LOCAL`): loaded into the process's
  global namespace, its libraries make the V4L2 encoder in the same process fail to set up its
  VIC.

## Quality and speed

PSNR of the module's [test video](include/holoscan/codec/test_video.hpp) (a moving gradient,
color bars, and scrolling blocky text, 1280x720, 30 frames) after a round trip, as `codec_test`
measures it; the test's threshold per setting is the first column:

| Setting | Threshold, mean / worst frame | x86_64 (RTX 5090), H.264 | H.265 | Jetson Orin, H.264 | H.265 |
| --- | --- | --- | --- | --- | --- |
| `kConstQp` 20 | 38 / 36 dB | 39.9 dB, 1.95 Mbit/s | 40.9 dB, 2.47 Mbit/s | 40.1 dB, 1.95 Mbit/s | 40.8 dB, 2.43 Mbit/s |
| `kConstQp` 30 | 32 / 30 dB | 34.1 dB, 1.07 Mbit/s | 35.2 dB, 1.50 Mbit/s | 34.4 dB, 1.10 Mbit/s | 35.3 dB, 1.44 Mbit/s |
| `kConstQp` 40 | 26 / 24 dB | 28.4 dB, 0.66 Mbit/s | 29.0 dB, 0.80 Mbit/s | 28.6 dB, 0.66 Mbit/s | 29.2 dB, 0.79 Mbit/s |
| `kCbr` 8 Mbit/s | 36 / 32 dB | 40.5 dB, 5.2 Mbit/s | 39.9 dB, 4.4 Mbit/s | 41.5 dB, 3.5 Mbit/s | 41.3 dB, 4.2 Mbit/s |
| `kVbr` 2 Mbit/s | 28 / 19 dB | 30.5 dB, 1.8 Mbit/s | 30.0 dB, 1.6 Mbit/s | 37.4 dB, 1.7 Mbit/s | 32.1 dB, 1.2 Mbit/s |

Jetson Thor (`kNvcodec`, in its CI lane) measures within 1 dB of the RTX 5090 at every
setting. With a bitrate, the worst frame is taken after the first five, as the stream's first keyframe
must fit one frame's worth of bits; easy content takes less than the target. The RGB to NV12
conversion alone, at QP 0, gives about 51 dB.

Throughput of `VideoEncoder` and `VideoDecoder` (the demo's `--benchmark` mode), in frames per second of the
whole call (color conversion, the hardware, and the copies in and out), one stream at a time, low
latency, 8 Mbit/s at 1080p and 20 Mbit/s at 2160p:

| | x86_64 (RTX 5090) encode | decode | Jetson Thor encode | decode | Jetson Orin encode | decode |
| --- | --- | --- | --- | --- | --- | --- |
| H.264 1920x1080 | 700 | 2200 | 172 | 419 | 48 to 162 | 28 to 63 |
| H.264 3840x2160 | 203 | 676 | 79 | 273 | 30 to 39 | 16 to 34 |
| H.265 1920x1080 | 895 | 2120 | 190 | 418 | 50 to 152 | 43 to 81 |
| H.265 3840x2160 | 467 | 629 | 124 | 274 | 30 to 37 | 22 to 31 |

The Orin figures (AGX Orin 64 GB, MAXN) vary from run to run with the CPU and engine clocks,
which JetPack scales on demand; the ranges are those of six runs, one of them in CI. Thor's are
from its CI lane. Its V4L2 path costs a copy into
the encoder's buffers on the way in, and a VIC conversion and a copy on the way out, with a poll of
the device queues for each frame.

On the RTX 5090, encoding takes 1 to 3 % of the GPU's SMs (the color conversion) while NVENC runs
at 32 % (1080p H.264) to 63 % (2160p H.265) of one engine's capacity; decoding takes 1 to 4 % of
the SMs, NVDEC about 48 %. The demo's `ci` mode prints these figures in every CI lane.

## Use it in your graph

```cmake
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core sensor_io)
find_package(holoscan-codec 1.0 REQUIRED CONFIG)
target_link_libraries(my_app PRIVATE holoscan::codec)
```

```cpp
#include <holoscan/codec/video_decoder_op.hpp>
#include <holoscan/codec/video_encoder_op.hpp>

namespace hc = holoscan::codec;

auto encoder = graph.op<hc::VideoEncoderOp>(
    "encoder", hc::VideoEncoderConfig{.settings = {.codec = hc::Codec::kH265, .bitrate = 6'000'000},
                                      .output_path = "camera.h265"});
auto decoder = graph.op<hc::VideoDecoderOp>("decoder", hc::VideoDecoderConfig{});
graph.add_flow(camera->image, encoder->image);
// Every access unit is needed to decode the next: never drop them.
graph.add_flow(encoder->bitstream, decoder->bitstream,
               {.queue_policy = holoscan::QueuePolicy::kBlocking, .queue_depth = 8U});
graph.add_flow(decoder->image, display->image);
// The decoder's images are in device memory: bind its output to a GPU when compiling.
compile_options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
    .operator_path = "decoder", .output_port = "image", .device = holoscan::DeviceId{0}});
```

The file is a raw Annex-B stream that `ffplay camera.h265` and most players open. Without a graph,
`VideoEncoder`, `VideoDecoder` and `convert_from_nv12()` in `<holoscan/codec/video_codec.hpp>` do
the same on images in device memory; `<holoscan/codec/bitstream.hpp>` reads NAL units, keyframes
and parameter sets from an Annex-B stream; and `<holoscan/codec/test_video.hpp>` renders the test
video and measures PSNR. `holoscan-codec_WITH_V4L2` tells whether the installed module has the
`kV4l2` backend.

## Requirements

- Holoscan SDK 5.0 (Core and Sensor I/O; Testing for the tests), CUDA 13, CMake 4.0 or later and a
  C++20 compiler.
- At run time, an NVIDIA driver with `libnvidia-encode` and `libnvcuvid` (R530 or newer), or
  Jetson Orin's V4L2 codec devices with VPI 4.
- At build time, the Video Codec SDK interface headers (downloaded, see above) and, for the `kV4l2`
  backend, VPI 4's headers (`vpi4-dev`, in the repository's CI image).

## Build, test and install

From the repository root, in the repository's CI image or with the requirements above:

```bash
cmake -S modules/codec -B build/codec -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build/codec
ctest --test-dir build/codec --output-on-failure
cmake --install build/codec --prefix "$PWD/install"
```

| CMake option | Default | |
| --- | --- | --- |
| `HOLOSCAN_CODEC_WITH_V4L2` | `ON` | Build the `kV4l2` backend when VPI 4's headers are found |
| `HOLOSCAN_CODEC_NV_CODEC_HEADERS_DIR` | empty | A local nv-codec-headers `include/` folder instead of the download |
| `BUILD_TESTING` | `OFF` | Build the tests, when the module is built on its own |
| `HOLOSCAN_CODEC_BUILD_TESTING` | `OFF` | Build the tests inside another project |
| `CMAKE_CUDA_ARCHITECTURES` | `native` (`all-major` for the package) | GPUs to compile for, unless already set |

Another project can also build the module with `add_subdirectory()` or `FetchContent` and link
`holoscan::codec` without `find_package()`; it then builds none of its tests unless
`HOLOSCAN_CODEC_BUILD_TESTING` is `ON`.

The tests generate their own frames. Those that need hardware skip (exit 77) where no backend can
encode and decode, saying why:

| Test | Needs | Checks |
| --- | --- | --- |
| `codec_round_trip` | GPU with NVENC/NVDEC, or Jetson Orin | On every available backend, for H.264 and H.265: PSNR per rate-control setting (the table above); parameter sets on every keyframe and keyframes exactly every GOP; every frame back once, in order, with its pts, each one access unit after the next in low-latency mode; odd sizes cropped back; a size change in one decoder; access units before the first keyframe and empty ones; the smallest size; rejected settings |
| `codec_operators` | The same | Both operators through the SDK's operator test harness: every input encoding in host and device memory, headers, frame ids and pts carried, keyframes, the raw file, RGB8, BGRA8 and NV12 output, an odd size, a size change (a new stream, or refused), empty and undecodable access units, bad configuration |
| `codec_host` | Nothing | The Annex-B reader on hand-made H.264 and H.265 streams, settings and their names, the test video and PSNR; also built on its own from `tests/host/` by the hosted CI job |

[`tests/installed_consumer/`](tests/installed_consumer/) is a separate project that uses the
installed package: it builds both operators and round-trips ten frames. Build it with the install
prefix on `CMAKE_PREFIX_PATH` and run `ctest` there.

## Debian package

Inside the repository's CI image, with the module's folder as the Holoscan CLI project root:

```bash
holoscan --project-root modules/codec package holoscan-codec --language cpp \
  --pkg-generator DEB --local
```

This writes `holoscan-codec_1.0.0_<arch>.deb`, which installs to `/usr` and depends on
`holoscan (>= 5.0.0)`; the codec libraries come with the NVIDIA driver and VPI with JetPack.

## Licenses

Apache-2.0. The Video Codec SDK interface headers the library is compiled against are MIT; the
few NVIDIA V4L2 extension ids it restates are BSD-3-Clause; both notices are in [NOTICE](NOTICE),
which is installed with the module. The libraries it loads at run time are NVIDIA's, under
NVIDIA's terms.
