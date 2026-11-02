# Codec

Hardware video encoding and decoding in a Holoscan graph, with
[holoscan-codec](../../modules/codec/). A synthetic camera sends frames of the module's test
video (a moving gradient, color bars and scrolling blocky text); `VideoEncoderOp` encodes them to
H.264 or H.265 on NVENC and `VideoDecoderOp` decodes them on NVDEC. A reporter pairs each decoded
frame with its access unit and checks it against the frame that was sent: its PSNR, its order, its
timestamp and header, and the bitrate. HoloViz shows the original and the decoded frame side by
side, with the bitrate and PSNR, in a window or rendered offscreen.

```text
source ─► VideoEncoderOp ─► VideoDecoderOp ─► report ─► HoloViz text
  │              └──────────────────────────────►┘          │
  └─────────────────────────────► HoloViz (original | decoded) ─► render check (offscreen)
```

## Build and run

Install the module, then build the demo against it, from the repository root:

```bash
cmake -S modules/codec -B build/codec -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/codec && cmake --install build/codec --prefix "$PWD/install"
cmake -S applications/codec -B build/codec_demo -G Ninja -DCMAKE_PREFIX_PATH="$PWD/install"
cmake --build build/codec_demo
build/codec_demo/codec_demo                     # a window, until it is closed
build/codec_demo/codec_demo --codec h265 --rate-control cqp --qp 30
build/codec_demo/codec_demo --display offscreen --count 120 --render-file render.ppm
build/codec_demo/codec_demo --display none --count 60 --min-psnr 34 --rate-control cqp --qp 25
```

In a container, `NVIDIA_DRIVER_CAPABILITIES` must include `video` (the repository's CI image sets
`all`) and, for a window, `graphics` and `display`, with `DISPLAY` and the X socket passed in.

| Option | Default | |
| --- | --- | --- |
| `--codec h264\|h265` | `h264` | |
| `--backend auto\|nvcodec\|v4l2` | `auto` | The first that works here: NVENC/NVDEC on x86_64 and Thor, V4L2 on Orin |
| `--rate-control cbr\|vbr\|cqp` | `vbr` | |
| `--bitrate BPS` | 8000000 | For `cbr` and `vbr` |
| `--qp N` | 23 | For `cqp`, 0 (best) to 51 |
| `--gop N` | 30 | Frames from one keyframe to the next |
| `--preset fast\|medium\|quality` | `fast` | |
| `--low-latency true\|false` | `true` | |
| `--output FILE` | none | Also write the raw stream (`.h264`, `.h265`), which `ffplay` plays |
| `--width N --height N` | 1280x720 | Any size from 160 to 4096; odd sizes work |
| `--count N` | until closed with a window, 60 without | Frames, then stop |
| `--period-ms N` | 33 | The camera's frame period |
| `--display auto\|window\|offscreen\|none` | `auto` | A window when there is a display, else none |
| `--render-file FILE` | none | Offscreen: the last rendered frame, as PPM |
| `--min-psnr DB` | 0 (off) | Fail on a decoded frame below this |
| `--timeout-seconds N` | 60 | Fail when no frame comes for this long |
| `--benchmark N` | off | Instead of the graph, time N frames of encoding and decoding at 1080p and 2160p for each codec, and print frames per second |

A finite run (`--count`) is lossless and checks every frame in order, and passes only when all of
them came, every check held, and offscreen, when at least 80 % of the rendered frames show both
halves right; it prints a summary and a verdict:

```text
codec_demo: codec=h265 backend=v4l2 frames=60/60 keyframes=2 psnr_db: mean=38.23 min=37.14 bitrate_mbps=1.93
codec_demo: verdict=PASS
```

The bitrate is at the encoder's planned 30 frames per second, whatever `--period-ms`. A run with a
window is real time: frames the encoder is too busy for are skipped, never access units.

## Modes

| Mode | |
| --- | --- |
| `window` | Real time, H.264 at 8 Mbit/s, in a window until it is closed |
| `offscreen` | 120 frames of H.265 rendered offscreen and checked, the last one written to `codec_render.ppm` |
| `ci` | 60 frames of H.265 at QP 25, every frame decoded, in order, with its timestamp, at 34 dB or more; then `--benchmark 120` |
| `benchmark` | `--benchmark 300`: the throughput table of the [module's README](../../modules/codec/README.md#quality-and-speed) |

## Tests

`ctest` in the build folder runs the demo with H.265 at a constant QP, with H.264 at a constant
bitrate on an odd frame size, and offscreen with every rendered frame checked, and checks that bad
options are refused. Each needs a GPU with the video engines, or Jetson Orin; without one the demo
exits 77 and the test is skipped.
