# holoscan-codec developer guide

## Layout

```text
codec/
├── CMakeLists.txt              Standalone project against an installed Holoscan SDK 5.0
├── metadata.json               Module metadata (urn:holoscan:module:v2)
├── cmake/                      Package config template; the CLI template's Debian helper
├── include/holoscan/codec/     Installed public API
├── schemas/                    The EncodedVideoFrame FlatBuffers schema
├── src/                        Operators, VideoEncoder/VideoDecoder, backends, CUDA kernels
├── pkg/holoscan-codec/         Debian packaging, entered when PKG_holoscan_codec is ON
└── tests/
    ├── host/                   SDK-free tests, also a project of their own
    └── installed_consumer/     Uses the installed package
```

## How it fits together

```text
VideoEncoderOp ─► VideoEncoder ─► color_convert.cu (RGB/NV12 → NV12, even size)
                                └► EncoderBackend: nvcodec_encoder.cpp | v4l2_encoder.cpp
VideoDecoderOp ─► VideoDecoder ─► DecoderBackend: nvcodec_decoder.cpp | v4l2_decoder.cpp
                                └► color_convert.cu (NV12 → RGB, cropped)
```

- `VideoEncoder` owns the stream logic: it picks the backend (`probe_*` results are cached per
  direction and codec), rebuilds the session on a size change, places keyframes (`index % gop`),
  and checks each access unit with `bitstream.hpp`, repeating the parameter sets on a keyframe
  that lacks them. Backends only encode one NV12 frame.
- `VideoDecoder` skips access units until the first keyframe, tags each one with an id that maps
  back to its pts, and copies each decoded frame into a buffer of its own, valid until the next
  call; backends hand frames to a sink while they are mapped.
- NVENC needs the frame's chroma right after its luma at the same pitch: `VideoEncoder`'s NV12
  buffer is one allocation. NVENC's IDRs are forced per frame (`NV_ENC_PIC_FLAG_FORCEIDR`) with an
  infinite GOP; the V4L2 encoder's IDR interval is the GOP, which matches because every session
  starts at frame 0.
- The V4L2 devices: NVIDIA's libv4l2 maps a buffer through its exported dma-buf (`VIDIOC_EXPBUF`),
  not through the device; its file descriptors do not work with `poll()`, so the queues are polled
  in short sleeps; the encoder derives the frame rate from the buffers' timestamps; controls that
  take a structure (initial QP, reference frames) are passed by pointer. The decoder must take
  whole frames (`DISABLE_COMPLETE_FRAME_INPUT` 0), or it waits for the next access unit to find
  where one ends, and sets up its frame buffers only after its resolution-change event.
- The V4L2 decoder's frames are block-linear NvBufSurfaces. A VPI wrapper made once per buffer
  keeps seeing the buffer's first contents, so each frame gets a new wrapper. VPI is loaded with
  `RTLD_LOCAL`: in the global namespace its libraries break the V4L2 encoder's VIC setup in the
  same process.

## Conventions

- Code lives in `holoscan::codec`; `holoscan::codec::detail` is private to `src/`. Inside it,
  `schema::` means this module's FlatBuffers namespace; write the SDK's as `holoscan::schema::`.
- Public headers carry only SDK, CUDA runtime and standard types. The Video Codec SDK, V4L2 and
  VPI types stay in `src/`.
- No library is linked but the CUDA runtime and driver: the codec libraries, libv4l2 and VPI are
  loaded with `dlopen`, so the module builds and runs without them and reports why a backend is
  unavailable.
- Logging goes through `HOLOSCAN_LOG_*`. Nothing is logged per frame: the backend once, a
  host/device copy once, totals at stop.
- Options other than the standard ones are prefixed `HOLOSCAN_CODEC_`; the tests build only with
  `HOLOSCAN_CODEC_BUILD_TESTING` inside another project.
- A new backend or setting needs a case in `codec_round_trip`; a new port or behavior of an
  operator, a case in `codec_operators`; a change to the Annex-B reader, a case in `codec_host`.
