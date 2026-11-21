# holoscan-foundation-stereo developer guide

## Layout

```text
foundation-stereo/
├── CMakeLists.txt                     Standalone project against an installed Holoscan SDK 5.0
├── metadata.json                      Module metadata (urn:holoscan:module:v2)
├── cmake/                             Package config template; the CLI template's Debian helper
├── include/holoscan/foundation_stereo/  Installed public API
├── src/                               The operator, its kernels and camera info rules, the stub model, the synthetic scene
├── tools/prepare_model.py             Fetches or exports the model; tools/README.md says how
├── pkg/holoscan-foundation-stereo/    Debian packaging, entered when PKG_holoscan_foundation_stereo is ON
└── tests/
    ├── host/                          SDK-free tests, also a project of their own
    └── installed_consumer/            Uses the installed package
```

## Build and package boundary

| | |
| --- | --- |
| Build contexts | Top-level CMake project, or included by another project (`add_subdirectory()`, `FetchContent`, the repository root) |
| Options | Top level: `BUILD_TESTING` (`OFF`), `BUILD_SHARED_LIBS` (`ON`). Included: `HOLOSCAN_FOUNDATION_STEREO_BUILD_TESTING` (`OFF`), always shared, the includer's `BUILD_TESTING` and `BUILD_SHARED_LIBS` left alone. Both: `PKG_holoscan_foundation_stereo` (`OFF`), `HOLOSCAN_FOUNDATION_STEREO_MODEL` (tests) |
| Public target | `holoscan::foundation_stereo` (`libholoscan_foundation_stereo.so.1`) |
| Include root | `include/`, installed to `<prefix>/include` |
| Package | `<prefix>/lib/cmake/holoscan-foundation-stereo/holoscan-foundation-stereo-config.cmake`, `SameMajorVersion`; sets `holoscan-foundation-stereo_PREPARE_MODEL` |
| Installed tool | `<prefix>/share/holoscan-foundation-stereo/prepare_model.py`; its README as `share/doc/holoscan-foundation-stereo/prepare_model.md` |
| Debian package | `holoscan-foundation-stereo`, prefix `/usr`; depends on `holoscan (>= 5.0.0)` and `holoscan-messages (>= 1.0.0)` |
| External dependencies | Holoscan SDK 5.0 (`core`, `sensor_io`, `inference`; `testing` for the tests), holoscan-messages 1.0 (public: its camera info types are in the operator's ports), the CUDA runtime, TensorRT 10 through HoloInfer (private; the package asks for the SDK's `inference` component only for a static library) |
| Test labels | `foundation_stereo` on all; `gpu` for the GPU tests; `host` for the SDK-free tests |

## Conventions

- Code lives in `holoscan::foundation_stereo`; the tests' helpers in `holoscan::foundation_stereo::test`,
  the CPU references in `holoscan::foundation_stereo::reference`.
- Public headers carry only SDK, holoscan-messages, CUDA runtime and standard types; HoloInfer stays
  private to `src/foundation_stereo_op.cpp`. `model_input.hpp`, `depth.hpp`, `camera_info.hpp`,
  `stub_model.hpp` and `reference.hpp` need no SDK or CUDA, so `tests/host/` builds their sources,
  with holoscan-messages' SDK-free ones, without them.
- Calibration is holoscan-messages' `StereoCameraInfo`, taken with `camera_info_contract()` and
  `LatestCameraInfo`, and the depth's camera is its `CameraInfo`: the module has no calibration
  type or message of its own. The port names and conventions it shares with the stereo disparity
  module's `StereoDisparityOp` (`left`, `right`, `camera_info`, `disparity`, `depth`,
  `depth_camera_info`) follow that operator's; keep them in step.
- The kernels and the synthetic scene's renderer are compiled without FMA contraction
  (`-fmad=false`), and the host code that recomputes them without `-ffp-contract`, so the CPU
  references and the host-rendered scene match the GPU to float rounding.
- The repository holds no weights and no upstream code, and must not. Tests use the stub model; the
  real-model test and the demo's real-model and recording tests skip (exit 77) without the model or
  `HOLOSCAN_TEST_DATA`, so CI, which has neither, runs synthetic data only.
- A new setting or behavior of the operator needs a case in `foundation_stereo_operator`; a change to
  a kernel, a case in `foundation_stereo_kernels`.

## How the operator runs a frame

Take the newest camera info (skip the frame before the first), check it against the images; wait on
the host for the frame before last, so at most two frames are on the GPU; then, all on the runtime's
CUDA stream for the activation: wait for the previous frame's end event (the model's buffers and the
staging buffer are shared by every frame, and the runtime may give each activation another stream);
copy host images to the staging buffer, made once at the largest size; record a start event; resize
and normalize each eye into HoloInfer's input buffers; `execute_inference()`; resize and rescale the
model's output into the output loans; emit the disparity, the depth's camera info and the depth;
record the end event. The start and end events of the last four frames form a ring; a frame's GPU
time is read from them once its end event has completed. An imported pinned input whose memory the
runtime cannot hold for the GPU is kept until an event recorded after its copy has completed, which
`compute()` waits for once the frame's GPU work is queued: it cannot be kept longer, since the
runtime waits for a frame's inputs to be released before it stops.

`start()` measures the device's free memory before and after loading the model for
`FoundationStereoStats::model_bytes`; it is only meaningful while nothing else on the device
allocates. Before HoloInfer loads an `.onnx` model, `start()` and `prepare_engine()` compare its
content's digest with `<model>.digest` and delete the engines cached for other content.
