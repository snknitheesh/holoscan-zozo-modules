# holoscan-denoise developer guide

## Layout

```text
denoise/
├── CMakeLists.txt             Standalone project against an installed Holoscan SDK 5.0
├── metadata.json              Module metadata (urn:holoscan:module:v2)
├── cmake/                     Package config template; the CLI template's Debian helper
├── include/holoscan/denoise/  Installed public API
├── src/                       The operators, the backends, the CUDA kernels; private headers
├── pkg/holoscan-denoise/      Debian packaging, entered when PKG_holoscan_denoise is ON
└── tests/
    ├── host/                  SDK-free tests of the settings and the references
    └── installed_consumer/    Uses the installed package
```

| Source | |
| --- | --- |
| `include/holoscan/denoise/denoise_params.hpp`, `src/denoise_params.cpp` | Denoiser settings, backends, the auto choice, the strength's mapping (`tuning::`), image views. Plain C++ |
| `include/holoscan/denoise/fusion_params.hpp`, `src/fusion_params.cpp` | Fusion and tone map settings, HDR views, pyramid depth, the double-precision references. Plain C++ |
| `include/holoscan/denoise/denoiser.hpp`, `src/denoiser.cpp`, `src/denoiser_impl.hpp` | `Denoiser`: checks views, resolves the backend, builds it |
| `src/cuda_denoiser.cu` | The cuda backend: motion, blend, bilateral and non-local means kernels |
| `src/noise_estimate.{hpp,cu}` | The noise estimate and the changed-pixel count both backends use |
| `src/vpi_denoiser.cpp`, `src/nv12_convert.{hpp,cu}` | The vpi backend: VPI TNR, and the NV12 conversions around it |
| `include/holoscan/denoise/exposure_fusion.hpp`, `src/exposure_fusion.cu` | `ExposureFusion` and `ToneMapper` |
| `include/holoscan/denoise/*_op.hpp`, `src/*_op.cpp`, `src/image_ports.{hpp,cpp}` | The operators and what they share at their ports: reading images, staging host input, the outputs |
| `include/holoscan/denoise/synthetic_scene.hpp`, `src/synthetic_scene.cu` | The synthetic scenes, their truth and the noise |

## Build and package boundary

| | |
| --- | --- |
| Build contexts | Top-level CMake project, or included by another project (`add_subdirectory()`, `FetchContent`, the repository root); it is not built inside the SDK tree |
| Options | Top level: `BUILD_TESTING` (`OFF`), `BUILD_SHARED_LIBS` (`ON`). Included: `HOLOSCAN_DENOISE_BUILD_TESTING` (`OFF`), always shared, the includer's `BUILD_TESTING` and `BUILD_SHARED_LIBS` left alone. Both: `HOLOSCAN_DENOISE_WITH_VPI` (`ON`), `PKG_holoscan_denoise` (`OFF`) |
| Public targets | `holoscan::denoise` (`libholoscan_denoise.so.1`) |
| Include root | `include/`, installed to `<prefix>/include` |
| Package | `<prefix>/lib/cmake/holoscan-denoise/holoscan-denoise-config.cmake`, `SameMajorVersion`; sets `holoscan-denoise_WITH_VPI` |
| Consumer form | `find_package(holoscan-denoise)`; the install prefix on `CMAKE_PREFIX_PATH`, or `/usr` from the Debian package |
| Debian package | `holoscan-denoise`, all install rules, prefix `/usr`; depends on `holoscan (>= 5.0.0)`, and `libnvvpi4 (>= 4.0)` with the vpi backend |
| External dependencies | Holoscan SDK 5.0 (`core`, `sensor_io`; `testing` for the tests), the CUDA runtime; optionally VPI 4 (linked privately) |
| Platforms | x86_64; aarch64 (Jetson Thor and Jetson Orin with JetPack 7.2) |
| Test labels | `denoise` and `gpu`; `denoise` and `host` for the SDK-free ones |

## Conventions

- Code lives in `holoscan::denoise`; what the backends and operators share stays in
  `holoscan::denoise::detail`.
- The cuda backend is deterministic and must stay so: every sum is an integer or reduced in a fixed
  order (no float atomics), and every kernel computes each pixel from inputs alone. A change that
  breaks this fails `denoise_denoiser`'s determinism case.
- Strength 0 must leave frames unchanged with every method (`tuning::history_frames()` gives 1 and
  the spatial kernels copy when their range is 0).
- The temporal history is kept before the spatial filter, so that spatial smoothing never feeds
  back. The spatial filter's noise level is the frame's times `tuning::residual_variance()` of the
  pixel's age: a new setting of the temporal filter must keep that estimate honest.
- The tuning constants (`tuning::` in `denoise_params.hpp`) are shared by the kernels and the tests;
  the gains in the README come from `denoise_denoiser`'s output, which prints them.
- The vpi backend waits on the host: VPI cannot wait for a CUDA stream it does not own. It keeps the
  previous output locked for CUDA until the next frame has waited for the stream, as the conversion
  may still read it. VPI's TNR needs NV12 of even size (and on the VIC at least 64x16), which the
  backend pads to and converts.
- Host memory reaches the GPU on the operator's stream; pinned host memory is held until the copy
  is done (`set_deallocation_stream`), and the first copy is logged once.
- An operator has an optional port only when configured: an emit into an unconnected output fails,
  and an unconnected input never fires.
- A new method, format or backend needs cases in `denoise_denoiser` (gain, ghosting, scene cut,
  determinism) and in `denoise_operators`; a new fusion or tone map setting, a case in
  `denoise_fusion` against the reference.
