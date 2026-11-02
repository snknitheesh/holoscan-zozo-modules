# holoscan-image-quality developer guide

## Layout

```text
image-quality/
├── CMakeLists.txt                   Standalone project against an installed Holoscan SDK 5.0
├── metadata.json                    Module metadata (urn:holoscan:module:v2)
├── cmake/                           Package config template; the CLI template's Debian helper
├── include/holoscan/image_quality/  Installed public API
├── src/                             The operator, the kernels, the evaluator, the CPU reference
├── pkg/holoscan-image-quality/      Debian packaging, entered when PKG_holoscan_image_quality is ON
└── tests/
    ├── host/                        SDK-free tests of the CPU reference and the evaluator
    └── installed_consumer/          Uses the installed package
```

The folder is `image-quality`, as the package is `holoscan-image-quality`; C++ names cannot hold a
hyphen, so the headers, namespace and target say `image_quality`.

## How it fits together

```text
image ──> GpuStats (analyzer.cu) ─┐
                                  ├─> FrameStats ──> QualityEvaluator (quality.cpp) ──> QualityReport
image ──> ReferenceStats (CPU) ───┘     integer sums,     metrics, flags, tiles,
                                        histograms        temporal state
```

- `GpuStats` gathers a frame's `FrameStats` with four kernels: luma (with the color sums, the hash,
  the difference to the previous frame and the luma histogram), stencils (Laplacian and Sobel sums,
  the gradient histogram), the flat-pixel threshold (one block, a CUB scan) and the high-pass
  histogram of the flat pixels. Each tile is split into row bands, one block each, enough to fill the
  GPU; each band's sums go to its own slot, reduced with `cub::BlockReduce`, and the host adds the
  bands. Everything is integer, so neither the order nor the split changes a bit.
- `ReferenceStats` gathers the same statistics on the CPU, pixel by pixel; the GPU test checks
  `GpuStats` against it for equality.
- `QualityEvaluator` is the only place metrics, thresholds and the temporal state (previous hash,
  frozen run, each tile's low-texture window) live. It is plain C++, so the host test covers it.
- `ImageQualityOp` checks the image, runs `ImageQualityAnalyzer` (both of the above) on the
  runtime's CUDA stream, or its own when there is none, and emits.

## Build and package boundary

| | |
| --- | --- |
| Build contexts | Top-level CMake project, or included by another project (`add_subdirectory()`, `FetchContent`, the repository root); it is not built inside the SDK tree |
| Options | Top level: `BUILD_TESTING` (`OFF`), `BUILD_SHARED_LIBS` (`ON`). Included: `HOLOSCAN_IMAGE_QUALITY_BUILD_TESTING` (`OFF`), always shared, the includer's `BUILD_TESTING` and `BUILD_SHARED_LIBS` left alone. Both: `PKG_holoscan_image_quality` (`OFF`) |
| Public targets | `holoscan::image_quality` (`libholoscan_image_quality.so.1`) |
| Include root | `include/`, installed to `<prefix>/include` |
| Package | `<prefix>/lib/cmake/holoscan-image-quality/holoscan-image-quality-config.cmake`, `SameMajorVersion` |
| Consumer form | `find_package(holoscan-image-quality)`; the install prefix on `CMAKE_PREFIX_PATH`, or `/usr` from the Debian package |
| Debian package | `holoscan-image-quality`, all install rules, prefix `/usr`; depends on `holoscan (>= 5.0.0)` |
| External dependencies | Holoscan SDK 5.0 (`core`, `sensor_io`; `testing` for the tests), the CUDA runtime, CUB from the CUDA Toolkit |
| Platforms | x86_64; aarch64 (Jetson Thor with JetPack 7, Jetson Orin with JetPack 7.2) |
| Test labels | `image_quality`, and `gpu` or `host` |

## Conventions

- Code lives in `holoscan::image_quality`. Public headers carry only SDK, CUDA runtime and standard
  types; `quality.hpp`, `reference.hpp` and `synthetic.hpp` need neither SDK nor CUDA.
- The kernels call the host's `constexpr` helpers (`luma()`, `tile_begin()`, `pixel_hash()`), with
  `--expt-relaxed-constexpr`, so both sides compute them from the same code. Keep every statistic an
  integer: a floating-point sum on the GPU would break the equality with the reference.
- Device code needs nothing newer than CUDA 13.0 (CUB from its own CCCL), so the module also builds
  with JetPack 7's toolkit.
- An operator has an optional port only when configured: an emit into an unconnected output fails.
- A new metric or flag needs a case in `tests/accuracy_cases.hpp`, which runs on the CPU reference
  (host test) and on the GPU from host and device memory; a new statistic, a case in
  `matches_reference` of `analyzer_test.cpp`.
