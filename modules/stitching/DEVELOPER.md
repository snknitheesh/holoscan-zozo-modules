# holoscan-stitching developer guide

## Layout

```text
stitching/
├── CMakeLists.txt               Standalone project against an installed Holoscan SDK 5.0
├── metadata.json                Module metadata (urn:holoscan:module:v2)
├── cmake/                       Package config template; the CLI template's Debian helper
├── include/holoscan/stitching/  Installed public API
├── src/                         Operator, Stitcher, CUDA kernels, CPU reference; private headers
├── pkg/holoscan-stitching/      Debian packaging, entered when PKG_holoscan_stitching is ON
└── tests/
    ├── host/                    SDK-free tests, also built on their own by the hosted CI job
    └── installed_consumer/      Uses the installed package
```

| Header | What it has | Needs |
| --- | --- | --- |
| `camera_rig.hpp` | `Camera`, `CameraRig`, distortion, `mount()`, rig files | Nothing |
| `projection.hpp` | `OutputGeometry`, the projections, pixel and rig point mappings | Nothing |
| `remap.hpp` | The CPU reference: tables, blend, overlap statistics, gains, `resolution_map()` | Nothing |
| `stitcher.hpp` | `Stitcher`, the GPU engine | CUDA runtime |
| `stitch_op.hpp` | `StitchOp`, `StitchConfig` | SDK |
| `synthetic_world.hpp` | The tests' world, rigs, renders and comparison | CUDA runtime |

## Build and package boundary

| | |
| --- | --- |
| Build contexts | Top-level CMake project, or included by another project (`add_subdirectory()`, `FetchContent`, the repository root); it is not built inside the SDK tree |
| Options | Top level: `BUILD_TESTING` (`OFF`), `BUILD_SHARED_LIBS` (`ON`). Included: `HOLOSCAN_STITCHING_BUILD_TESTING` (`OFF`), always shared, the includer's `BUILD_TESTING` and `BUILD_SHARED_LIBS` left alone. Both: `PKG_holoscan_stitching` (`OFF`) |
| Public targets | `holoscan::stitching` (`libholoscan_stitching.so.1`) |
| Include root | `include/`, installed to `<prefix>/include` |
| Package | `<prefix>/lib/cmake/holoscan-stitching/holoscan-stitching-config.cmake`, `SameMajorVersion` |
| Consumer form | `find_package(holoscan-stitching)`; the install prefix on `CMAKE_PREFIX_PATH`, or `/usr` from the Debian package |
| Debian package | `holoscan-stitching`, all install rules, prefix `/usr`; depends on `holoscan (>= 5.0.0)` |
| External dependencies | Holoscan SDK 5.0 (`core`, `sensor_io`; `testing` for the tests), the CUDA runtime |
| Platforms | x86_64; aarch64 (Jetson Thor with JetPack 7, Jetson Orin with JetPack 7.2) |
| Test labels | `stitching` and `gpu`, or `stitching` and `host` |

## Conventions

- Code lives in `holoscan::stitching`; what the kernels and the CPU share lives in
  `holoscan::stitching::detail`, in `src/stitch_math.hpp`: the projection of an output pixel into a
  camera, bilinear sampling, the blend of one pixel, the luminance samples and the gain solve, all
  `__host__ __device__`. Change the arithmetic there, once, never in a kernel or in `remap.cpp`.
- The GPU blend equals the CPU reference byte for byte because nothing contracts floating-point
  arithmetic: the `.cu` files build with `--fmad=false`, the library's and the tests' host code with
  `-ffp-contract=off`. The tables differ from the CPU's only through the math library's `sin`,
  `cos` and `atan2`, in double: below 1e-3 px.
- The per-camera loops in `stitch_math.hpp` are unrolled over `kMaxCameras` with constant indices,
  so the kernels read `BlendParams` from the parameter space instead of copying it to local memory.
- The overlap statistics are integer sums, so the gains (and so the output) do not depend on the
  order of the GPU's atomic additions.
- An operator has a port only when configured: `StitchOp` declares `image_<i>` for the rig's
  cameras only; an unconnected input is never satisfied.
- `synthetic_world` renders in float: it is the test's camera, not the code under test.
- A new projection or blend option needs a case in `stitching_stitcher` (against the truth and the
  CPU reference) and in `stitching_reference`; a new port or behaviour of `StitchOp`, a case in
  `stitching_operator`; a new calibration field, a case in `stitching_calibration`.

## Benchmark

`stitching_benchmark`, built with the tests, times the tables and each frame for four 1280x720
fisheye cameras into a 1024x1024 bird's-eye view and 4096x1024 panoramas, and prints the table of
the README's Performance section:

```bash
build/stitching/tests/stitching_benchmark
```
