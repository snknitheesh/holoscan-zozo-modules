# holoscan-rectify developer guide

## Layout

```text
rectify/
├── CMakeLists.txt              Standalone project against an installed Holoscan SDK 5.0
├── metadata.json               Module metadata (urn:holoscan:module:v2)
├── cmake/                      Package config template; the CLI template's Debian helper
├── include/holoscan/rectify/   Installed public API
├── schemas/                    The geometry message's FlatBuffers schema
├── src/                        Geometry, loaders, backends, operators; private headers
├── pkg/holoscan-rectify/       Debian packaging, entered when PKG_holoscan_rectify is ON
└── tests/
    ├── host/                   SDK-free tests, also built on their own by the hosted CI job
    └── installed_consumer/     Uses the installed package
```

| Part | Files | Needs |
| --- | --- | --- |
| Calibration and loaders | `camera_model.hpp`, `camera_model.cpp`, `calibration_io.cpp`, `json.cpp` | Nothing |
| Geometry and maps | `rectification.hpp`, `rectification.cpp` | Nothing |
| Backend rules | `backend.hpp`, `backend_rules.cpp` | Nothing |
| Synthetic pair | `synthetic_stereo.hpp`, `synthetic_stereo.cpp` | Nothing |
| Remap backends | `remapper.hpp`, `remapper.cpp`, `cuda_remap.cu`, `vpi_remap.cpp` | CUDA; VPI 4 for `vpi_remap.cpp` |
| Operators | `undistort_op`, `stereo_rectify_op`, `op_support` | Holoscan SDK |

The SDK-free parts are compiled again, from source, by `tests/host/`, so the hosted CI job tests
them without the SDK or a GPU.

## Build and package boundary

| | |
| --- | --- |
| Build contexts | Top-level CMake project, or included by another project (`add_subdirectory()`, `FetchContent`, the repository root) |
| Options | Top level: `BUILD_TESTING` (`OFF`), `BUILD_SHARED_LIBS` (`ON`). Included: `HOLOSCAN_RECTIFY_BUILD_TESTING` (`OFF`), always shared. Both: `HOLOSCAN_RECTIFY_WITH_VPI` (`ON`), `PKG_holoscan_rectify` (`OFF`) |
| Public targets | `holoscan::rectify` (`libholoscan_rectify.so.1`) |
| Include root | `include/` and the generated `stereo_geometry` headers, installed to `<prefix>/include` |
| Package | `<prefix>/lib/cmake/holoscan-rectify/holoscan-rectify-config.cmake`, `SameMajorVersion`; sets `holoscan-rectify_WITH_VPI` |
| Debian package | `holoscan-rectify`, prefix `/usr`; depends on `holoscan (>= 5.0.0)` and, with VPI, `libnvvpi4` |
| External dependencies | Holoscan SDK 5.0 (`core`, `sensor_io`; `testing` for the tests), the CUDA runtime, optionally VPI 4 |
| Test labels | `rectify` and `gpu`, or `rectify` and `host` |

## Conventions

- Code lives in `holoscan::rectify`. Pixel centers are at integer coordinates everywhere (OpenCV,
  ROS, VPI). Matrices are row-major `std::array<double, 9>`.
- The geometry is computed once, at construction, in double precision; only the remap runs per
  frame. A `RemapModel` evaluates the map at any output position, so the dense table (cuda) and
  VPI's sparse warp grid (VIC, PVA) come from the same function.
- Every backend gives zero for a source outside `[0, size - 1]`, as VPI's `VPI_BORDER_ZERO` does,
  and the maps move a source within half a pixel outside onto the edge. Keep the CUDA kernel, the
  CPU reference in `tests/reference_remap.hpp` and this rule together.
- The VIC and the PVA cannot use CUDA memory in place: VPI copies a wrapped buffer for them and
  refreshes the copy only when the buffer is wrapped again. Never cache wrappers on that path.
- The VIC corrupts the process's heap on an odd image size; `backend_limitation()` must refuse
  every size VPI cannot take before VPI sees it. New constraints go there, with a case in
  `tests/host/backend_rules_test.cpp`.
- An operator reads device input after `synchronize_for_consumer_stream()` on its own stream; the
  runtime does not order it there for the operator.
- An operator has an optional port only when configured: an emit into an unconnected output fails.
- The operators' handles in a graph are not the instances the runtime runs, so a value that
  matters after the run (the backend auto chose) travels in a message, not an accessor.
