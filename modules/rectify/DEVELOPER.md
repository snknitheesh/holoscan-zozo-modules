# holoscan-rectify developer guide

## Layout

```text
rectify/
├── CMakeLists.txt              Standalone project against an installed Holoscan SDK 5.0
├── metadata.json               Module metadata (urn:holoscan:module:v2)
├── cmake/                      Package config template; the CLI template's Debian helper
├── include/holoscan/rectify/   Installed public API
├── src/                        Lens models, geometry, backends, operators; private headers
├── pkg/holoscan-rectify/       Debian packaging, entered when PKG_holoscan_rectify is ON
└── tests/
    ├── host/                   SDK-free tests, also built on their own by the hosted CI job
    └── installed_consumer/     Uses the installed package
```

| Part | Files | Needs |
| --- | --- | --- |
| Lens models | `lens.hpp`, `lens.cpp` | holoscan-messages' SDK-free part |
| Geometry and maps | `rectification.hpp`, `rectification.cpp` | holoscan-messages' SDK-free part |
| Backend rules | `backend.hpp`, `backend_rules.cpp` | Nothing |
| Synthetic pair | `synthetic_stereo.hpp`, `synthetic_stereo.cpp` | holoscan-messages' SDK-free part |
| Remap backends | `remapper.hpp`, `remapper.cpp`, `cuda_remap.cu`, `vpi_remap.cpp` | CUDA; VPI 4 for `vpi_remap.cpp` |
| Operators | `undistort_op`, `stereo_rectify_op`, `op_support` | Holoscan SDK |

The SDK-free parts are compiled again, from source, by `tests/host/`, with the SDK-free sources of
holoscan-messages (`camera_info.cpp`, `camera_info_yaml.cpp`, `synthetic_calibration.cpp`,
`yaml_reader.cpp`), so the hosted CI job tests them without the SDK or a GPU.

## Build and package boundary

| | |
| --- | --- |
| Build contexts | Top-level CMake project, or included by another project (`add_subdirectory()`, `FetchContent`, the repository root) |
| Options | Top level: `BUILD_TESTING` (`OFF`), `BUILD_SHARED_LIBS` (`ON`). Included: `HOLOSCAN_RECTIFY_BUILD_TESTING` (`OFF`), always shared. Both: `HOLOSCAN_RECTIFY_WITH_VPI` (`ON`), `PKG_holoscan_rectify` (`OFF`) |
| Public targets | `holoscan::rectify` (`libholoscan_rectify.so.1`) |
| Include root | `include/`, installed to `<prefix>/include` |
| Package | `<prefix>/lib/cmake/holoscan-rectify/holoscan-rectify-config.cmake`, `SameMajorVersion`; sets `holoscan-rectify_WITH_VPI` |
| Debian package | `holoscan-rectify`, prefix `/usr`; depends on `holoscan (>= 5.0.0)`, `holoscan-messages (>= 1.0.0)` and, with VPI, `libnvvpi4` |
| Module dependencies | `messages` (`module.dependencies`): `holoscan::messages` from the same build when the repository root added it, else `find_package(holoscan-messages 1.0)`; the package config finds it too |
| External dependencies | Holoscan SDK 5.0 (`core`, `sensor_io`; `testing` for the tests), the CUDA runtime, optionally VPI 4 |
| Test labels | `rectify` and `gpu`, or `rectify` and `host` |

## Conventions

- Code lives in `holoscan::rectify`. Pixel centers are at integer coordinates everywhere (OpenCV,
  ROS, VPI). Matrices are row-major `std::array<double, 9>` (`messages::Matrix3`).
- Calibration is holoscan-messages' `CameraInfo` and `StereoCameraInfo`, read only from ROS
  `camera_info` files with its readers; this module adds no type, format or parser of its own.
  The first operator of a chain (here, both operators) reads the files; what it sends downstream is
  camera info on an output, never configuration.
- The geometry is computed once, at construction, in double precision; only the remap runs per
  frame. A `RemapModel` evaluates the map at any output position, so the dense table (cuda) and
  VPI's sparse warp grid (VIC, PVA) come from the same function.
- The cuda backend gives zero for a source outside `[0, size - 1]`, and the maps move a source
  within half a pixel outside onto the edge. (VPI's `VPI_BORDER_ZERO` may blend a source less than
  a pixel outside with zero instead.) Keep the CUDA kernel, the CPU reference in
  `tests/reference_remap.hpp` and this rule together.
- VPI interpolates the source between the warp grid's control points, so they carry the true
  source (`RemapModel::grid_source()`), limited only to `kGridMargin` pixels outside the input;
  clamped close to the border, as the dense map's are, a cell across the border would sample the
  edge where the input has nothing.
- The VIC and the PVA cannot use CUDA memory in place: VPI copies a wrapped buffer for them and
  refreshes the copy only when the buffer is wrapped again. So the remap runs between VPI-owned
  images, and the operator's images are reached through wrappers only VPI's CUDA backend uses,
  copied on the GPU by `vpiSubmitCopyImage()`.
- A `RemapQueue` is the one place that waits. cuda never waits. vpi-vic and vpi-pva put every
  remapper's copy in, remap and copy out on one VPI stream that wraps the operator's CUDA stream,
  and `finish()` waits once for all of it: VPI allows no other work on that CUDA stream until its
  own is done, and has no fence the GPU could wait on. `remap_statistics()` counts every wait, and
  the tests pin the count per pair; a new wait must show up there.
- Use a VPI image, payload or wrapper on one VPI stream only: VPI reports `NvError_InvalidState`
  when a second stream that used it goes. The warm-up has throwaway objects of its own for this,
  and a remapper submitted to another VPI stream than its last (another queue, or a queue's new
  stream after a failure) makes new ones. A wrapper takes only memory of its first pitch; another
  pitch gets a new wrapper.
- A VPI status is a refusal (`std::invalid_argument`, which auto falls back from) only when it says
  VPI does not take the work here (`src/vpi_status.hpp`); any other is a failure.
- A CUDA pointer of a VPI image is valid only while the image is locked; after `vpiImageUnlock()`
  VPI may move the data between the engines' copies. Copy with VPI's own tasks instead.
- The runtime hands an input's memory back to its producer once the operator's stream is done
  with device memory, but at once for pinned host memory. Wait for the stream before `compute()`
  returns when it reads pinned input (`release_inputs()`); holding the tensor longer stalls a
  producer whose pool is full.
- The VIC corrupts the process's heap on an odd image size; `backend_limitation()` must refuse
  every size VPI cannot take before VPI sees it. New constraints go there, with a case in
  `tests/host/backend_rules_test.cpp`.
- An operator reads device input after `synchronize_for_consumer_stream()` on its own stream; the
  runtime does not order it there for the operator.
- Every declared output must be connected: an emit into an unconnected output fails (`kNotReady`),
  so the operator tests capture `camera_info` even where they do not check it.
- The operators' handles in a graph are not the instances the runtime runs: read nothing from them
  after the run. The backend auto chose is logged once; a caller that needs it calls
  `create_remapper(Backend::kAuto, ...)` with the same images, which chooses the same way.
