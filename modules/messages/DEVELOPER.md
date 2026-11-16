# holoscan-messages developer guide

## Layout

```text
messages/
├── CMakeLists.txt               Standalone project against an installed Holoscan SDK 5.0
├── metadata.json                Module metadata (urn:holoscan:module:v2)
├── cmake/                       Package config template; the CLI template's Debian helper
├── schemas/                     The three messages' FlatBuffers schemas
├── include/holoscan/messages/   Installed public API (the generated headers join it at build time)
├── src/                         Implementation; private headers
├── pkg/holoscan-messages/       Debian packaging, entered when PKG_holoscan_messages is ON
└── tests/
    ├── host/                    SDK-free tests, also built on their own by the hosted CI job
    └── installed_consumer/      Uses the installed package
```

| Part | Files | Needs |
| --- | --- | --- |
| Plain values and checks | `camera_info.hpp`, `camera_info.cpp`, `geometry.hpp` | Nothing |
| camera_info files | `camera_info_yaml.hpp`, `camera_info_yaml.cpp`, `yaml_reader.hpp`, `yaml_reader.cpp` | Nothing |
| Point cloud layout | `point_cloud_layout.hpp`, `point_cloud_layout.cpp` | Nothing |
| Synthetic calibration | `synthetic_calibration.hpp`, `synthetic_calibration.cpp` | Nothing |
| Messages | `schemas/*.fbs`, `conversion.hpp`, `conversion.cpp` | Holoscan SDK |
| Consumer contract | `camera_info_input.hpp`, `camera_info_input.cpp` | Holoscan SDK |
| Point cloud ports | `point_cloud.hpp`, `point_cloud.cpp` | Holoscan SDK, CUDA runtime headers |
| `CameraInfoSourceOp` | `camera_info_source_op.hpp`, `camera_info_source_op.cpp` | Holoscan SDK |

The SDK-free parts are compiled again, from source, by `tests/host/`, so the hosted CI job tests them
without the SDK or a GPU.

## Build and package boundary

| | |
| --- | --- |
| Build contexts | Top-level CMake project, or included by another project (`add_subdirectory()`, `FetchContent`, the repository root) |
| Options | Top level: `BUILD_TESTING` (`OFF`), `BUILD_SHARED_LIBS` (`ON`). Included: `HOLOSCAN_MESSAGES_BUILD_TESTING` (`OFF`), always shared. Both: `PKG_holoscan_messages` (`OFF`) |
| Public targets | `holoscan::messages` (`libholoscan_messages.so.1`) |
| Include root | `include/` and the generated headers, installed to `<prefix>/include`; the schemas to `<prefix>/share/holoscan-messages/schemas` |
| Package | `<prefix>/lib/cmake/holoscan-messages/holoscan-messages-config.cmake`, `SameMajorVersion` |
| Debian package | `holoscan-messages`, prefix `/usr`; depends on `holoscan (>= 5.0.0)` |
| External dependencies | Holoscan SDK 5.0 (`core`), the CUDA runtime |
| Test labels | `messages` and `gpu`, or `messages` and `host` |

## Conventions

- Code lives in `holoscan::messages`, the generated messages in `holoscan::messages::schema`.
- Schema changes keep each root's name, file identifier and explicit field ids, and only append
  fields. A new field also goes into the plain value, both conversions, `validate()`,
  `same_calibration()` and, for camera info, the YAML reader and writer when the file holds it.
- `MAX_SERIALIZED_SIZE` of each schema holds frame ids of `kMaxFrameIdBytes`; the graph tests send
  messages with the longest frame ids. A field that can grow needs a bound in `validate()` and a
  larger size.
- The YAML reader refuses what it does not support; never let it guess. Every refusal of a
  camera_info file names the key or line.
- Every check throws `std::invalid_argument` naming the field; the SDK-facing helpers turn it into
  `kInvalidArgument`.
- A PointCloud always carries all four tensors; absent is `[0]`. Do not use a null tensor: the
  SDK's descriptor codec refuses it.
- The consumer contract is measured, not assumed: a change to it needs a case in
  `messages_contract` for camera info at its own rate, with every frame, and before the first frame.
- An operator has an optional port only when configured: an emit into an unconnected output fails.
