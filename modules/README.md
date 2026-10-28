# Modules

Each folder here is one Holoscan Module, and this folder is what HoloHub registers: it finds every
module by its `metadata.json`. The [repository README](../README.md) lists the modules.

## What CI expects of a module

CI discovers every `modules/<name>/`, so a new module needs no workflow change. Every folder here
(other than a hidden one) is a module, and one without `metadata.json` fails CI rather than being
skipped. A module must have:

- `metadata.json` with the `module` envelope (`urn:holoscan:module:v2`), whose `name` is
  `holoscan-<name>`. The lint hooks check it against the Holoscan CLI's schema.
- `CMakeLists.txt`, a standalone CMake project against an installed Holoscan SDK. With
  `-DBUILD_TESTING=ON` it builds the module's full test suite and registers it with CTest; the GPU
  jobs run all of it, and it must not be empty. `cmake --install` installs the headers, the
  libraries and a CMake package config, `holoscan-<name>-config.cmake`.
- `tests/installed_consumer/`, a standalone CMake project that finds the installed package (CI
  passes the install prefix in `CMAKE_PREFIX_PATH`), links its targets and registers at least one
  CTest test that runs them.

CI builds and installs the modules in name order, and each module's build gets the install prefix
in `CMAKE_PREFIX_PATH`, so a module may `find_package` another module of this collection only if
that module's name sorts before its own.

The GPU jobs set `HOLOSCAN_CI_PLATFORM` to `x86_64` or `jetson` for the tests, so a test can
require what holds only on one platform, such as the Jetson's PVA.

Optionally, `tests/host/` is a standalone CMake project for tests that need no Holoscan SDK, CUDA
or GPU. The hosted `Lint` job builds and runs it on every pull request.

Other conventions:

- Public headers go under `include/holoscan/<name>/` and are included as `<holoscan/<name>/...>`.
- Targets are exported in the `holoscan::` namespace, and code lives in `holoscan::<name>`.
- The module's README covers its operators, requirements, build options, tests and the licenses of
  what it uses, and the module adds its row to the repository README.

## What the root build expects of a module

The repository's root [`CMakeLists.txt`](../CMakeLists.txt) adds each enabled module with
`add_subdirectory()`, inside whatever project includes the repository: its own build, a
consumer's `FetchContent` or `add_subdirectory()`, or the Holoscan CLI's
`holohub_declare_external_module()`. The module is then not the top-level project, and its targets
live in the consumer's build. For that to work, a module must:

1. **Name its operators in `metadata.json`.** `module.operator_names` lists the operator classes
   (`AprilTagDetectorOp`). Each must be a C identifier, unique in the collection: the root makes
   one `OP_<operator>` option of each, and any of them ON enables the module. `module.name` gives
   the module's option, `MODULE_holoscan_<name>` (`-` becomes `_`), which enables all of it.
2. **Define its targets in the build tree under their exported names.** For every target it
   exports as `holoscan::<target>`, `add_library(holoscan::<target> ALIAS <library>)`. The root
   checks for `holoscan::<name>` (`-` becomes `_`), and a consumer links these aliases without
   `find_package(holoscan-<name>)`, which finds only an installed package.
3. **Use its own paths only.** `CMAKE_CURRENT_SOURCE_DIR` and `CMAKE_CURRENT_BINARY_DIR`, or
   `PROJECT_SOURCE_DIR` and `PROJECT_BINARY_DIR` after its own `project()`; never
   `CMAKE_SOURCE_DIR` or `CMAKE_BINARY_DIR`, which are the consumer's. The packaging is the one
   exception: `holohub_configure_deb()` writes its CPack configuration to
   `${CMAKE_BINARY_DIR}/pkg/`, where `holoscan package` and the release workflow look for it.
4. **Find the SDK with the components it needs, unconditionally.**
   `find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS ...)` loads only what is not loaded yet:
   the SDK's config skips every `holoscan::` target that is already defined, whether the consumer
   imported it or another module did, so finding it again is safe and cheap. Do not guard the call
   with `if(NOT holoscan_FOUND)` or `if(NOT TARGET holoscan::core)`: the consumer may have found
   other components, and yours would be missing.
5. **Leave the consumer's switches alone.** Declare `BUILD_TESTING` and `BUILD_SHARED_LIBS` as
   options only when `PROJECT_IS_TOP_LEVEL`; otherwise set what the module needs as normal
   variables, in its own scope. Prefix every other option and cache variable with
   `HOLOSCAN_<NAME>_`.
6. **Build its tests only on its own, or when asked.** A consumer's `include(CTest)` turns
   `BUILD_TESTING` on, and must not pull in the module's test suite and the SDK's `testing`
   component:

   ```cmake
   if(PROJECT_IS_TOP_LEVEL)
     option(BUILD_SHARED_LIBS "Build shared libraries" ON)
     option(BUILD_TESTING "Build the module's tests" OFF)
     set(HOLOSCAN_<NAME>_BUILD_TESTING ${BUILD_TESTING})
   else()
     set(BUILD_SHARED_LIBS ON)
     option(HOLOSCAN_<NAME>_BUILD_TESTING "Build the module's tests in the including project" OFF)
   endif()
   ```

   and use `HOLOSCAN_<NAME>_BUILD_TESTING` wherever the module used `BUILD_TESTING`.
7. **Set `CMAKE_CUDA_ARCHITECTURES` only when it is not defined** (`native` for a development
   build, `all-major` for the package), before its `project()`, so that a consumer's choice holds.
8. **Keep packaging opt-in.** `PKG_holoscan_<name>` (`-` becomes `_`) defaults to `OFF` and, when
   ON, adds `pkg/holoscan-<name>/`, whose `holohub_configure_deb()` writes
   `${CMAKE_BINARY_DIR}/pkg/CPackConfig-holoscan-<name>.cmake`.

Optionally, a module may leave out a part that only some of its operators need. The root sets
`HOLOSCAN_ZOZO_MODULES_OPERATORS` to the operators asked for (all of them for
`MODULE_holoscan_<name>`); it is undefined when the module is built on its own:

```cmake
set(_overlay_default ON)
if(DEFINED HOLOSCAN_ZOZO_MODULES_OPERATORS)
  if(NOT "AprilTagOverlayOp" IN_LIST HOLOSCAN_ZOZO_MODULES_OPERATORS)
    set(_overlay_default OFF)
  endif()
endif()
option(HOLOSCAN_APRILTAG_BUILD_OVERLAY "Build AprilTagOverlayOp" ${_overlay_default})
```

The root builds the modules in name order, as CI does. A module that depends on another module of
this collection links that module's `holoscan::` targets when they are defined
(`if(TARGET holoscan::<other>)`), and otherwise finds its installed package.
