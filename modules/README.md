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
