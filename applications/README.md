# Applications

Demo applications for the modules in [`modules/`](../modules/). Each one uses its module the way
another project would: it finds the installed package with `find_package`, never the module's
sources.

## What CI expects of a demo

CI discovers every `applications/<demo>/` that has a `metadata.json`, so a new demo needs no
workflow change. A demo must have:

- `metadata.json` with the `application` envelope (`urn:holoscan:application:v1`) and a `ci` mode:
  a finite, headless run that exits 0 only when it succeeded. `modes.ci.run.command` may use
  `<holohub_app_bin>`, which CI replaces with the demo's build directory, and
  `modes.ci.build.cmake_options` adds CMake options for that build.
- `CMakeLists.txt`, a standalone CMake project. CI configures it with the modules' install prefix in
  `CMAKE_PREFIX_PATH` and `-DBUILD_TESTING=ON`, builds it, runs its CTest tests if it has any, and
  then runs the `ci` mode.

Optionally, `tests/host/` is a standalone CMake project for tests that need no Holoscan SDK, CUDA
or GPU; the hosted `Lint` job runs it.

The tests and the `ci` mode see `HOLOSCAN_CI_PLATFORM`, `x86_64` or `jetson`, for checks that only
hold on one platform.
