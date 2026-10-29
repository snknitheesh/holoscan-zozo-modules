# Applications

Demo applications for the modules in [`modules/`](../modules/). Each one uses its module the way
another project would: it finds the installed package with `find_package`, never the module's
sources.

## What CI expects of a demo

CI discovers every `applications/<demo>/`, so a new demo needs no workflow change. Every folder
here (other than a hidden one) is a demo, and one without `metadata.json` fails CI rather than
being skipped. A demo must have:

- `metadata.json` with the `application` envelope (`urn:holoscan:application:v1`) and a `ci` mode:
  a finite, headless run that exits 0 only when it succeeded. `modes.ci.run.command` may use
  `<holohub_app_bin>`, which CI replaces with the demo's build directory, and
  `modes.ci.build.cmake_options` adds CMake options for that build. CI stops the run, with
  everything it started, and fails after `modes.ci.run.timeout` seconds (default 900).
- `CMakeLists.txt`, a standalone CMake project. CI configures it with the modules' install prefix in
  `CMAKE_PREFIX_PATH` and `-DBUILD_TESTING=ON`, builds it, runs its CTest tests if it has any, and
  then runs the `ci` mode.
- In `metadata.json`, every module it uses, in `dependencies.modules`, by the module's folder name
  in `modules/`, which is where the Holoscan CLI looks for it:

  ```json
  "dependencies": {
    "modules": [
      {
        "name": "point-cloud"
      }
    ]
  }
  ```

  CI fails when the demo's CMake files find a `holoscan-<name>` package of a module it does not
  declare, or when it declares a name that is not a module.

A pull request runs a demo when it changes something in the demo's folder, other than its Markdown
files, or in a module it uses, or one those depend on. CI then builds and installs the modules it
uses first, and runs their tests only if the pull request selected them too. See
[What a run tests](../.github/RUNNERS.md#what-a-run-tests), which also says how to run every test.

Optionally, `tests/host/` is a standalone CMake project for tests that need no Holoscan SDK, CUDA
or GPU; the hosted `Lint` job runs it.

The tests and the `ci` mode see `HOLOSCAN_CI_PLATFORM`, `x86_64` or `jetson`, for checks that only
hold on one platform.
