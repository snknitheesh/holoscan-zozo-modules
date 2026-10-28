# Holoscan zozo Modules

A collection of [Holoscan SDK](https://github.com/nvidia-holoscan/holoscan-sdk) 5.0 Modules:
reusable C++ operators that another Holoscan project installs and finds with `find_package`, the
way it uses the SDK's own modules. Each module also comes with a demo application that runs it
end to end.

The collection is registered once in [HoloHub](https://github.com/nvidia-holoscan/holohub), as a
module source whose path is [`modules/`](modules/).

## Modules

| Module | CMake package and targets | C++ namespace | What it does |
| --- | --- | --- | --- |
| [holoscan-apriltag](modules/apriltag/) | `holoscan-apriltag`: `holoscan::apriltag`, `holoscan::apriltag_overlay` | `holoscan::apriltag` | AprilTag detection and 6-DoF poses on the CPU, the GPU (CUDA) or the PVA (VPI 4), and a HoloViz overlay. Demo: [AprilTag Detection](applications/apriltag_detection/) |
| [holoscan-point-cloud](modules/point-cloud/) | `holoscan-point-cloud`: `holoscan::point_cloud` | `holoscan::point_cloud` | Depth images to organized, optionally colored point clouds on the GPU. Demo: [Depth to Point Cloud](applications/depth_to_point_cloud/) |
| [holoscan-yolo26](modules/yolo26/) | `holoscan-yolo26`: `holoscan::yolo26`, `holoscan::yolo26_overlay` | `holoscan::yolo26` | End-to-end YOLO26 object detection on the GPU: CUDA letterbox, TensorRT through HoloInfer and GPU decode, and a HoloViz overlay. The AGPL-3.0 model is not included; a tool exports it. Demo: [YOLO26 Detection](applications/yolo26_detection/) |

Each module's README lists its operators, requirements, build options and tests.

## Layout

```text
CMakeLists.txt           Builds the enabled modules together; what FetchContent and the CLI fetch
modules/<name>/          One Holoscan Module: a standalone CMake project
  metadata.json          Module metadata (urn:holoscan:module:v2), read by HoloHub and the CLI
  include/holoscan/<name>/  Installed public headers
  src/                   Implementation
  tests/                 Its full test suite (CTest)
  tests/host/            Optional: tests that need no SDK, CUDA or GPU
  tests/installed_consumer/  A project that uses the installed package
  pkg/                   Debian packaging
applications/<demo>/     A demo application, built against the installed module
  metadata.json          Application metadata with a finite, headless `ci` mode
.github/                 CI workflows, the CI image, scripts and runner setup
```

[modules/README.md](modules/README.md) and [applications/README.md](applications/README.md) give
the rules a module or a demo follows so that CI picks it up without changes.

## Requirements

- Linux on x86_64 with an NVIDIA GPU and a driver for CUDA 13.2 (R595 or newer), or on aarch64:
  Jetson Thor or Jetson Orin with JetPack 7.2.
- Holoscan SDK 5.0 with CUDA 13, CMake 4.0 or later and a C++20 compiler.
- Module-specific dependencies, listed in each module's README.

The CI container has all of these, for both architectures; see
[Use the CI container](#use-the-ci-container).

## Use the CI container

Every CI job runs in one image, built from
[`.github/docker/Dockerfile`](.github/docker/Dockerfile) for `linux/amd64` and `linux/arm64`
and published to the GitHub Container Registry. It has CUDA 13.2, the Holoscan SDK 5.0.0,
TensorRT, VPI 4, CMake 4 and the Holoscan CLI, and makes a good development container:

```bash
docker pull ghcr.io/snknitheesh/holoscan-zozo-modules:latest
```

From a clone of this repository, build and test every module and demo the way CI does. On
x86_64 with an NVIDIA GPU and an R595 or newer driver:

```bash
docker run --rm --init --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD:$PWD" -w "$PWD" ghcr.io/snknitheesh/holoscan-zozo-modules:latest \
  python3 .github/scripts/ci.py sdk --platform x86_64
```

On Jetson Thor or Jetson Orin with JetPack 7.2 (the PVA needs `--privileged`, and
`DISPLAY` must stay unset):

```bash
docker run --rm --init --runtime nvidia --privileged --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD:$PWD" -w "$PWD" ghcr.io/snknitheesh/holoscan-zozo-modules:latest \
  python3 .github/scripts/ci.py sdk --platform jetson
```

To build the image yourself instead:

```bash
docker build --force-rm -f .github/docker/Dockerfile -t holoscan-modules-ci:local .github/docker
```

**Notice:** the image contains NVIDIA CUDA, TensorRT, VPI and the Holoscan SDK, which are
distributed under NVIDIA's own licenses, not this repository's. By pulling or using the image
you accept those licenses.

## Build, test and install a module

Each module is a standalone CMake project:

```bash
cmake -S modules/<name> -B build/<name> -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build/<name>
ctest --test-dir build/<name> --output-on-failure
cmake --install build/<name> --prefix "$PWD/install"
```

To build a Debian package instead, see the module's README.

The repository root is a CMake project too, which builds the modules together, in one build tree:
all of them by default, or a selection with `BUILD_ALL=OFF` and the options of
[Build it from source](#b-build-it-from-source-with-fetchcontent):

```bash
cmake -S . -B build/all -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake -S . -B build/apriltag -G Ninja -DBUILD_ALL=OFF -DMODULE_holoscan_apriltag=ON
```

## Use a module in your project

A project uses a module in one of three ways, as NVIDIA's
[Use a Holoscan Module](https://github.com/nvidia-holoscan/holoscan-tutorials/tree/main/tutorials/holoscan-modules/use-a-module)
tutorial describes them: (A) install its Debian package, (B) build it from source with CMake's
`FetchContent`, or (C) declare it in a Holoscan CLI project's `metadata.json`. In B and C the
module is built as part of the consuming project, from this repository's root
[`CMakeLists.txt`](CMakeLists.txt), which builds only the modules that project enables. The
examples use `holoscan-apriltag` and its `AprilTagDetectorOp`; every module's README names its
operators and targets.

### A. Install its Debian package

Each released version of a module is a [GitHub Release](https://github.com/snknitheesh/holoscan-zozo-modules/releases)
tagged `<name>-v<version>`, with a package for `amd64` (x86_64) and one for `arm64` (Jetson Thor
and Orin with JetPack 7.2), and their SHA-256 sums. The package depends on `holoscan (>= 5.0.0)`,
so APT needs NVIDIA's CUDA repository, and on what the module's README lists (the AprilTag
module's VPI backend needs NVIDIA's Jetson repository for `libnvvpi4`):

```bash
release=https://github.com/snknitheesh/holoscan-zozo-modules/releases/download/apriltag-v1.0.0
deb="holoscan-apriltag_1.0.0_$(dpkg --print-architecture).deb"
curl -fL -O "$release/$deb" -O "$release/SHA256SUMS"
sha256sum --check --ignore-missing SHA256SUMS
sudo apt install "./$deb"
```

It installs to `/usr`, where CMake finds it. Find the package and link its targets:

```cmake
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core)
find_package(holoscan-apriltag 1.0 REQUIRED CONFIG)
target_link_libraries(my_app PRIVATE holoscan::apriltag)
```

A module installed from source with `cmake --install` works the same way, with its prefix on
`CMAKE_PREFIX_PATH`:

```bash
cmake -S my_app -B build/my_app -DCMAKE_PREFIX_PATH="$PWD/install"
```

### B. Build it from source with FetchContent

Fetch the repository at a pinned commit, and enable what you use before making it available:
`OP_<operator>` builds the module that has that operator, `MODULE_holoscan_<name>` all of a
module (`-` in the name becomes `_`). Nothing else is built:

```cmake
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core)

set(OP_AprilTagDetectorOp ON)
include(FetchContent)
FetchContent_Declare(
  holoscan_zozo_modules
  GIT_REPOSITORY https://github.com/snknitheesh/holoscan-zozo-modules.git
  GIT_TAG <40-character commit SHA>
)
FetchContent_MakeAvailable(holoscan_zozo_modules)

target_link_libraries(my_app PRIVATE holoscan::apriltag)
```

The module's targets are part of your build, so link them without `find_package(holoscan-apriltag)`.
Its own options, in its README, apply as well; a module builds its tests only when asked
(`HOLOSCAN_<NAME>_BUILD_TESTING=ON`), as [modules/README.md](modules/README.md) requires. To build against a local checkout instead of fetching,
configure with `-DFETCHCONTENT_SOURCE_DIR_HOLOSCAN_ZOZO_MODULES=/path/to/holoscan-zozo-modules`, or
use `add_subdirectory(/path/to/holoscan-zozo-modules holoscan-zozo-modules)` in place of the
`FetchContent` calls.

| Option | Default | What it does |
| --- | --- | --- |
| `BUILD_ALL` | `ON` when the repository is the top-level project, else `OFF` (or the including project's `BUILD_ALL`) | Enables every module |
| `MODULE_holoscan_<name>` | `BUILD_ALL` | Enables the module, all of its operators |
| `OP_<operator>` | `OFF` | Enables the module that has the operator; one for each entry of `operator_names` in a module's `metadata.json` |

### C. Declare it in a Holoscan CLI project

In a project built with the [Holoscan CLI](https://github.com/nvidia-holoscan/holoscan-cli)
(`holoscan build`), declare the module in the application's `metadata.json`. `name` is the
module's own `module.name`, `git_url` this repository, and `ref` a full commit SHA, which the CLI
requires:

```json
"dependencies": {
  "modules": [
    {
      "name": "holoscan-apriltag",
      "source": {
        "git_url": "https://github.com/snknitheesh/holoscan-zozo-modules.git",
        "ref": "<40-character commit SHA>"
      },
      "provides_operators": ["AprilTagDetectorOp"]
    }
  ]
}
```

`holoscan build` writes a `FetchContent` declaration of this repository, with the provider name
`holoscan_apriltag`, to `external_operators_manifest.cmake` in the application's build folder. The
project's root `CMakeLists.txt` includes it, enables the operators its applications use, and then
makes the providers of the enabled operators available, as in section 6.3 of the tutorial. With the
CLI's `HoloHubConfigHelpers.cmake` in `cmake/`:

```cmake
include(FetchContent)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/HoloHubConfigHelpers.cmake")
include("${CMAKE_BINARY_DIR}/external_operators_manifest.cmake" OPTIONAL)

# add_holohub_application(my_app DEPENDS OPERATORS AprilTagDetectorOp) sets OP_AprilTagDetectorOp.
add_subdirectory(applications)

get_cmake_property(_variables VARIABLES)
set(_providers "")
foreach(_variable IN LISTS _variables)
  if(_variable MATCHES "^HOLOHUB_EXT_OP_(.+)_PROVIDER$")
    if(OP_${CMAKE_MATCH_1})
      list(APPEND _providers "${${_variable}}")
    endif()
  endif()
endforeach()
list(REMOVE_DUPLICATES _providers)
if(_providers)
  FetchContent_MakeAvailable(${_providers})
endif()
```

The application links `holoscan::apriltag` as in B. Then:

```bash
holoscan build my_app --dryrun --verbose
holoscan build my_app
```

For several modules of this collection, add one entry for each, all with the same `ref`; the
first one fetched builds every module enabled by then, and the others add nothing. To build against
a local checkout, point the CLI's override for the module at the module's folder (in a container
build, also mount it and pass `--forward-env`, as the tutorial shows):

```bash
export HOLOSCAN_CLI_LOCAL_HOLOSCAN_APRILTAG="$HOME/src/holoscan-zozo-modules/modules/apriltag"
holoscan build my_app --local
```

A single entry named `holoscan-zozo-modules`, whose `provides_operators` lists the operators of
every module you use, works too and fetches the repository once. The CLI's override cannot point
at it, though, because it requires a `metadata.json` at the root of the folder it names, and this
repository has none: build against a checkout with
`holoscan build my_app --configure-args=-DFETCHCONTENT_SOURCE_DIR_HOLOSCAN_ZOZO_MODULES=/path/to/holoscan-zozo-modules`
instead.

## Run a demo

Install the module first, then build the demo against it and run it:

```bash
cmake -S applications/<demo> -B build/<demo> -DCMAKE_PREFIX_PATH="$PWD/install"
cmake --build build/<demo>
```

Each demo's README has its options and modes.

## Run CI locally

[`.github/scripts/ci.py`](.github/scripts/ci.py) is what the GPU jobs run: for every module, in
dependency order, it builds, runs the full test suite, installs, and builds and runs the
installed-package consumer; then it builds every demo against the installed modules and runs its
`ci` mode. A pull request runs only what it changes and what depends on it; see
[What a run tests](.github/RUNNERS.md#what-a-run-tests).

```bash
python3 .github/scripts/ci.py list                                  # modules, demos, dependencies
python3 .github/scripts/ci.py select --changed-since origin/main    # what a pull request tests
```

Running the whole of `ci.py sdk` in the CI container is shown in
[Use the CI container](#use-the-ci-container).

The workflows, their runners, the image's publishing and the repository settings are described in
[.github/RUNNERS.md](.github/RUNNERS.md); releasing a module in
[.github/RELEASING.md](.github/RELEASING.md).

## Contributing

Every change goes through a pull request into `main`. Sign off each commit (`git commit -s`) to
certify the [Developer Certificate of Origin](https://developercertificate.org/), and run the
linters before pushing. They run with [prek](https://prek.j178.dev), as in CI; `pre-commit` reads
the same `.pre-commit-config.yaml` and works too. Both check tracked files only, so `git add` new
files first.

```bash
python3 -m pip install -r requirements.txt
prek install
prek run --all-files
```

## License

Apache License 2.0; see [LICENSE](LICENSE). Each module's README lists the licenses of the
third-party software it uses. The CI container image also holds NVIDIA software under NVIDIA's
licenses; see the notice in [Use the CI container](#use-the-ci-container).
