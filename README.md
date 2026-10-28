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
all of them by default, or a selection with `BUILD_ALL=OFF` and the `MODULE_holoscan_<name>` or
`OP_<operator>` options described in [modules/README.md](modules/README.md#what-the-root-build-expects-of-a-module):

```bash
cmake -S . -B build/all -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake -S . -B build/apriltag -G Ninja -DBUILD_ALL=OFF -DMODULE_holoscan_apriltag=ON
```

## Use a module in your project

Point CMake at the install prefix (not needed for the Debian package, which installs to `/usr`),
then find the module's package and link its targets:

```cmake
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core)
find_package(holoscan-<name> REQUIRED CONFIG)
target_link_libraries(my_app PRIVATE holoscan::<name>)
```

```bash
cmake -S my_app -B build/my_app -DCMAKE_PREFIX_PATH="$PWD/install"
```

## Run a demo

Install the module first, then build the demo against it and run it:

```bash
cmake -S applications/<demo> -B build/<demo> -DCMAKE_PREFIX_PATH="$PWD/install"
cmake --build build/<demo>
```

Each demo's README has its options and modes.

## Run CI locally

[`.github/scripts/ci.py`](.github/scripts/ci.py) is what the GPU jobs run: for every module it
builds, runs the full test suite, installs, and builds and runs the installed-package consumer;
then it builds every demo against the installed modules and runs its `ci` mode.

```bash
python3 .github/scripts/ci.py list
```

Running the whole of `ci.py sdk` in the CI container is shown in
[Use the CI container](#use-the-ci-container).

The workflows, their runners, the image's publishing and the repository settings are described in
[.github/RUNNERS.md](.github/RUNNERS.md).

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
