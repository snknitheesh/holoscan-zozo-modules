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
.github/                 CI workflow, CI images, scripts and runner setup
```

[modules/README.md](modules/README.md) and [applications/README.md](applications/README.md) give
the rules a module or a demo follows so that CI picks it up without changes.

## Requirements

- Linux on x86_64 or aarch64 (Jetson Thor with JetPack 7) with an NVIDIA GPU.
- Holoscan SDK 5.0 with CUDA 13, CMake 4.0 or later and a C++20 compiler.
- Module-specific dependencies, listed in each module's README.

The CI images in [`.github/docker/`](.github/docker/) have all of these, and make good
development containers:

```bash
docker build -f .github/docker/x86_64.Dockerfile -t holoscan-modules-ci:x86_64 .github/docker
docker run --rm -it --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "$PWD:$PWD" -w "$PWD" holoscan-modules-ci:x86_64 bash
```

## Build, test and install a module

Each module is a standalone CMake project:

```bash
cmake -S modules/<name> -B build/<name> -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build/<name>
ctest --test-dir build/<name> --output-on-failure
cmake --install build/<name> --prefix "$PWD/install"
```

To build a Debian package instead, see the module's README.

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
docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all -e HOME=/tmp \
  --user "$(id -u):$(id -g)" -v "$PWD:$PWD" -w "$PWD" \
  holoscan-modules-ci:x86_64 python3 .github/scripts/ci.py sdk
```

The workflow, its runners and the repository settings are described in
[.github/RUNNERS.md](.github/RUNNERS.md).

## Contributing

Every change goes through a pull request into `main`. Sign off each commit (`git commit -s`) to
certify the [Developer Certificate of Origin](https://developercertificate.org/), and run the
linters before pushing:

```bash
python3 -m pip install -r requirements.txt
pre-commit install
pre-commit run --all-files
```

## License

Apache License 2.0; see [LICENSE](LICENSE). Each module's README lists the licenses of the
third-party software it uses.
