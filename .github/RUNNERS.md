# CI runners

[`workflows/ci.yml`](workflows/ci.yml) has three jobs. Their names are the checks branch protection
requires:

| Check | Runs on | What it does |
| --- | --- | --- |
| `Lint` | GitHub-hosted `ubuntu-24.04` | pre-commit, actionlint, the CI driver's unit tests, and every module's and demo's SDK-free tests (`tests/host/`) |
| `x86 GPU` | Self-hosted, labels `self-hosted, linux, x64, gpu` | `ci.py sdk` in the x86_64 CI image |
| `Jetson` | Self-hosted, labels `self-hosted, linux, arm64, jetson` | `ci.py sdk` in the Jetson CI image |

`ci.py sdk` builds each module, runs its full test suite, installs it, and builds and runs a
consumer project against the installed package; then it builds each demo against the installed
modules, runs its tests and runs its finite `ci` mode. Any failure fails the job.

Both GPU jobs build their image from [`docker/`](docker/) on the runner (Docker's layer cache keeps
this fast after the first run), run the build in a throwaway container as the runner's user, and
remove the container afterwards even when a step fails.

## Security

The repository is public and the GPU runners are machines on a private network, so:

- The GPU jobs never run automatically for a pull request from a fork. They run for pushes to
  `main`, for manual runs, and for pull requests whose branch is in this repository. A fork's pull
  request gets only `Lint`; to test it on the GPUs, a maintainer reviews the code and pushes it to a
  branch of this repository.
- The workflow has `permissions: contents: read`, never uses `pull_request_target`, and checks out
  with `persist-credentials: false`, so the build cannot push or read secrets.
- Every action is pinned to a commit, and the base images to a digest.

Because a skipped job counts as passing for branch protection, review a fork's pull request
yourself before merging it.

## Repository settings

- **Settings > Actions > General > Fork pull request workflows from outside collaborators**:
  "Require approval for all external contributors".
- **Settings > Actions > General > Workflow permissions**: "Read repository contents and packages
  permissions".
- **Settings > Branches > Branch protection rule for `main`** (or a ruleset): require a pull
  request, require these status checks to pass: `Lint`, `x86 GPU`, `Jetson`; require branches to
  be up to date before merging; and do not allow bypassing these settings.

## Registering a runner

Register each machine as a repository runner: **Settings > Actions > Runners > New self-hosted
runner** shows the download and `config.sh` commands with a one-time token. Add the custom label
when `config.sh` asks, or pass `--labels`:

```bash
# x86_64 workstation with an NVIDIA GPU
./config.sh --url https://github.com/snknitheesh/holoscan-zozo-modules --token <token> \
  --name x86-gpu --labels gpu --unattended
# Jetson Thor
./config.sh --url https://github.com/snknitheesh/holoscan-zozo-modules --token <token> \
  --name jetson-thor --labels jetson --unattended
```

The runner adds `self-hosted`, `linux` and `x64` or `arm64` itself. Install it as a service so it
survives reboots, under a dedicated user that is in the `docker` group:

```bash
sudo ./svc.sh install <runner-user>
sudo ./svc.sh start
```

A job waits in the queue until a runner with its labels is online.

### x86_64 with an NVIDIA GPU

1. An NVIDIA driver for CUDA 13 (R580 or newer).
2. [Docker Engine](https://docs.docker.com/engine/install/ubuntu/) with the Buildx plugin.
3. The [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
   configured for Docker:

   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   docker run --rm --gpus all ubuntu:24.04 nvidia-smi   # must list the GPU
   ```

4. About 60 GB of free disk for the SDK image and the build trees.

### Jetson Thor

1. JetPack 7 (L4T R38), which brings the CUDA 13.0 driver, the PVA firmware and the NVIDIA
   Container Toolkit with its `nvidia` runtime.
2. Docker Engine with the Buildx plugin, and the `nvidia` runtime registered:

   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   docker run --rm --runtime nvidia nvcr.io/nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi
   ```

3. The job runs its container with `--runtime nvidia --privileged`. The PVA needs both, and the
   `nvidia` runtime also mounts the Jetson Linux libraries that VPI links against (`libnvrm_*`,
   `libnvmedia`, `libnvbufsurface`, `libnvvic`, `libnvos`), so even building with VPI needs it.
   It runs without `DISPLAY`: with `DISPLAY` set, even to an empty value, VPI fails with
   `EGL_NOT_INITIALIZED`. Do not set `DISPLAY` in the runner service's environment.
4. About 60 GB of free disk on the NVMe drive that holds Docker's data root.

## CI images

### x86_64: [`docker/x86_64.Dockerfile`](docker/x86_64.Dockerfile)

The official `nvcr.io/nvidia/clara-holoscan/holoscan:v5.0.0-cuda13` image, pinned by digest, which
already ships CMake 4; plus VPI 4.0.5 from NVIDIA's Jetson repository (`x86_64/noble`, release
`r38.4`), accepted only with the repository key's pinned fingerprint; plus `holoscan-cli` 5.0.0.

### Jetson: [`docker/jetson.Dockerfile`](docker/jetson.Dockerfile)

`docker manifest inspect nvcr.io/nvidia/clara-holoscan/holoscan:v5.0.0-cuda13` shows an arm64
variant, but it is the discrete-GPU (SBSA) build: CUDA 13.2 user-space libraries with a CUDA 13.2
data-center forward-compatibility driver in `/usr/local/cuda/compat`, which the image puts first on
the library path. JetPack 7 ships a CUDA 13.0 driver for the integrated GPU, and that
forward-compatibility driver is not made for it. The image also has no VPI.

The Jetson image is therefore built the way the SDK was validated on Thor:

- `nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu24.04` (arm64), pinned by digest, matching JetPack 7's
  CUDA 13.0;
- the Holoscan SDK 5.0.0 arm64 packages from NVIDIA's `ubuntu2404/sbsa` CUDA repository:
  `holoscan-core-cuda-13`, `libholoscan-visualization-cuda-13` and `-dev`, and
  `libholoscan-inference-cuda-13` and `-dev`, each checked against its SHA-256, with TensorRT
  pinned to its CUDA 13.0 build;
- VPI 4.0.5 from the Jetson repository (`common`, release `r38.4`), whose PVA backend the
  `nvidia` runtime connects to the board's PVA, with the key fingerprint pinned;
- CMake 4.4.4 and `holoscan-cli` 5.0.0 from PyPI.

This choice is not yet proven on the runner: run the `Jetson` job once by hand
(`workflow_dispatch`) after registering the Thor. The job needs only `python3`, CMake 4, the SDK
and VPI in its image, so switching images later touches only the Dockerfile.
