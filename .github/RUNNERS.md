# CI runners

[`workflows/ci.yml`](workflows/ci.yml) has four jobs. Their names are the checks branch protection
requires:

| Check | Runs on | What it does |
| --- | --- | --- |
| `Lint` | GitHub-hosted `ubuntu-24.04` | pre-commit, actionlint, the CI driver's unit tests, and every module's and demo's SDK-free tests (`tests/host/`) |
| `x86 GPU` | Self-hosted, labels `self-hosted, linux, x64, gpu` | `ci.py sdk --platform x86_64` in the CI image |
| `Jetson Thor` | Self-hosted, labels `self-hosted, linux, arm64, jetson, thor` | `ci.py sdk --platform jetson` in the CI image |
| `Jetson Orin` | Self-hosted, labels `self-hosted, linux, arm64, jetson, orin` | `ci.py sdk --platform jetson` in the CI image |

`ci.py sdk` builds each module, runs its full test suite, installs it, and builds and runs a
consumer project against the installed package; then it builds each demo against the installed
modules, runs its tests and runs its finite `ci` mode. Any failure fails the job.

The GPU jobs all build the same image, [`docker/Dockerfile`](docker/Dockerfile), on the runner for
its own architecture with `docker buildx build`, and tag it `holoscan-modules-ci:local` (Docker's
BuildKit cache keeps this fast after the first run). `--platform` tells `ci.py` which lane it is,
for tests and demos that only hold on Jetson.

## No container outlives its job

[`scripts/ci_container.py`](scripts/ci_container.py) runs the tests as the runner's user in a
container labeled `io.github.snknitheesh.holoscan-zozo-modules.ci` (plus `...ci.run` with the run
ID), started with `--rm` and `--init`. Each GPU job:

1. removes every container with that label (`ci_container.py sweep`). A runner runs one job at a
   time, so any of them is left over;
2. builds the image;
3. runs the tests with `exec ci_container.py run`, so the runner's cancel signals (SIGINT, then
   SIGTERM) reach it, and it removes the container with `docker rm -f` on any exit or signal;
4. sweeps again in an `if: always()` step, and removes this repository's untagged images (earlier
   builds of the image). It trims Docker's BuildKit cache, least recently used first, only when
   the disk has less than 100 GB free (`CI_BUILD_CACHE_MIN_FREE_GB`), because that cache is shared
   with everything else built on the machine.

The job's `timeout-minutes` bounds a hung test. If the job dies before its own cleanup (a runner
crash, a killed docker CLI), the runner's [job hook](#install-the-job-hook) removes the container
before and after the next job. Containers without the label, and images without this
repository's `org.opencontainers.image.source` label or with a tag of their own, are never
touched.

[`workflows/container.yml`](workflows/container.yml) publishes that image; see
[Publishing the image](#publishing-the-image).

## Security

The repository is public and the GPU runners are machines on a private network, so:

- The GPU jobs never run automatically for a pull request from a fork. They run for pushes to
  `main`, for manual runs, and for pull requests whose branch is in this repository. A fork's pull
  request gets only `Lint`; to test it on the GPUs, a maintainer reviews the code and pushes it to a
  branch of this repository.
- `ci.yml` has `permissions: contents: read`, never uses `pull_request_target`, and checks out
  with `persist-credentials: false`, so the build cannot push or read secrets.
- `container.yml`, the only workflow that can write, never runs for a pull request: only for a
  push to `main` that changes the image, a `v*` tag, or a manual run, all of which need write
  access to the repository. Only its jobs get `packages: write`, and only for this repository's
  package.
- Every action is pinned to a commit, and the base images (and the BuildKit image the publishing
  jobs build with) to a digest.

Because a skipped job counts as passing for branch protection, review a fork's pull request
yourself before merging it.

## Repository settings

- **Settings > Actions > General > Fork pull request workflows from outside collaborators**:
  "Require approval for all external contributors".
- **Settings > Actions > General > Workflow permissions**: "Read repository contents and packages
  permissions". `container.yml` asks for `packages: write` itself.
- **Settings > Branches > Branch protection rule for `main`** (or a ruleset): require a pull
  request, require these status checks to pass: `Lint`, `x86 GPU`, `Jetson Thor`, `Jetson Orin`; require branches to
  be up to date before merging; and do not allow bypassing these settings.

## Registering a runner

Register each machine as a repository runner: **Settings > Actions > Runners > New self-hosted
runner** shows the download and `config.sh` commands with a one-time token. Add the custom label
when `config.sh` asks, or pass `--labels`:

```bash
# x86_64 workstation with an NVIDIA GPU
./config.sh --url https://github.com/snknitheesh/holoscan-zozo-modules --token <token> \
  --name x86-gpu --labels gpu --unattended
# Jetson Thor (JetPack 7)
./config.sh --url https://github.com/snknitheesh/holoscan-zozo-modules --token <token> \
  --name jetson-thor --labels jetson,thor --unattended
# Jetson Orin (JetPack 7.2 or later, which brings CUDA 13 to Orin)
./config.sh --url https://github.com/snknitheesh/holoscan-zozo-modules --token <token> \
  --name jetson-orin --labels jetson,orin --unattended
```

The runner adds `self-hosted`, `linux` and `x64` or `arm64` itself. Install it as a service so it
starts at boot and restarts if it stops, under a user that is in the `docker` group:

```bash
sudo ./svc.sh install <runner-user>
sudo ./svc.sh start
```

A job waits in the queue until a runner with its labels is online.

### Install the job hook

[`runner/job-hook.sh`](runner/job-hook.sh) removes left-over CI containers. Install it outside the
runner's work folder, which jobs rewrite, and point both of the runner's job hooks at it:

```bash
cd <runner directory>
mkdir -p hooks
install -m 755 <checkout>/.github/runner/job-hook.sh hooks/job-hook.sh
echo "ACTIONS_RUNNER_HOOK_JOB_STARTED=$PWD/hooks/job-hook.sh" >> .env
echo "ACTIONS_RUNNER_HOOK_JOB_COMPLETED=$PWD/hooks/job-hook.sh" >> .env
```

The runner reads `.env` when it starts, so restart it, while it is idle (**Settings > Actions >
Runners** shows it as Idle), for the hooks to take effect. Every job's log then shows "A job
started hook has been configured by the self-hosted runner administrator" and, at the end, the
completed hook. With `svc.sh` that is `sudo ./svc.sh stop && sudo ./svc.sh start`. If the runner
runs as a systemd unit of your own with `KillSignal=SIGINT` and `KillMode=process`, `run.sh` does
not pass the signal on and the stop hangs until the timeout; send SIGINT to the
`bin/Runner.Listener` process instead, and the unit restarts it.

### x86_64 with an NVIDIA GPU

1. An NVIDIA driver for CUDA 13 (R580 or newer).
2. [Docker Engine](https://docs.docker.com/engine/install/ubuntu/) with the Buildx plugin
   (`docker buildx version` must work): `docker-buildx-plugin` with Docker's packages, or
   `sudo apt-get install docker-buildx` with Ubuntu's `docker.io`. The jobs fail without it.
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
2. Docker Engine with the Buildx plugin, as on x86_64 (Ubuntu's `docker.io` needs
   `sudo apt-get install docker-buildx`), and the `nvidia` runtime registered:

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

## The CI image

[`docker/Dockerfile`](docker/Dockerfile) builds the one image every lane uses, for `linux/amd64`
and `linux/arm64`:

- `nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu24.04`, pinned by its multi-arch index digest;
- the Holoscan SDK 5.0.0 packages from NVIDIA's `ubuntu2404` CUDA repository (`x86_64` or
  `sbsa`): `holoscan-core-cuda-13`, `libholoscan-visualization-cuda-13` and `-dev`, and
  `libholoscan-inference-cuda-13` and `-dev`, each checked against its SHA-256;
- TensorRT 10.14 for CUDA 13.0 (libraries, headers and `trtexec`), pinned;
- VPI 4.0.5 from NVIDIA's Jetson repository, release `r38.4`: `x86_64/noble` on amd64, `common`
  on arm64, whose PVA backend the `nvidia` runtime connects to the board's PVA; the repository
  key is accepted only with its pinned fingerprint;
- from PyPI, CMake 4.4.4, `holoscan-cli` 5.0.0 and the SDK's Python bindings
  (`holoscan-*-cu13` 5.0.0), installed from [`docker/requirements.txt`](docker/requirements.txt),
  which pins every package and dependency to the SHA-256 of its files. Regenerate it with the
  command at its top after changing [`docker/requirements.in`](docker/requirements.in).

It is not the official `nvcr.io/nvidia/clara-holoscan/holoscan:v5.0.0-cuda13` image because that
image's arm64 variant is the discrete-GPU (SBSA) build: CUDA 13.2 user-space libraries with a CUDA
13.2 data-center forward-compatibility driver in `/usr/local/cuda/compat`, which the image puts
first on the library path. JetPack 7 ships a CUDA 13.0 driver for the integrated GPU, and that
forward-compatibility driver is not made for it. The image also has no VPI. A CUDA 13.0 user space
runs on both JetPack 7 and a discrete-GPU driver for CUDA 13 (R580 or newer), so one image built
from the CUDA 13.0 base serves every lane.

## Publishing the image

[`workflows/container.yml`](workflows/container.yml) publishes the image as
`ghcr.io/snknitheesh/holoscan-zozo-modules`. It runs on a push to `main` that changes
`docker/Dockerfile` or the workflow, on every `v*` tag, and by hand (`workflow_dispatch`); never
for a pull request.

| Job | Runs on | What it does |
| --- | --- | --- |
| `Build linux/amd64` | The x86_64 runner | Builds the image natively and pushes it by digest, untagged |
| `Build linux/arm64` | The Jetson Thor runner | The same for arm64, without emulation |
| `Publish the multi-arch image` | GitHub-hosted `ubuntu-24.04` | Joins both digests into one multi-arch image and tags it |

The tags are `sha-<short commit>` always, `latest` for `main` and for `v*` tags, and the version
for a `v*` tag (`v1.2.3` gives `1.2.3`). The jobs log in with the workflow's own `GITHUB_TOKEN`;
no secret is needed.

The first push creates the package **private**. Make it public once: open the package from the
repository's or the profile's **Packages**, then **Package settings > Danger Zone > Change
visibility > Public**. The image's `org.opencontainers.image.source` label links the package to
this repository, which gives the repository's workflows write access to it.
