# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0
#
# The CI and development image for every lane: x86_64 with a discrete GPU, and
# Jetson Thor and Orin (JetPack 7, aarch64). One Dockerfile for linux/amd64 and
# linux/arm64: NVIDIA's CUDA 13.0 image, plus the Holoscan SDK 5.0.0 Debian
# packages and Python bindings, TensorRT for CUDA 13.0, VPI 4, CMake 4 and the
# Holoscan CLI. A CUDA 13.0 user space runs on a discrete-GPU driver for CUDA 13
# (R580 or newer) and on JetPack 7's CUDA 13.0 driver alike. See .github/RUNNERS.md
# for why this is not the official SDK image.
#
#   docker build -f .github/docker/Dockerfile -t holoscan-modules-ci:local .github/docker
#
# Published as ghcr.io/snknitheesh/holoscan-zozo-modules. Run it with `--gpus all` on
# x86_64, and with `--runtime nvidia --privileged` and without DISPLAY on Jetson.

ARG BASE_IMAGE=nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.source="https://github.com/snknitheesh/holoscan-zozo-modules" \
      org.opencontainers.image.description="Build and test environment for Holoscan SDK 5.0 Modules: CUDA 13.0, Holoscan SDK 5.0.0, TensorRT, VPI 4 and CMake 4, for x86_64 and Jetson Thor and Orin." \
      org.opencontainers.image.licenses="Apache-2.0 AND LicenseRef-NVIDIA-Proprietary"

ARG DEBIAN_FRONTEND=noninteractive
# Set by BuildKit to the platform being built, and unset by Docker's legacy builder.
# The packages below are picked from the image's own architecture, which must agree
# with it when it is set.
ARG TARGETARCH

# Holoscan SDK 5.0.0 from NVIDIA's ubuntu2404 CUDA repository (x86_64 or sbsa), each
# package checked against its SHA-256 from that repository's index. TensorRT (for the
# inference component, with its headers and trtexec) is pinned to the newest CUDA 13.0
# build; apt verifies it, and the SDK's dependencies, against the CUDA repository key
# the base image trusts.
ARG HOLOSCAN_REPOSITORY=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404
ARG HOLOSCAN_DEB_VERSION=5.0.0.8.0-1
ARG TENSORRT_VERSION=10.14.1.48-1+cuda13.0
RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    if [ -n "${TARGETARCH:-}" ] && [ "${TARGETARCH:-}" != "${arch}" ]; then \
      echo "TARGETARCH ${TARGETARCH:-} does not match the image's architecture ${arch}" >&2; exit 1; \
    fi; \
    debs="$(mktemp -d)"; \
    cd "${debs}"; \
    case "${arch}" in \
      amd64) \
        repository="${HOLOSCAN_REPOSITORY}/x86_64"; \
        printf '%s  %s\n' \
          f5185060d6ccc917c97eebea9875735c0916a6839f79c3387187308372dcfadb "holoscan-core-cuda-13_${HOLOSCAN_DEB_VERSION}_amd64.deb" \
          bbbadbc434c53169fbd938b553122d086378a6da8d3ec550ec1dc85b3152774a "libholoscan-visualization-cuda-13_${HOLOSCAN_DEB_VERSION}_amd64.deb" \
          b3177d8019bdecf58a645c91e33268484c6510bce94ee11919bce2e3d01ad8b0 "libholoscan-visualization-dev-cuda-13_${HOLOSCAN_DEB_VERSION}_amd64.deb" \
          7f71878159e31fedacb13f52f571eb6250e95ea8781931c8716d100e95ac1e26 "libholoscan-inference-cuda-13_${HOLOSCAN_DEB_VERSION}_amd64.deb" \
          3de9704835f32a9f963c76b8e49100c6b0088be3a71b0ec5a66ed3c019cbc476 "libholoscan-inference-dev-cuda-13_${HOLOSCAN_DEB_VERSION}_amd64.deb" \
          > SHA256SUMS ;; \
      arm64) \
        repository="${HOLOSCAN_REPOSITORY}/sbsa"; \
        printf '%s  %s\n' \
          b176cd45911294a40963f38ec7016e760fcc1956d44b4c30c276a4cf96e4c160 "holoscan-core-cuda-13_${HOLOSCAN_DEB_VERSION}_arm64.deb" \
          077eb3bea8727203f622c570b0146d3c262fcd2df180e1d7ef0cf62044bd126c "libholoscan-visualization-cuda-13_${HOLOSCAN_DEB_VERSION}_arm64.deb" \
          814ae0e51f1f4b8d9d78a6c58436cb0e0f0789b010c8d09e5e91f91931bf1783 "libholoscan-visualization-dev-cuda-13_${HOLOSCAN_DEB_VERSION}_arm64.deb" \
          77c076cf0e2672a6fec75b8652dd9fc25d0ffe1eda2e08bb1ea7f0870ce95a7c "libholoscan-inference-cuda-13_${HOLOSCAN_DEB_VERSION}_arm64.deb" \
          821ab9939bc7ff0c9cc8e798b6b9d08f27e4240ff2ffc94bd1c0ae5469d8f2cb "libholoscan-inference-dev-cuda-13_${HOLOSCAN_DEB_VERSION}_arm64.deb" \
          > SHA256SUMS ;; \
      *) echo "unsupported architecture ${arch}" >&2; exit 1 ;; \
    esac; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl; \
    awk '{ print $2 }' SHA256SUMS | while read -r deb; do \
      curl -fsSLO "${repository}/${deb}"; \
    done; \
    sha256sum -c SHA256SUMS; \
    apt-get install -y --no-install-recommends \
      ./*.deb \
      "libnvinfer10=${TENSORRT_VERSION}" \
      "libnvinfer-plugin10=${TENSORRT_VERSION}" \
      "libnvonnxparsers10=${TENSORRT_VERSION}" \
      "libnvinfer-dev=${TENSORRT_VERSION}" \
      "libnvinfer-headers-dev=${TENSORRT_VERSION}" \
      "libnvinfer-plugin-dev=${TENSORRT_VERSION}" \
      "libnvinfer-headers-plugin-dev=${TENSORRT_VERSION}" \
      "libnvonnxparsers-dev=${TENSORRT_VERSION}" \
      "libnvinfer-bin=${TENSORRT_VERSION}" \
      "libnvinfer-lean10=${TENSORRT_VERSION}" \
      "libnvinfer-vc-plugin10=${TENSORRT_VERSION}" \
      "libnvinfer-dispatch10=${TENSORRT_VERSION}" \
      git gnupg libvulkan-dev ninja-build python3 python3-pip; \
    cd /; \
    rm -rf "${debs}" /var/lib/apt/lists/*

# VPI 4 from NVIDIA's Jetson package repository: its x86_64/noble component on
# amd64, and on arm64 its common component, whose PVA backend the nvidia runtime
# connects to the board's PVA. The build accepts the repository's signing key only
# with the fingerprint below, apt verifies every package against that key, and the
# packages are pinned. VPI also needs the EGL loader (libegl1). On Jetson it links
# Jetson Linux's own libraries (libnvrm_*, libnvmedia, libnvbufsurface, libnvvic,
# libnvos, the EGL driver), which the nvidia runtime mounts from the board: there,
# building against VPI, not only running it, needs `--runtime nvidia`.
ARG VPI_REPOSITORY_RELEASE=r38.4
ARG VPI_VERSION=4.0.5
ARG VPI_REPOSITORY_FINGERPRINT=3C6D1FF3100C8C3ABB0869C0E6543461A9996195
RUN set -eux; \
    case "$(dpkg --print-architecture)" in \
      amd64) component=x86_64/noble ;; \
      arm64) component=common ;; \
      *) echo "unsupported architecture" >&2; exit 1 ;; \
    esac; \
    export GNUPGHOME="$(mktemp -d)"; \
    key="${GNUPGHOME}/nvidia-jetson.asc"; \
    curl -fsSL https://repo.download.nvidia.com/jetson/jetson-ota-public.asc -o "${key}"; \
    keys="$(gpg --show-keys --with-colons "${key}" | awk -F: '$1 == "pub"' | wc -l)"; \
    fingerprint="$(gpg --show-keys --with-colons "${key}" | awk -F: '$1 ~ /^fp/ { print $10; exit }')"; \
    if [ "${keys}" != 1 ] || [ "${fingerprint}" != "${VPI_REPOSITORY_FINGERPRINT}" ]; then \
      echo "unexpected VPI repository signing key: ${keys} key(s), ${fingerprint}" >&2; exit 1; \
    fi; \
    gpg --dearmor < "${key}" > /usr/share/keyrings/nvidia-jetson.gpg; \
    rm -rf "${GNUPGHOME}"; \
    echo "deb [signed-by=/usr/share/keyrings/nvidia-jetson.gpg] https://repo.download.nvidia.com/jetson/${component} ${VPI_REPOSITORY_RELEASE} main" \
      > /etc/apt/sources.list.d/nvidia-vpi.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends libegl1 \
      "libnvvpi4=${VPI_VERSION}" "vpi4-dev=${VPI_VERSION}"; \
    rm -rf /var/lib/apt/lists/*

# From PyPI: CMake 4, the Holoscan CLI, and the SDK's Python bindings, which SDK 5.0
# ships only as wheels (holoscan-*-cu13). requirements.txt pins every package and
# its dependencies to a version and to the SHA-256 of each of its files, and pip
# refuses anything else; it is generated from requirements.in. `import holoscan`
# loads the driver's libcuda, so it works in a container with a GPU, not here.
COPY requirements.txt /tmp/holoscan-modules-ci/requirements.txt
RUN set -eux; \
    python3 -m pip install --no-cache-dir --break-system-packages --require-hashes \
      -r /tmp/holoscan-modules-ci/requirements.txt; \
    rm -rf /tmp/holoscan-modules-ci; \
    holoscan version; \
    cmake --version

# Holoviz needs the graphics and display capabilities, not only compute and utility.
ENV NVIDIA_DRIVER_CAPABILITIES=all
