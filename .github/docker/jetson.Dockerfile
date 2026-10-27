# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0
#
# CI image for Jetson Thor (JetPack 7, aarch64). JetPack 7 runs the SBSA CUDA stack
# with a CUDA 13.0 driver, so the image is NVIDIA's CUDA 13.0 arm64 (SBSA) image plus
# the Holoscan SDK 5.0.0 arm64 Debian packages, VPI 4 from the Jetson repository
# (which drives the PVA), CMake 4 and the Holoscan CLI. See .github/RUNNERS.md for
# why this is not the official SDK image.
#
#   docker build -f .github/docker/jetson.Dockerfile -t holoscan-modules-ci:jetson .github/docker
#
# Run it with `--runtime nvidia --privileged` and without DISPLAY.

ARG BASE_IMAGE=nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

# Holoscan SDK 5.0.0 from NVIDIA's ubuntu2404/sbsa CUDA repository, each package
# checked against its SHA-256 from the repository index. TensorRT (for the inference
# component) is pinned to the newest CUDA 13.0 build.
ARG HOLOSCAN_REPOSITORY=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa
ARG HOLOSCAN_DEB_VERSION=5.0.0.8.0-1
ARG TENSORRT_VERSION=10.14.1.48-1+cuda13.0
RUN set -eux; \
    test "$(dpkg --print-architecture)" = arm64; \
    debs="$(mktemp -d)"; \
    cd "${debs}"; \
    printf '%s  %s\n' \
      b176cd45911294a40963f38ec7016e760fcc1956d44b4c30c276a4cf96e4c160 "holoscan-core-cuda-13_${HOLOSCAN_DEB_VERSION}_arm64.deb" \
      077eb3bea8727203f622c570b0146d3c262fcd2df180e1d7ef0cf62044bd126c "libholoscan-visualization-cuda-13_${HOLOSCAN_DEB_VERSION}_arm64.deb" \
      814ae0e51f1f4b8d9d78a6c58436cb0e0f0789b010c8d09e5e91f91931bf1783 "libholoscan-visualization-dev-cuda-13_${HOLOSCAN_DEB_VERSION}_arm64.deb" \
      77c076cf0e2672a6fec75b8652dd9fc25d0ffe1eda2e08bb1ea7f0870ce95a7c "libholoscan-inference-cuda-13_${HOLOSCAN_DEB_VERSION}_arm64.deb" \
      821ab9939bc7ff0c9cc8e798b6b9d08f27e4240ff2ffc94bd1c0ae5469d8f2cb "libholoscan-inference-dev-cuda-13_${HOLOSCAN_DEB_VERSION}_arm64.deb" \
      > SHA256SUMS; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl; \
    awk '{ print $2 }' SHA256SUMS | while read -r deb; do \
      curl -fsSLO "${HOLOSCAN_REPOSITORY}/${deb}"; \
    done; \
    sha256sum -c SHA256SUMS; \
    apt-get install -y --no-install-recommends \
      ./*.deb \
      "libnvinfer10=${TENSORRT_VERSION}" \
      "libnvinfer-plugin10=${TENSORRT_VERSION}" \
      "libnvonnxparsers10=${TENSORRT_VERSION}" \
      git gnupg libvulkan-dev ninja-build python3 python3-pip; \
    cd /; \
    rm -rf "${debs}" /var/lib/apt/lists/*

# VPI 4 from NVIDIA's Jetson package repository. The build accepts the repository's
# signing key only with the fingerprint below, apt verifies every package against
# that key, and the packages are pinned. VPI also needs the EGL loader (libegl1) from
# the image, and Jetson Linux's own libraries (libnvrm_*, libnvmedia, libnvbufsurface,
# libnvvic, libnvos, the EGL driver), which the nvidia runtime mounts from the board:
# building against VPI, not only running it, needs `--runtime nvidia`.
ARG VPI_REPOSITORY_RELEASE=r38.4
ARG VPI_VERSION=4.0.5
ARG VPI_REPOSITORY_FINGERPRINT=3C6D1FF3100C8C3ABB0869C0E6543461A9996195
RUN set -eux; \
    apt-get update; \
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
    echo "deb [signed-by=/usr/share/keyrings/nvidia-jetson.gpg] https://repo.download.nvidia.com/jetson/common ${VPI_REPOSITORY_RELEASE} main" \
      > /etc/apt/sources.list.d/nvidia-vpi.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends libegl1 \
      "libnvvpi4=${VPI_VERSION}" "vpi4-dev=${VPI_VERSION}"; \
    rm -rf /var/lib/apt/lists/*

ARG HOLOSCAN_CLI_VERSION=5.0.0
ARG CMAKE_VERSION=4.4.4
RUN set -eux; \
    python3 -m pip install --no-cache-dir --break-system-packages \
      "holoscan-cli==${HOLOSCAN_CLI_VERSION}" "cmake==${CMAKE_VERSION}"; \
    holoscan version; \
    cmake --version
