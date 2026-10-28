# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0
#
# The CI and development image for every lane: x86_64 with a discrete GPU, and
# Jetson Thor and Orin (JetPack 7.2, aarch64). One Dockerfile for linux/amd64 and
# linux/arm64: NVIDIA's CUDA 13.2 image, plus the Holoscan SDK 5.0.0 Debian
# packages and Python bindings, TensorRT for CUDA 13.2, VPI 4, the RealSense SDK
# (librealsense 2), CMake 4 and the Holoscan CLI. A CUDA 13.2 user space needs a driver for CUDA 13.2: R595 or newer
# with a discrete GPU, and JetPack 7.2 (L4T R39.2) on Jetson. See .github/RUNNERS.md
# for why this is not the official SDK image.
#
#   docker build --force-rm -f .github/docker/Dockerfile -t holoscan-modules-ci:local \
#     .github/docker
#
# Published as ghcr.io/snknitheesh/holoscan-zozo-modules. Run it with `--gpus all` on
# x86_64, and with `--runtime nvidia --privileged` and without DISPLAY on Jetson.

ARG BASE_IMAGE=nvcr.io/nvidia/cuda:13.2.1-devel-ubuntu24.04@sha256:44a9504c6dfb50b1241464241b02a93871928f373de6f5a644cf5fe9f080aa63
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.source="https://github.com/snknitheesh/holoscan-zozo-modules" \
      org.opencontainers.image.description="Build and test environment for Holoscan SDK 5.0 Modules: CUDA 13.2, Holoscan SDK 5.0.0, TensorRT, VPI 4, librealsense 2 and CMake 4, for x86_64 and Jetson Thor and Orin with JetPack 7.2." \
      org.opencontainers.image.licenses="Apache-2.0 AND LicenseRef-NVIDIA-Proprietary"

ARG DEBIAN_FRONTEND=noninteractive
# A failing command anywhere in a pipe fails the build.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
# Set by BuildKit to the platform being built, and unset by Docker's legacy builder.
# The packages below are picked from the image's own architecture, which must agree
# with it when it is set.
ARG TARGETARCH

# The base image's CUDA forward-compatibility driver (cuda-compat, in
# /usr/local/cuda/compat) goes: every lane has a driver for CUDA 13.2, so nothing
# needs it, and on JetPack 7.2 the NVIDIA Container Toolkit's cuda-compat hook
# (1.19) fails to read the 13.2 one and refuses to start the container at all.
RUN set -eux; \
    packages="$(dpkg-query -W -f '${Package}\n' 'cuda-compat-*')"; \
    test -n "${packages}"; \
    printf '%s\n' "${packages}" | xargs dpkg --purge; \
    if [ -n "$(find /usr/local -path '/usr/local/cuda*/compat/libcuda.so*')" ]; then \
      echo "a CUDA forward-compatibility driver is left in /usr/local/cuda*/compat" >&2; exit 1; \
    fi

# NVIDIA's Jetson package repository, release r39.2 (JetPack 7.2): its x86_64/noble
# component on amd64, for VPI; on arm64 its common component, for VPI and TensorRT,
# whose Jetson builds support every Jetson GPU (the CUDA repository's sbsa TensorRT
# does not support Orin's SM 87). The build accepts the repository's signing key only
# with the fingerprint below, and apt verifies every package against that key.
# The fingerprint's awk reads all of gpg's output: under pipefail, exiting at the first
# match could fail gpg, and the build, on EPIPE.
ARG JETSON_REPOSITORY_RELEASE=r39.2
ARG JETSON_REPOSITORY_FINGERPRINT=3C6D1FF3100C8C3ABB0869C0E6543461A9996195
RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    if [ -n "${TARGETARCH:-}" ] && [ "${TARGETARCH:-}" != "${arch}" ]; then \
      echo "TARGETARCH ${TARGETARCH:-} does not match the image's architecture ${arch}" >&2; exit 1; \
    fi; \
    case "${arch}" in \
      amd64) component=x86_64/noble ;; \
      arm64) component=common ;; \
      *) echo "unsupported architecture ${arch}" >&2; exit 1 ;; \
    esac; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl gnupg; \
    GNUPGHOME="$(mktemp -d)"; \
    export GNUPGHOME; \
    key="${GNUPGHOME}/nvidia-jetson.asc"; \
    curl -fsSL https://repo.download.nvidia.com/jetson/jetson-ota-public.asc -o "${key}"; \
    keys="$(gpg --show-keys --with-colons "${key}" | awk -F: '$1 == "pub"' | wc -l)"; \
    fingerprint="$(gpg --show-keys --with-colons "${key}" | awk -F: '$1 ~ /^fp/ && !n++ { print $10 }')"; \
    if [ "${keys}" != 1 ] || [ "${fingerprint}" != "${JETSON_REPOSITORY_FINGERPRINT}" ]; then \
      echo "unexpected Jetson repository signing key: ${keys} key(s), ${fingerprint}" >&2; exit 1; \
    fi; \
    gpg --dearmor < "${key}" > /usr/share/keyrings/nvidia-jetson.gpg; \
    rm -rf "${GNUPGHOME}"; \
    echo "deb [signed-by=/usr/share/keyrings/nvidia-jetson.gpg] https://repo.download.nvidia.com/jetson/${component} ${JETSON_REPOSITORY_RELEASE} main" \
      > /etc/apt/sources.list.d/nvidia-jetson.list; \
    rm -rf /var/lib/apt/lists/*

# Holoscan SDK 5.0.0 from NVIDIA's ubuntu2404 CUDA repository (x86_64 or sbsa), each
# package checked against its SHA-256 from that repository's index. The packages
# depend on CUDA 13, not on one minor release, and run on the CUDA 13.2 user space.
# TensorRT (for the inference component, with its headers and trtexec) is pinned per
# architecture to its newest CUDA 13.2 build: on amd64 from the CUDA repository, on
# arm64 from the Jetson repository above. VPI 4 is pinned too; on Jetson it links
# Jetson Linux's own libraries (libnvrm_*, libnvmedia, libnvbufsurface, libnvvic,
# libnvos, the EGL driver), which the nvidia runtime mounts from the board: there,
# building against VPI, not only running it, needs `--runtime nvidia`. It also needs
# the EGL loader (libegl1). apt verifies every package against the key of its
# repository.
ARG HOLOSCAN_REPOSITORY=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404
ARG HOLOSCAN_DEB_VERSION=5.0.0.8.0-1
ARG TENSORRT_VERSION_AMD64=10.16.1.11-1+cuda13.2
ARG TENSORRT_VERSION_ARM64=10.16.2.10-1+cuda13.2
ARG VPI_VERSION=4.1.4
# The packages are downloaded to, and installed from, a folder this step removes.
WORKDIR /tmp/holoscan-debs
RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    case "${arch}" in \
      amd64) \
        repository="${HOLOSCAN_REPOSITORY}/x86_64"; \
        tensorrt="${TENSORRT_VERSION_AMD64}"; \
        printf '%s  %s\n' \
          f5185060d6ccc917c97eebea9875735c0916a6839f79c3387187308372dcfadb "holoscan-core-cuda-13_${HOLOSCAN_DEB_VERSION}_amd64.deb" \
          bbbadbc434c53169fbd938b553122d086378a6da8d3ec550ec1dc85b3152774a "libholoscan-visualization-cuda-13_${HOLOSCAN_DEB_VERSION}_amd64.deb" \
          b3177d8019bdecf58a645c91e33268484c6510bce94ee11919bce2e3d01ad8b0 "libholoscan-visualization-dev-cuda-13_${HOLOSCAN_DEB_VERSION}_amd64.deb" \
          7f71878159e31fedacb13f52f571eb6250e95ea8781931c8716d100e95ac1e26 "libholoscan-inference-cuda-13_${HOLOSCAN_DEB_VERSION}_amd64.deb" \
          3de9704835f32a9f963c76b8e49100c6b0088be3a71b0ec5a66ed3c019cbc476 "libholoscan-inference-dev-cuda-13_${HOLOSCAN_DEB_VERSION}_amd64.deb" \
          > SHA256SUMS ;; \
      arm64) \
        repository="${HOLOSCAN_REPOSITORY}/sbsa"; \
        tensorrt="${TENSORRT_VERSION_ARM64}"; \
        printf '%s  %s\n' \
          b176cd45911294a40963f38ec7016e760fcc1956d44b4c30c276a4cf96e4c160 "holoscan-core-cuda-13_${HOLOSCAN_DEB_VERSION}_arm64.deb" \
          077eb3bea8727203f622c570b0146d3c262fcd2df180e1d7ef0cf62044bd126c "libholoscan-visualization-cuda-13_${HOLOSCAN_DEB_VERSION}_arm64.deb" \
          814ae0e51f1f4b8d9d78a6c58436cb0e0f0789b010c8d09e5e91f91931bf1783 "libholoscan-visualization-dev-cuda-13_${HOLOSCAN_DEB_VERSION}_arm64.deb" \
          77c076cf0e2672a6fec75b8652dd9fc25d0ffe1eda2e08bb1ea7f0870ce95a7c "libholoscan-inference-cuda-13_${HOLOSCAN_DEB_VERSION}_arm64.deb" \
          821ab9939bc7ff0c9cc8e798b6b9d08f27e4240ff2ffc94bd1c0ae5469d8f2cb "libholoscan-inference-dev-cuda-13_${HOLOSCAN_DEB_VERSION}_arm64.deb" \
          > SHA256SUMS ;; \
      *) echo "unsupported architecture ${arch}" >&2; exit 1 ;; \
    esac; \
    awk '{ print $2 }' SHA256SUMS | while read -r deb; do \
      curl -fsSLO "${repository}/${deb}"; \
    done; \
    sha256sum -c SHA256SUMS; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      ./*.deb \
      "libnvinfer10=${tensorrt}" \
      "libnvinfer-plugin10=${tensorrt}" \
      "libnvonnxparsers10=${tensorrt}" \
      "libnvinfer-dev=${tensorrt}" \
      "libnvinfer-headers-dev=${tensorrt}" \
      "libnvinfer-safe-headers-dev=${tensorrt}" \
      "libnvinfer-plugin-dev=${tensorrt}" \
      "libnvinfer-headers-plugin-dev=${tensorrt}" \
      "libnvonnxparsers-dev=${tensorrt}" \
      "libnvinfer-bin=${tensorrt}" \
      "libnvinfer-lean10=${tensorrt}" \
      "libnvinfer-vc-plugin10=${tensorrt}" \
      "libnvinfer-dispatch10=${tensorrt}" \
      "libnvvpi4=${VPI_VERSION}" "vpi4-dev=${VPI_VERSION}" libegl1 \
      git libvulkan-dev ninja-build python3 python3-pip; \
    rm -rf /tmp/holoscan-debs /var/lib/apt/lists/*
WORKDIR /

# The RealSense SDK (librealsense 2) from RealSense's own package repository, for
# modules with a RealSense camera backend: its library (librealsense2) and its
# headers, CMake package and pkg-config file (librealsense2-dev), pinned per
# architecture, whose builds have their own revisions, and pkg-config to read that
# file. Not its udev rules or kernel modules (librealsense2-udev-rules,
# librealsense2-dkms): a container has no camera to give permissions to and loads no
# kernel modules. The repository's key file holds its signing key and an older one;
# apt trusts only the signing key, accepted only with the fingerprint below.
ARG LIBREALSENSE_REPOSITORY=https://librealsense.realsenseai.com/Debian
ARG LIBREALSENSE_FINGERPRINT=5381411D24E659FB18195FA5FB0B24895113F120
ARG LIBREALSENSE_VERSION_AMD64=2.58.4-0~realsense.19927
ARG LIBREALSENSE_VERSION_ARM64=2.58.4-0~realsense.8908
RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    case "${arch}" in \
      amd64) librealsense="${LIBREALSENSE_VERSION_AMD64}" ;; \
      arm64) librealsense="${LIBREALSENSE_VERSION_ARM64}" ;; \
      *) echo "unsupported architecture ${arch}" >&2; exit 1 ;; \
    esac; \
    GNUPGHOME="$(mktemp -d)"; \
    export GNUPGHOME; \
    key="${GNUPGHOME}/librealsense.asc"; \
    curl -fsSL "${LIBREALSENSE_REPOSITORY}/librealsenseai.asc" -o "${key}"; \
    gpg --batch --quiet --import "${key}"; \
    fingerprints="$(gpg --batch --with-colons --list-keys | awk -F: '$1 == "pub" { primary = 1; next } $1 ~ /^fp/ && primary { print $10; primary = 0 }')"; \
    if ! printf '%s\n' "${fingerprints}" | grep -qx "${LIBREALSENSE_FINGERPRINT}"; then \
      echo "the RealSense repository's key file has no key ${LIBREALSENSE_FINGERPRINT}" >&2; exit 1; \
    fi; \
    gpg --batch --export "${LIBREALSENSE_FINGERPRINT}" > /usr/share/keyrings/librealsense.gpg; \
    rm -rf "${GNUPGHOME}"; \
    echo "deb [signed-by=/usr/share/keyrings/librealsense.gpg] ${LIBREALSENSE_REPOSITORY}/apt-repo noble main" \
      > /etc/apt/sources.list.d/librealsense.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      "librealsense2=${librealsense}" "librealsense2-dev=${librealsense}" pkgconf; \
    rm -rf /var/lib/apt/lists/*; \
    pkg-config --modversion realsense2

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

# Images built FROM this one get Docker's default shell back, not this file's pipefail.
SHELL ["/bin/sh", "-c"]
