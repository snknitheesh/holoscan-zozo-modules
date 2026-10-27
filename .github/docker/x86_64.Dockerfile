# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0
#
# CI and development image for x86_64: the Holoscan SDK 5.0.0 image (CUDA 13, which
# already ships CMake 4), plus NVIDIA VPI 4 and the Holoscan CLI.
#
#   docker build -f .github/docker/x86_64.Dockerfile -t holoscan-modules-ci:x86_64 .github/docker

ARG BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v5.0.0-cuda13@sha256:faa23a3a2479697fd64babcb27d28cf97f21d5da54d65c92225f8a9b0d99264a
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

# VPI 4 from NVIDIA's Jetson package repository, which also publishes it for x86_64.
# The build accepts the repository's signing key only with the fingerprint below,
# apt verifies every package against that key, and the packages are pinned.
ARG VPI_REPOSITORY_RELEASE=r38.4
ARG VPI_VERSION=4.0.5
ARG VPI_REPOSITORY_FINGERPRINT=3C6D1FF3100C8C3ABB0869C0E6543461A9996195
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl gnupg; \
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
    echo "deb [signed-by=/usr/share/keyrings/nvidia-jetson.gpg] https://repo.download.nvidia.com/jetson/x86_64/noble ${VPI_REPOSITORY_RELEASE} main" \
      > /etc/apt/sources.list.d/nvidia-vpi.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends "libnvvpi4=${VPI_VERSION}" "vpi4-dev=${VPI_VERSION}"; \
    rm -rf /var/lib/apt/lists/*

ARG HOLOSCAN_CLI_VERSION=5.0.0
RUN set -eux; \
    python3 -m pip install --no-cache-dir "holoscan-cli==${HOLOSCAN_CLI_VERSION}"; \
    holoscan version; \
    cmake_major="$(cmake --version | awk 'NR == 1 { split($3, v, "."); print v[1] }')"; \
    test "${cmake_major}" -ge 4
