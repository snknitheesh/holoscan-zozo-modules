#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0
#
# Job hook for the self-hosted runners: the runner runs it before and after every job
# (ACTIONS_RUNNER_HOOK_JOB_STARTED and ACTIONS_RUNNER_HOOK_JOB_COMPLETED in its .env;
# see .github/RUNNERS.md). It removes every container with the CI label: a runner runs
# one job at a time, so any such container is left over from a job that died before
# its own cleanup ran (a crashed runner, a killed docker CLI). It also removes stopped
# containers made from this repository's image or its build steps, which a failed
# legacy build leaves behind. A running container without the CI label is never
# touched, nor is any container without either label.

set -uo pipefail

label=io.github.snknitheesh.holoscan-zozo-modules.ci
source=org.opencontainers.image.source=https://github.com/snknitheesh/holoscan-zozo-modules
hook="holoscan-zozo-modules job hook"

if ! ci="$(docker ps --all --quiet --filter "label=${label}")" ||
  ! stopped="$(docker ps --all --quiet --filter "label=${source}" \
    --filter status=created --filter status=exited --filter status=dead)"; then
  echo "${hook}: cannot reach Docker; no containers removed" >&2
  exit 0
fi
# One container ID per line in each list; an array of them, without duplicates.
mapfile -t ids < <(printf '%s\n' "${ci}" "${stopped}" | sed '/^$/d' | sort -u)
if [ "${#ids[@]}" -eq 0 ]; then
  echo "${hook}: no CI or build containers left over"
  exit 0
fi
echo "${hook}: removing left-over CI or build container(s): ${ids[*]}"
docker rm --force "${ids[@]}" > /dev/null || echo "${hook}: docker rm failed" >&2
exit 0
