#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0
#
# Job hook for the self-hosted runners: the runner runs it before and after every job
# (ACTIONS_RUNNER_HOOK_JOB_STARTED and ACTIONS_RUNNER_HOOK_JOB_COMPLETED in its .env;
# see .github/RUNNERS.md). It removes every container with the CI label. A runner runs
# one job at a time, so any such container is left over from a job that died before
# its own cleanup ran: a cancelled or crashed job, or a killed docker CLI. Containers
# without the label are never touched.

set -uo pipefail

label=io.github.snknitheesh.holoscan-zozo-modules.ci
hook="holoscan-zozo-modules job hook"

if ! ids="$(docker ps --all --quiet --filter "label=${label}")"; then
  echo "${hook}: cannot reach Docker; no CI containers removed" >&2
  exit 0
fi
if [ -z "${ids}" ]; then
  echo "${hook}: no CI containers left over"
  exit 0
fi
# shellcheck disable=SC2086  # One container ID per word.
echo "${hook}: removing left-over CI container(s):" ${ids}
# shellcheck disable=SC2086
docker rm --force ${ids} > /dev/null || echo "${hook}: docker rm failed" >&2
exit 0
