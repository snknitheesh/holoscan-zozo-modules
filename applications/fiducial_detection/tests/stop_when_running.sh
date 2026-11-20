#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Starts the demo with the given arguments, sends it SIGTERM half a second after it says it
# is running (however long its start takes, up to two minutes), and exits with its exit status after
# printing its output.
#
#   stop_when_running.sh <log file> <demo> [arguments...]
#
# A shell starts background jobs with SIGINT ignored, so this sends SIGTERM.

log="$1"
shift
"$@" > "$log" 2>&1 &
pid=$!
tries=0
until grep -q ': running' "$log"; do
  tries=$((tries + 1))
  if [ "$tries" -gt 1200 ] || ! kill -0 "$pid" 2> /dev/null; then
    break
  fi
  sleep 0.1
done
sleep 0.5
kill -TERM "$pid" 2> /dev/null
wait "$pid"
status=$?
cat "$log"
exit "$status"
