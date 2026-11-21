#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0
#
# Makes the reordered test streams next to this script: the first 30 frames of the module's test
# video at 320x240, encoded by FFmpeg's libx264 and libx265 with two B-frames between I and P
# frames (no B-frame references), a keyframe every 15 frames with its parameter sets, and an
# access unit delimiter before every access unit. Each stream comes with a .order file: the
# display index of each access unit, in decode order. NVENC in this module never makes B-frames;
# these streams test the decoders on streams from other encoders.
#
#   tests/data/make_reordered_streams.sh   (needs a C++20 compiler and ffmpeg with both encoders)
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
module="$here/../.."
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

cat > "$work/dump.cpp" << 'CPP'
#include <cstdio>
#include <holoscan/codec/test_video.hpp>
int main() {
  for (unsigned i = 0; i < 30; ++i) {
    const auto frame = holoscan::codec::render_test_frame(320, 240, i);
    std::fwrite(frame.data(), 1, frame.size(), stdout);
  }
}
CPP
"${CXX:-c++}" -std=c++20 -O2 -I"$module/include" "$work/dump.cpp" "$module/src/test_video.cpp" \
  -o "$work/dump"

input=(-f rawvideo -pixel_format rgb24 -video_size 320x240 -framerate 30 -i -)
# BT.709 in limited range, as the module's decoder converts.
convert=(-vf scale=out_color_matrix=bt709:out_range=tv -pix_fmt yuv420p -colorspace bt709
  -color_primaries bt709 -color_trc bt709 -color_range tv)
x264=keyint=15:min-keyint=15:scenecut=0:bframes=2:b-adapt=0:b-pyramid=none:aud=1:repeat-headers=1
x265=keyint=15:min-keyint=15:scenecut=0:bframes=2:b-adapt=0:b-pyramid=0:open-gop=0:aud=1
x265+=:repeat-headers=1:log-level=error
"$work/dump" | ffmpeg -loglevel error -y "${input[@]}" "${convert[@]}" -c:v libx264 -qp 22 \
  -x264-params "$x264" "$work/h264.mkv"
"$work/dump" | ffmpeg -loglevel error -y "${input[@]}" "${convert[@]}" -c:v libx265 \
  -x265-params "$x265:qp=22" "$work/h265.mkv"

for codec in h264 h265; do
  if [ "$codec" = h264 ]; then format=h264; else format=hevc; fi
  ffmpeg -loglevel error -y -i "$work/$codec.mkv" -c copy -bsf:v "${format}_mp4toannexb" \
    -f "$format" "$here/reordered.$codec"
  # Packets in decode order; their pts in frames at 30 frames per second (milliseconds in MKV).
  ffprobe -loglevel error -select_streams v -show_entries packet=pts -of csv=p=0 \
    "$work/$codec.mkv" | awk '{ printf "%d\n", ($1 * 30 + 500) / 1000 }' > "$here/reordered.$codec.order"
done
