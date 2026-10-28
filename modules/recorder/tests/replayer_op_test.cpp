// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// ReplayerOp's modes: real-time pacing by the recorded capture times, as fast
// as possible, looping, a frame range and a subset of the streams; and the
// recordings and configurations it refuses.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/recorder/format.hpp>
#include <holoscan/recorder/replayer_op.hpp>

#include "graph_support.hpp"
#include "test_support.hpp"

namespace holoscan::recorder::test {
namespace {

using std::chrono::milliseconds;

const holoscan::ConnectionOptions kLossless{.queue_policy = holoscan::QueuePolicy::kBlocking};

// Two tensor streams of `count` samples each: "a" every `interval`, from
// t0 = 5 s, and "b" half an interval after each "a". Frame ids are 1, 2, ...
std::filesystem::path write_recording(const std::string& name, std::uint64_t count,
                                      std::chrono::nanoseconds interval) {
  const std::filesystem::path folder = fresh_folder(name);
  RecordingWriter writer(folder, {{"a", StreamKind::kTensor}, {"b", StreamKind::kTensor}});
  for (std::uint64_t i = 0; i < count; ++i) {
    for (std::uint32_t s = 0; s < 2; ++s) {
      Record record;
      record.stream = s;
      record.sequence = i;
      record.timestamp_ns = 5'000'000'000LL + static_cast<std::int64_t>(i) * interval.count() +
                            s * interval.count() / 2;
      record.frame_id = i + 1;
      record.dtype = ElementType{.code = 1, .bits = 8, .lanes = 1};
      record.shape = {8};
      const std::vector<std::uint8_t> data = pattern_bytes(s, i, 8);
      writer.append(record, std::as_bytes(std::span(data)));
    }
  }
  writer.finish();
  return folder;
}

const std::vector<Stream> kTwoStreams{
    Stream{.name = "a", .kind = StreamKind::kTensor, .dtype = kU8, .shape = {8}},
    Stream{.name = "b", .kind = StreamKind::kTensor, .dtype = kU8, .shape = {8}},
};

struct Replayed {
  RunOutcome outcome;
  std::shared_ptr<Collected> collected;
  std::shared_ptr<ReplayerStats> stats;
};

Replayed replay(ReplayerConfig config, const std::vector<Stream>& streams, std::uint64_t expected) {
  holoscan::Graph graph{"replay"};
  Replayed result{.collected = std::make_shared<Collected>(),
                  .stats = std::make_shared<ReplayerStats>()};
  config.stats = result.stats;
  auto replayer = graph.op<ReplayerOp>("replayer", config);
  auto sink = graph.op<CollectSinkOp>(
      "sink", SinkConfig{.streams = streams, .expected = expected, .collected = result.collected});
  for (const Stream& stream : streams) {
    graph.add_flow(replayer->tensor(stream.name), sink->tensor(stream.name), kLossless);
  }
  holoscan::CompileOptions options;
  replayer->bind_device(options, "replayer");
  result.outcome = run_graph(graph, std::move(options));
  VERIFY_MSG(result.outcome.compiled && result.outcome.completed, result.outcome.diagnostics);
  return result;
}

// How far a sample may arrive from its time. Jetson runners share their cores
// with other work, so they get more room.
std::chrono::nanoseconds tolerance() {
  const char* platform = std::getenv("HOLOSCAN_CI_PLATFORM");
  return platform != nullptr && std::string(platform) == "jetson" ? milliseconds{20}
                                                                  : milliseconds{10};
}

// Each sample arrives when its recorded time comes: its capture time after the
// first one's, after the first arrival.
void realtime_pacing(ReplayMemory memory) {
  constexpr std::uint64_t kCount = 15;
  constexpr milliseconds kInterval{20};
  const std::filesystem::path folder = write_recording("realtime", kCount, kInterval);
  const Replayed replayed = replay(
      ReplayerConfig{
          .directory = folder.string(), .pacing = ReplayPacing::kRealtime, .memory = memory},
      kTwoStreams, 2 * kCount);
  std::vector<const Received*> all;
  for (const auto& [name, samples] : replayed.collected->streams) {
    for (const Received& sample : samples) all.push_back(&sample);
  }
  VERIFY(all.size() == 2 * kCount);
  if (all.empty()) return;
  const auto first = std::min_element(all.begin(), all.end(), [](const auto* x, const auto* y) {
    return x->capture_ns < y->capture_ns;
  });
  const std::int64_t t0 = (*first)->capture_ns;
  const auto arrival0 = (*first)->arrival;
  std::int64_t worst = 0;
  double total = 0;
  for (const Received* sample : all) {
    const std::int64_t due = sample->capture_ns - t0;
    const std::int64_t arrived =
        std::chrono::duration_cast<std::chrono::nanoseconds>(sample->arrival - arrival0).count();
    const std::int64_t error = std::llabs(arrived - due);
    worst = std::max(worst, error);
    total += static_cast<double>(error);
  }
  const double mean = total / static_cast<double>(all.size());
  std::cout << "  real-time pacing: worst error " << worst / 1000 << " us, mean "
            << static_cast<std::int64_t>(mean) / 1000 << " us, worst lateness "
            << replayed.stats->max_lateness_ns / 1000 << " us\n";
  VERIFY_MSG(worst <= tolerance().count(),
             "worst pacing error " + std::to_string(worst / 1000) + " us");
  VERIFY_MSG(mean <= static_cast<double>(tolerance().count()) / 2,
             "mean pacing error " + std::to_string(static_cast<std::int64_t>(mean) / 1000) + " us");
  // The recording is 290 ms long: real time takes that long.
  VERIFY(replayed.outcome.took >= milliseconds{280});
}

// As fast as possible: every sample, in order, far faster than real time.
void as_fast_as_possible() {
  constexpr std::uint64_t kCount = 30;
  const std::filesystem::path folder = write_recording("fast", kCount, milliseconds{100});
  const Replayed replayed =
      replay(ReplayerConfig{.directory = folder.string()}, kTwoStreams, 2 * kCount);
  VERIFY(replayed.stats->emitted == 2 * kCount);
  VERIFY(replayed.stats->finished);
  VERIFY(replayed.stats->passes == 1U);
  for (std::size_t s = 0; s < kTwoStreams.size(); ++s) {
    const std::vector<Received>& samples = replayed.collected->streams[kTwoStreams[s].name];
    VERIFY(samples.size() == kCount);
    for (std::uint64_t i = 0; i < samples.size(); ++i) {
      VERIFY(samples[i].frame_id == i + 1);
      VERIFY(samples[i].bytes == pattern_bytes(s, i, 8));
    }
  }
  // Real time would take 3 s.
  VERIFY_MSG(
      replayed.outcome.took < std::chrono::seconds{2},
      "took " +
          std::to_string(std::chrono::duration_cast<milliseconds>(replayed.outcome.took).count()) +
          " ms");
}

// Looping: each pass moves the capture times on by the recording's length plus
// one interval, and the frame ids on by the largest, so both keep increasing.
void loops() {
  constexpr std::uint64_t kCount = 3;
  constexpr milliseconds kInterval{10};
  const std::filesystem::path folder = write_recording("loop", kCount, kInterval);
  constexpr std::uint64_t kSamples = 8;  // Two passes and two thirds.
  const Replayed replayed =
      replay(ReplayerConfig{.directory = folder.string(), .loop = true, .streams = {"a"}},
             {kTwoStreams[0]}, kSamples);
  const std::vector<Received>& samples = replayed.collected->streams["a"];
  VERIFY(samples.size() >= kSamples);
  // a is at 0, 10 and 20 ms, b 5 ms later: a pass spans 20 + 10 ms.
  constexpr std::int64_t kSpan = 30'000'000;
  for (std::uint64_t n = 0; n < kSamples && n < samples.size(); ++n) {
    const std::uint64_t pass = n / kCount;
    const std::uint64_t i = n % kCount;
    VERIFY_MSG(samples[n].capture_ns == 5'000'000'000LL +
                                            static_cast<std::int64_t>(i) * 10'000'000 +
                                            static_cast<std::int64_t>(pass) * kSpan,
               "sample " + std::to_string(n) + ": capture time");
    VERIFY_MSG(samples[n].frame_id == i + 1 + pass * kCount,
               "sample " + std::to_string(n) + ": frame id");
    VERIFY(samples[n].bytes == pattern_bytes(0, i, 8));
  }
  VERIFY(replayed.stats->passes >= 2U);
  VERIFY(!replayed.stats->finished);
}

// A frame range of one stream: only its samples, only in the range.
void range_and_subset() {
  const std::filesystem::path folder = write_recording("range", 10, milliseconds{10});
  const Replayed replayed = replay(ReplayerConfig{.directory = folder.string(),
                                                  .first_frame = 3,
                                                  .end_frame = 7,
                                                  .streams = {"b"},
                                                  .memory = ReplayMemory::kDevice},
                                   {kTwoStreams[1]}, 4);
  const std::vector<Received>& samples = replayed.collected->streams["b"];
  VERIFY(samples.size() == 4U);
  for (std::uint64_t n = 0; n < samples.size(); ++n) {
    VERIFY(samples[n].frame_id == 4 + n);
    VERIFY(samples[n].device == kDLCUDA);
    VERIFY(samples[n].bytes == pattern_bytes(1, 3 + n, 8));
  }
  VERIFY(replayed.stats->emitted == 4U);
  VERIFY(replayed.collected->streams.count("a") == 0U);
}

void rejects() {
  const std::filesystem::path folder = write_recording("rejects", 3, milliseconds{10});
  const auto message = [](const ReplayerConfig& config) -> std::string {
    try {
      ReplayerOp op(config);
    } catch (const std::exception& error) {
      return error.what();
    }
    return "";
  };
  const auto contains = [](const std::string& text, const std::string& part) {
    const bool found = text.find(part) != std::string::npos;
    if (!found) std::cerr << "  expected \"" << part << "\" in \"" << text << "\"\n";
    return found;
  };
  VERIFY(contains(message(ReplayerConfig{}), "directory must not be empty"));
  VERIFY(
      contains(message(ReplayerConfig{.directory = (folder / "nope").string()}), "no such folder"));
  VERIFY(contains(message(ReplayerConfig{.directory = folder.string(), .streams = {"c"}}),
                  "no stream \"c\"; it has a, b"));
  VERIFY(contains(message(ReplayerConfig{.directory = folder.string(), .streams = {"a", "a"}}),
                  "twice"));
  VERIFY(contains(
      message(ReplayerConfig{.directory = folder.string(), .first_frame = 5, .end_frame = 5}),
      "first_frame must be below end_frame"));
  VERIFY(contains(message(ReplayerConfig{.directory = folder.string(), .first_frame = 3}),
                  "no samples"));
  VERIFY(contains(message(ReplayerConfig{.directory = folder.string(), .period = milliseconds{0}}),
                  "period"));
  // A recording that is not one fails as a RecordingError naming the file.
  std::ofstream(folder / "index.csv", std::ios::app)
      << "a,99,1,a.bin,0,800,1,uint8,100,,,,,,,,,,,,,\n";
  VERIFY(
      throws<RecordingError>([&] { ReplayerOp op(ReplayerConfig{.directory = folder.string()}); }));
  VERIFY(contains(message(ReplayerConfig{.directory = folder.string()}), "index.csv: line 8"));
  ReplayerOp op(
      ReplayerConfig{.directory = write_recording("ports", 2, milliseconds{10}).string()});
  VERIFY(op.stream_names() == (std::vector<std::string>{"a", "b"}));
  VERIFY(op.samples() == 4U);
  VERIFY(throws<std::out_of_range>([&] { (void)op.image("a"); }));
  VERIFY(throws<std::out_of_range>([&] { (void)op.tensor("c"); }));
}

}  // namespace
}  // namespace holoscan::recorder::test

int main() {
  using holoscan::recorder::test::TestCase;
  namespace t = holoscan::recorder::test;
  using holoscan::recorder::ReplayMemory;
  return t::run({
      TestCase{"real-time pacing, host memory", [] { t::realtime_pacing(ReplayMemory::kHost); }},
      TestCase{"real-time pacing, device memory",
               [] { t::realtime_pacing(ReplayMemory::kDevice); }},
      TestCase{"as fast as possible, lossless", [] { t::as_fast_as_possible(); }},
      TestCase{"loop", [] { t::loops(); }},
      TestCase{"frame range and stream subset", [] { t::range_and_subset(); }},
      TestCase{"rejects configurations and recordings", [] { t::rejects(); }},
  });
}
