// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Recorder: the demo of holoscan-recorder. Three modes:
//
//   ci      A synthetic source (RGB8 and MONO16 from the GPU, MONO8 and NV12
//           from the host, and a tensor) is recorded by RecorderOp, then
//           replayed by ReplayerOp into host and into device memory, and every
//           sample is checked against what was sent, byte for byte, with its
//           capture time and frame id. Passes only if all of it matches.
//   replay  Plays a recording (or a stereo camera recording) into a HoloViz
//           window, or headless into a counter.
//   bench   Records a synthetic camera at a given rate and size through
//           kDropOldest edges, as a live camera would be, and reports what the
//           disk took and what was dropped.

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/recorder/format.hpp>
#include <holoscan/recorder/recorder_op.hpp>
#include <holoscan/recorder/replayer_op.hpp>
#include <holoscan/time/realtime_clock.hpp>
#if RECORDER_ENABLE_VISUALIZATION
#include <holoscan/visualization/holoviz_op.hpp>
#endif

#include "demo_ops.hpp"

namespace {

namespace rec = holoscan::recorder;
namespace app = recorder_app;

constexpr std::string_view kName = "recorder";

enum class Mode { kCi, kReplay, kBench };
enum class Display { kAuto, kOn, kOff };

struct Options {
  Mode mode{Mode::kCi};
  std::filesystem::path recording;
  std::uint64_t frames{30};
  int width{320};
  int height{240};
  std::string encoding{"rgb8"};
  double fps{30};
  bool device{true};
  std::optional<std::string> stream;
  rec::ReplayPacing pacing{rec::ReplayPacing::kRealtime};
  bool loop{false};
  std::uint64_t count{0};
  Display display{Display::kAuto};
  bool compile_only{false};
  std::uint64_t queue_mb{512};
  bool keep{false};
  std::uint32_t timeout_s{120};
  bool help{false};
};

// Every sample, in order: a full queue holds back its producer.
holoscan::ConnectionOptions lossless() {
  return {.queue_policy = holoscan::QueuePolicy::kBlocking};
}
// A live edge: only the newest sample waits, and the producer is never held back.
holoscan::ConnectionOptions newest() {
  return {.queue_policy = holoscan::QueuePolicy::kDropOldest, .queue_depth = 1U};
}

void usage(std::ostream& out) {
  out << "Usage: recorder [--mode ci|replay|bench] [options]\n"
         "  --recording DIR       The recording's folder (ci: ./recording, bench:\n"
         "                        ./bench_recording; replay: required)\n"
         "  --frames N            Frames to record (ci: 30, bench: 300)\n"
         "  --width W --height H  Frame size (ci: 320x240, bench: 1280x720)\n"
         "  --encoding rgb8|mono8 bench: the camera's pixel format (rgb8)\n"
         "  --fps F               bench: the camera's rate (30)\n"
         "  --memory host|device  bench: the camera's memory; replay: the outputs' (device)\n"
         "  --stream NAME         replay: the image stream to show (rgb, else the first)\n"
         "  --pacing realtime|fast  replay: by the recorded times, or as fast as possible\n"
         "  --loop true|false     replay: start over at the end (false)\n"
         "  --count N             replay: stop after N frames (0: at the end)\n"
         "  --display auto|true|false  replay: a HoloViz window (auto: when DISPLAY is set)\n"
         "  --compile-only true|false  Compile the graph, print the plan's verdict, and stop\n"
         "  --queue-mb N          The recorder's queue of pinned memory, in MiB (512)\n"
         "  --keep true|false     bench: keep the recording (false)\n"
         "  --timeout-s N         Stop a run that has not ended after N seconds, and fail (120;\n"
         "                        0: never; a replay with a display never times out)\n";
}

bool parse_bool(std::string_view value, std::string_view option) {
  if (value == "true") return true;
  if (value == "false") return false;
  throw std::invalid_argument(std::string(option) + " takes true or false");
}

template <typename T>
T parse_number(std::string_view value, std::string_view option) {
  T number{};
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), number);
  if (value.empty() || error != std::errc() || end != value.data() + value.size()) {
    throw std::invalid_argument(std::string(option) + ": not a number: " + std::string(value));
  }
  return number;
}

Options parse_args(int argc, char** argv) {
  Options opts;
  std::optional<std::uint64_t> frames;
  std::optional<int> width;
  std::optional<int> height;
  std::optional<bool> device;
  for (int i = 1; i < argc; ++i) {
    const std::string_view option = argv[i];
    if (option == "--help" || option == "-h") {
      opts.help = true;
      return opts;
    }
    if (i + 1 >= argc) throw std::invalid_argument(std::string(option) + " needs a value");
    const std::string_view value = argv[++i];
    if (option == "--mode") {
      if (value == "ci") {
        opts.mode = Mode::kCi;
      } else if (value == "replay") {
        opts.mode = Mode::kReplay;
      } else if (value == "bench") {
        opts.mode = Mode::kBench;
      } else {
        throw std::invalid_argument("--mode takes ci, replay or bench");
      }
    } else if (option == "--recording") {
      opts.recording = std::string(value);
    } else if (option == "--frames") {
      frames = parse_number<std::uint64_t>(value, option);
    } else if (option == "--width") {
      width = parse_number<int>(value, option);
    } else if (option == "--height") {
      height = parse_number<int>(value, option);
    } else if (option == "--encoding") {
      if (value != "rgb8" && value != "mono8") {
        throw std::invalid_argument("--encoding takes rgb8 or mono8");
      }
      opts.encoding = std::string(value);
    } else if (option == "--fps") {
      opts.fps = parse_number<double>(value, option);
    } else if (option == "--memory") {
      if (value != "host" && value != "device") {
        throw std::invalid_argument("--memory takes host or device");
      }
      device = value == "device";
    } else if (option == "--stream") {
      opts.stream = std::string(value);
    } else if (option == "--pacing") {
      if (value == "realtime") {
        opts.pacing = rec::ReplayPacing::kRealtime;
      } else if (value == "fast") {
        opts.pacing = rec::ReplayPacing::kAsFastAsPossible;
      } else {
        throw std::invalid_argument("--pacing takes realtime or fast");
      }
    } else if (option == "--loop") {
      opts.loop = parse_bool(value, option);
    } else if (option == "--count") {
      opts.count = parse_number<std::uint64_t>(value, option);
    } else if (option == "--display") {
      opts.display = value == "auto"             ? Display::kAuto
                     : parse_bool(value, option) ? Display::kOn
                                                 : Display::kOff;
    } else if (option == "--compile-only") {
      opts.compile_only = parse_bool(value, option);
    } else if (option == "--queue-mb") {
      opts.queue_mb = parse_number<std::uint64_t>(value, option);
    } else if (option == "--keep") {
      opts.keep = parse_bool(value, option);
    } else if (option == "--timeout-s") {
      opts.timeout_s = parse_number<std::uint32_t>(value, option);
    } else {
      throw std::invalid_argument("unknown option " + std::string(option));
    }
  }
  const bool bench = opts.mode == Mode::kBench;
  opts.frames = frames.value_or(bench ? 300U : 30U);
  opts.width = width.value_or(bench ? 1280 : 320);
  opts.height = height.value_or(bench ? 720 : 240);
  opts.device = device.value_or(true);
  if (opts.recording.empty()) {
    if (opts.mode == Mode::kReplay) throw std::invalid_argument("--mode replay needs --recording");
    opts.recording = bench ? "bench_recording" : "recording";
  }
  if (opts.frames < 1 || opts.frames > 1'000'000) {
    throw std::invalid_argument("--frames must be 1..1000000");
  }
  if (opts.width < 2 || opts.height < 2 || opts.width > 8192 || opts.height > 8192 ||
      opts.width % 2 != 0 || opts.height % 2 != 0) {
    throw std::invalid_argument("--width and --height must be even, 2..8192");
  }
  if (!(opts.fps >= 1 && opts.fps <= 1000)) {
    throw std::invalid_argument("--fps must be 1..1000");
  }
  if (opts.queue_mb < 1) throw std::invalid_argument("--queue-mb must be positive");
  if (opts.display == Display::kAuto) {
    const char* display = std::getenv("DISPLAY");
    opts.display = display != nullptr && *display != '\0' && RECORDER_ENABLE_VISUALIZATION
                       ? Display::kOn
                       : Display::kOff;
  }
  if (opts.display == Display::kOn && !RECORDER_ENABLE_VISUALIZATION) {
    throw std::invalid_argument("--display needs a build with visualization enabled");
  }
  return opts;
}

struct Outcome {
  bool ok{};
  std::string diagnostics;
};

// Compiles `graph`; with --compile-only prints the verdict and runs nothing.
// A replay with a display runs until its window closes or a signal stops it.
// Every other run is caller-managed for signals (ci runs three graphs in one
// process) and stopped, and failed, after --timeout-s.
Outcome run(holoscan::Graph& graph, holoscan::CompileOptions options, const Options& opts) {
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (!plan.diagnostics().empty()) {
    std::cerr << holoscan::format_diagnostics(plan.diagnostics(), plan.graph()) << '\n';
  }
  if (opts.compile_only) {
    std::cout << kName << ": plan=" << (plan.ok() ? "OK" : "REJECTED") << '\n';
    return Outcome{.ok = plan.ok()};
  }
  if (!plan.ok()) return Outcome{};
  if (opts.mode == Mode::kReplay && opts.display == Display::kOn) {
    return Outcome{.ok = holoscan::run(plan) == 0};
  }
  holoscan::RunSession session = holoscan::run_async(plan);
  bool timed_out = false;
  if (opts.timeout_s != 0 && !session.wait_for(std::chrono::seconds{opts.timeout_s})) {
    timed_out = true;
    session.request_stop();
  }
  session.wait();
  Outcome outcome;
  const holoscan::RunTermination termination = session.termination();
  outcome.ok = !timed_out && session.diagnostics().empty() &&
               (termination == holoscan::RunTermination::kCompleted ||
                termination == holoscan::RunTermination::kRequestedCompleted);
  if (!session.diagnostics().empty()) {
    outcome.diagnostics = holoscan::format_diagnostics(session.diagnostics(), plan.graph());
    std::cerr << outcome.diagnostics << '\n';
  }
  if (timed_out) {
    std::cout << kName << ": a run did not end in " << opts.timeout_s << " s; stopped it\n";
  }
  return outcome;
}

double mib(std::uint64_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0); }

// ci: record, then replay into host and into device memory, checking everything.
int ci(const Options& opts) {
  const std::vector<app::Stream> streams = app::ci_streams(opts.width, opts.height);
  auto stats = std::make_shared<rec::RecorderStats>();
  {
    holoscan::Graph graph{"record"};
    auto source = graph.op<app::SyntheticSourceOp>(
        "source", app::SourceConfig{.streams = streams, .frames = opts.frames});
    std::vector<rec::RecorderStream> inputs;
    for (const app::Stream& stream : streams) inputs.push_back({stream.name, stream.kind});
    auto recorder = graph.op<rec::RecorderOp>(
        "recorder",
        rec::RecorderConfig{
            .directory = opts.recording.string(),
            .streams = inputs,
            .overwrite = true,
            .calibration = {{"synthetic", R"({"width": )" + std::to_string(opts.width) + "}"}},
            .metadata = {{"source", "recorder demo, ci mode"}},
            .queue_bytes = opts.queue_mb << 20U,
            .stats = stats});
    holoscan::CompileOptions options;
    for (const app::Stream& stream : streams) {
      // The ground-truth run is lossless: every frame is recorded.
      if (stream.kind == rec::StreamKind::kImage) {
        graph.add_flow(source->image(stream.name), recorder->image(stream.name), lossless());
      } else {
        graph.add_flow(source->tensor(stream.name), recorder->tensor(stream.name), lossless());
      }
      if (stream.device) {
        options.deployment.bind_tensor_output_device(
            holoscan::TensorOutputDevicePlacement{.operator_path = "source",
                                                  .output_port = stream.name,
                                                  .device = holoscan::DeviceId{0}});
      }
    }
    if (!run(graph, std::move(options), opts).ok) {
      std::cout << kName << ": the recording run failed\n" << kName << ": verdict=FAIL\n";
      return 1;
    }
  }
  if (opts.compile_only) return 0;
  const std::uint64_t samples = opts.frames * streams.size();
  std::cout << kName << ": recorded=" << stats->recorded << "/" << samples
            << " dropped=" << stats->dropped << " bytes=" << stats->bytes << " to "
            << opts.recording.string() << '\n';
  bool passed = stats->recorded == samples && stats->dropped == 0U;

  for (const bool device : {false, true}) {
    auto verified = std::make_shared<app::VerifyStats>();
    holoscan::Graph graph{"replay"};
    auto replayer = graph.op<rec::ReplayerOp>(
        "replayer", rec::ReplayerConfig{
                        .directory = opts.recording.string(),
                        .memory = device ? rec::ReplayMemory::kDevice : rec::ReplayMemory::kHost});
    auto verify = graph.op<app::VerifySinkOp>(
        "verify", app::VerifyConfig{.streams = streams, .frames = opts.frames, .stats = verified});
    for (const app::Stream& stream : streams) {
      if (stream.kind == rec::StreamKind::kImage) {
        graph.add_flow(replayer->image(stream.name), verify->image(stream.name), lossless());
      } else {
        graph.add_flow(replayer->tensor(stream.name), verify->tensor(stream.name), lossless());
      }
    }
    holoscan::CompileOptions options;
    replayer->bind_device(options, "replayer");
    const bool ran = run(graph, std::move(options), opts).ok;
    std::cout << kName << ": replayed to " << (device ? "device" : "host")
              << ": samples=" << verified->samples << "/" << samples
              << " mismatches=" << verified->mismatches << '\n';
    passed = passed && ran && verified->samples == samples && verified->mismatches == 0U;
  }
  std::cout << kName << ": verdict=" << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}

#if RECORDER_ENABLE_VISUALIZATION
std::optional<holoscan::visualization::ImageFormat> holoviz_format(const std::string& encoding) {
  using holoscan::visualization::ImageFormat;
  if (encoding == "MONO8") return ImageFormat::kR8Unorm;
  if (encoding == "MONO16") return ImageFormat::kR16Unorm;
  if (encoding == "RGB8") return ImageFormat::kR8G8B8Unorm;
  if (encoding == "RGBA8") return ImageFormat::kR8G8B8A8Unorm;
  return std::nullopt;
}
#endif

// replay: one image stream of a recording, into HoloViz or a counter.
int replay(const Options& opts) {
  const rec::RecordingReader reader(opts.recording);
  std::string stream;
  if (opts.stream) {
    stream = *opts.stream;
  } else if (reader.find_stream("rgb")) {
    stream = "rgb";
  } else {
    for (const rec::StreamInfo& info : reader.streams()) {
      if (info.kind == rec::StreamKind::kImage) {
        stream = info.name;
        break;
      }
    }
  }
  const std::optional<std::uint32_t> index = reader.find_stream(stream);
  if (!index || reader.streams()[*index].kind != rec::StreamKind::kImage) {
    throw std::invalid_argument("the recording has no image stream \"" + stream + "\"");
  }
  const rec::StreamInfo& info = reader.streams()[*index];
  if (opts.loop && opts.count == 0 && opts.display == Display::kOff) {
    throw std::invalid_argument("--loop true without a display needs --count");
  }
  const std::uint64_t count = opts.count != 0 ? opts.count : opts.loop ? 0 : info.records;
  // The first image's size and encoding, for the window.
  const rec::Record* first = nullptr;
  for (const rec::Record& record : reader.records()) {
    if (record.stream == *index) {
      first = &record;
      break;
    }
  }
  if (first == nullptr) throw std::invalid_argument("stream \"" + stream + "\" has no images");

  holoscan::Graph graph{"replay"};
  auto replay_stats = std::make_shared<rec::ReplayerStats>();
  auto replayer = graph.op<rec::ReplayerOp>(
      "replayer", rec::ReplayerConfig{
                      .directory = opts.recording.string(),
                      .pacing = opts.pacing,
                      .loop = opts.loop,
                      .streams = {stream},
                      .memory = opts.device ? rec::ReplayMemory::kDevice : rec::ReplayMemory::kHost,
                      .stats = replay_stats});
  auto counted = std::make_shared<app::CountStats>();
  auto counter = graph.op<app::CountSinkOp>("counter", count, counted);
  // Real time: the counter and the window take the newest image; fast: every one.
  const holoscan::ConnectionOptions edge =
      opts.pacing == rec::ReplayPacing::kRealtime ? newest() : lossless();
  graph.add_flow(replayer->image(stream), counter->image, edge);
#if RECORDER_ENABLE_VISUALIZATION
  if (opts.display == Display::kOn) {
    namespace viz = holoscan::visualization;
    const auto format = holoviz_format(first->image->encoding);
    if (!format) {
      throw std::invalid_argument("HoloViz cannot show " + first->image->encoding +
                                  " images; pick a MONO8, MONO16, RGB8 or RGBA8 stream");
    }
    const auto width = static_cast<std::uint32_t>(first->image->width);
    const auto height = static_cast<std::uint32_t>(first->image->height);
    viz::LayerStack layers;
    layers.add_image_layer(
        stream, viz::ImageLayerDesc{.format = *format, .max_width = width, .max_height = height});
    auto visualizer = graph.op<viz::HolovizOp>(
        "holoviz", std::move(layers),
        viz::WindowSinkDesc{
            .title = "Recorder: " + opts.recording.filename().string() + " " + stream,
            .width = width,
            .height = height});
    graph.add_flow(replayer->image(stream), visualizer->image_layer_input(0), newest());
  }
#endif
  holoscan::CompileOptions options;
  replayer->bind_device(options, "replayer");
  const auto started = std::chrono::steady_clock::now();
  const bool ran = run(graph, std::move(options), opts).ok;
  if (opts.compile_only) return ran ? 0 : 1;
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  const double recorded_span =
      static_cast<double>(counted->last_capture_ns - counted->first_capture_ns) * 1e-9;
  std::cout << kName << ": replayed " << stream << " from " << opts.recording.string()
            << ": frames=" << counted->frames << (count != 0 ? "/" + std::to_string(count) : "")
            << " emitted=" << replay_stats->emitted << std::fixed << std::setprecision(3)
            << " recorded_span=" << recorded_span << " s run=" << seconds
            << " s max_lateness=" << static_cast<double>(replay_stats->max_lateness_ns) * 1e-6
            << " ms\n";
  const bool passed = ran && (count == 0 || counted->frames >= count);
  std::cout << kName << ": verdict=" << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}

// bench: a live camera, recorded; what the disk took, and what was dropped.
int bench(const Options& opts) {
  const app::Stream stream = app::bench_stream(opts.encoding, opts.width, opts.height, opts.device);
  auto source_stats = std::make_shared<app::SourceStats>();
  auto stats = std::make_shared<rec::RecorderStats>();
  holoscan::Graph graph{"bench"};
  auto source = graph.op<app::SyntheticSourceOp>(
      "source", app::SourceConfig{
                    .streams = {stream},
                    .frames = opts.frames,
                    .period = std::chrono::nanoseconds(static_cast<std::int64_t>(1e9 / opts.fps)),
                    .live = true,
                    .reuse_pattern = true,
                    .stats = source_stats});
  auto recorder = graph.op<rec::RecorderOp>(
      "recorder", rec::RecorderConfig{.directory = opts.recording.string(),
                                      .streams = {{stream.name, rec::StreamKind::kImage}},
                                      .overwrite = true,
                                      .queue_bytes = opts.queue_mb << 20U,
                                      .stats = stats});
  // A live camera's edge: the recorder takes the newest frame.
  graph.add_flow(source->image(stream.name), recorder->image(stream.name), newest());
  holoscan::CompileOptions options;
  if (opts.device) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "source", .output_port = stream.name, .device = holoscan::DeviceId{0}});
  }
  const auto started = std::chrono::steady_clock::now();
  const bool ran = run(graph, std::move(options), opts).ok;
  if (opts.compile_only) return ran ? 0 : 1;
  const double total =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  const double sending = static_cast<double>(source_stats->last_ns - source_stats->first_ns) * 1e-9;
  const std::uint64_t frame_bytes = stream.bytes();
  const std::uint64_t edge_dropped = source_stats->sent - stats->received;
  std::cout << std::fixed << std::setprecision(1) << kName << ": bench " << opts.width << "x"
            << opts.height << " " << opts.encoding << " " << (opts.device ? "device" : "host")
            << " at " << opts.fps << " fps: frames=" << source_stats->sent << "/" << opts.frames
            << " offered=" << mib(frame_bytes * source_stats->sent) / std::max(sending, 1e-9)
            << " MiB/s received=" << stats->received << " recorded=" << stats->recorded
            << " dropped=" << stats->dropped << " edge_dropped=" << edge_dropped
            << " written=" << mib(stats->bytes) / std::max(total, 1e-9)
            << " MiB/s (with startup and drain) max_queued=" << stats->max_queued << '\n';
  if (!opts.keep) std::filesystem::remove_all(opts.recording);
  const bool passed = ran && stats->recorded + stats->dropped == stats->received;
  std::cout << kName << ": verdict=" << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) try {
  const Options opts = parse_args(argc, argv);
  if (opts.help) {
    usage(std::cout);
    return 0;
  }
  switch (opts.mode) {
    case Mode::kCi:
      return ci(opts);
    case Mode::kReplay:
      return replay(opts);
    case Mode::kBench:
      return bench(opts);
  }
  return 1;
} catch (const std::invalid_argument& error) {
  std::cerr << "error: " << error.what() << '\n';
  usage(std::cerr);
  return 2;
} catch (const rec::RecordingError& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 2;
} catch (const std::exception& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 1;
}
