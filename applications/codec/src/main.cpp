// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Codec: hardware video encoding and decoding in a Holoscan graph. A synthetic camera's frames go
// through VideoEncoderOp and VideoDecoderOp; a reporter measures each decoded frame against the
// one sent (PSNR), the bitrate and the order, and HoloViz shows the original and the decoded
// frame side by side, in a window or offscreen.

#include <algorithm>
#include <atomic>
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
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/codec/video_codec.hpp>
#include <holoscan/codec/video_decoder_op.hpp>
#include <holoscan/codec/video_encoder_op.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/visualization/holoviz_op.hpp>

#include "benchmark.hpp"
#include "report_op.hpp"
#include "test_source_op.hpp"

namespace {

using std::chrono_literals::operator""ms;
namespace app = codec_app;
namespace hc = holoscan::codec;
namespace viz = holoscan::visualization;

constexpr std::string_view kAppName = "codec_demo";

enum class Display { kAuto, kWindow, kOffscreen, kNone };

struct Options {
  hc::EncoderSettings settings;
  hc::Backend decoder_backend{hc::Backend::kAuto};
  int width{1280};
  int height{720};
  std::optional<std::uint32_t> count;
  std::uint32_t period_ms{33};
  Display display{Display::kAuto};
  std::filesystem::path output;
  std::filesystem::path render_file;
  double min_psnr{0.0};
  std::uint32_t timeout_seconds{60};
  // Frames per measurement of the --benchmark mode; 0 runs the graph instead.
  std::uint32_t benchmark{0};
  bool help{false};
};

bool display_available() {
  const auto set = [](const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0';
  };
  return set("DISPLAY") || (set("WAYLAND_DISPLAY") && set("XDG_RUNTIME_DIR"));
}

template <typename T>
T parse_number(std::string_view value, std::string_view option) {
  T result{};
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
  if (error != std::errc() || end != value.data() + value.size()) {
    throw std::invalid_argument(std::string(option) + " requires a number, got '" +
                                std::string(value) + "'");
  }
  return result;
}

template <typename T>
T parse_choice(std::optional<T> parsed, std::string_view option, const char* choices) {
  if (!parsed) {
    throw std::invalid_argument(std::string(option) + " must be " + choices);
  }
  return *parsed;
}

Options parse_args(int argc, char** argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    const std::string_view option{argv[i]};
    if (option == "--help" || option == "-h") {
      opts.help = true;
      return opts;
    }
    if (i + 1 >= argc) {
      throw std::invalid_argument("missing value for " + std::string(option));
    }
    const std::string_view value{argv[++i]};
    hc::EncoderSettings& s = opts.settings;
    if (option == "--codec") {
      s.codec = parse_choice(hc::parse_codec(value), option, "h264 or h265");
    } else if (option == "--backend") {
      s.backend = parse_choice(hc::parse_backend(value), option, "auto, nvcodec or v4l2");
      opts.decoder_backend = s.backend;
    } else if (option == "--rate-control") {
      s.rate_control = parse_choice(hc::parse_rate_control(value), option, "cbr, vbr or cqp");
    } else if (option == "--bitrate") {
      s.bitrate = parse_number<std::uint32_t>(value, option);
    } else if (option == "--qp") {
      s.qp = parse_number<int>(value, option);
    } else if (option == "--gop") {
      s.gop = parse_number<std::uint32_t>(value, option);
    } else if (option == "--preset") {
      s.preset = parse_choice(hc::parse_preset(value), option, "fast, medium or quality");
    } else if (option == "--low-latency") {
      if (value != "true" && value != "false") {
        throw std::invalid_argument("--low-latency requires true or false");
      }
      s.low_latency = value == "true";
    } else if (option == "--width") {
      opts.width = parse_number<int>(value, option);
    } else if (option == "--height") {
      opts.height = parse_number<int>(value, option);
    } else if (option == "--count") {
      opts.count = parse_number<std::uint32_t>(value, option);
    } else if (option == "--period-ms") {
      opts.period_ms = parse_number<std::uint32_t>(value, option);
    } else if (option == "--display") {
      if (value == "auto") {
        opts.display = Display::kAuto;
      } else if (value == "window") {
        opts.display = Display::kWindow;
      } else if (value == "offscreen") {
        opts.display = Display::kOffscreen;
      } else if (value == "none") {
        opts.display = Display::kNone;
      } else {
        throw std::invalid_argument("--display must be auto, window, offscreen or none");
      }
    } else if (option == "--output") {
      opts.output = value;
    } else if (option == "--render-file") {
      opts.render_file = value;
    } else if (option == "--min-psnr") {
      opts.min_psnr = parse_number<double>(value, option);
    } else if (option == "--benchmark") {
      opts.benchmark = parse_number<std::uint32_t>(value, option);
    } else if (option == "--timeout-seconds") {
      opts.timeout_seconds = parse_number<std::uint32_t>(value, option);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(option));
    }
  }
  return opts;
}

void usage(const char* argv0) {
  std::cout
      << "Usage: " << argv0
      << " [options]\n"
         "Encoding:\n"
         "  --codec h264|h265          (default: h264)\n"
         "  --backend auto|nvcodec|v4l2\n"
         "                             The hardware path (default: auto, the first that works)\n"
         "  --rate-control cbr|vbr|cqp (default: vbr)\n"
         "  --bitrate BPS              For cbr and vbr (default: 8000000)\n"
         "  --qp N                     For cqp, 0 (best) to 51 (default: 23)\n"
         "  --gop N                    Frames from one keyframe to the next (default: 30)\n"
         "  --preset fast|medium|quality (default: fast)\n"
         "  --low-latency true|false   (default: true)\n"
         "  --output FILE              Also write the raw stream (.h264 / .h265) here\n"
         "Frames:\n"
         "  --width N --height N       Synthetic frame size (default: 1280x720)\n"
         "  --count N                  Frames, then stop (default: until the window closes or\n"
         "                             Ctrl+C with a window; 60 without)\n"
         "  --period-ms N              Frame period, 1..1000 ms (default: 33)\n"
         "Display and checks:\n"
         "  --display auto|window|offscreen|none\n"
         "                             The original and the decoded frame side by side, in a\n"
         "                             window or rendered offscreen and checked (default: auto, a\n"
         "                             window when there is a display, else none)\n"
         "  --render-file FILE         Offscreen: write the last rendered frame here (PPM)\n"
         "  --min-psnr DB              Fail on a decoded frame below this PSNR (default: 0, off)\n"
         "  --timeout-seconds N        Fail when no frame comes for N seconds (default: 60)\n"
         "Benchmark:\n"
         "  --benchmark N              Instead of the graph, time N frames of encoding and\n"
         "                             decoding at 1080p and 2160p for each codec, and print\n"
         "                             frames per second\n";
}

void validate(Options& opts) {
  hc::validate_settings(opts.settings);
  if (opts.period_ms == 0 || opts.period_ms > 1000 || opts.timeout_seconds == 0) {
    throw std::invalid_argument("--period-ms must be 1..1000 and --timeout-seconds positive");
  }
  if (opts.width < hc::kMinFrameSide || opts.height < hc::kMinFrameSide || opts.width > 4096 ||
      opts.height > 4096) {
    throw std::invalid_argument("--width and --height must be 160 to 4096");
  }
  if (opts.display == Display::kAuto) {
    opts.display = display_available() ? Display::kWindow : Display::kNone;
  }
  if (opts.display == Display::kWindow && !display_available()) {
    throw std::invalid_argument(
        "a window needs a display server; set DISPLAY or use --display "
        "offscreen or none");
  }
  if (!opts.render_file.empty() && opts.display != Display::kOffscreen) {
    throw std::invalid_argument("--render-file needs --display offscreen");
  }
  if (!opts.count && opts.display != Display::kWindow) {
    opts.count = 60;
  }
  if (opts.count && *opts.count == 0) {
    throw std::invalid_argument("--count must be positive");
  }
}

// What the encoder will run on, for the labels: the configured backend, or the one auto picks.
std::string encoder_backend(const hc::EncoderSettings& settings) {
  std::string reasons;
  for (const hc::BackendStatus& status : hc::probe_encoder_backends(settings.codec)) {
    if (settings.backend != hc::Backend::kAuto && status.backend != settings.backend) {
      continue;
    }
    if (status.available) {
      return std::string(hc::backend_name(status.backend));
    }
    reasons += std::string(hc::backend_name(status.backend)) + ": " + status.detail + "; ";
  }
  throw std::runtime_error("no encoder for " + std::string(hc::codec_name(settings.codec)) +
                           " here: " + reasons);
}

// Every sample, in order: a full queue holds back its producer.
holoscan::ConnectionOptions lossless() {
  return {.queue_policy = holoscan::QueuePolicy::kBlocking, .queue_depth = 8U};
}

// Only the newest sample.
holoscan::ConnectionOptions newest() {
  return {.queue_policy = holoscan::QueuePolicy::kDropOldest, .queue_depth = 1U};
}

// The newest few, for a consumer that pairs samples by capture time.
holoscan::ConnectionOptions recent() {
  return {.queue_policy = holoscan::QueuePolicy::kDropOldest, .queue_depth = 8U};
}

using SignalDomain = holoscan::expected<holoscan::ManagedShutdownSignals, holoscan::Error>;

// The exit status that tells CTest a test was skipped.
constexpr int kSkipped = 77;

int execute(const Options& opts, SignalDomain& signals) {
  std::string backend;
  try {
    backend = encoder_backend(opts.settings);
  } catch (const std::runtime_error& error) {
    std::cout << kAppName << ": SKIPPED: " << error.what() << '\n';
    return kSkipped;
  }
  const std::uint32_t frames = opts.count.value_or(0);
  // A finite run checks every frame, in order; an endless one runs in real time.
  const bool finite = frames != 0;
  auto report = std::make_shared<app::CodecReport>();
  const bool render = opts.display != Display::kNone;

  holoscan::Graph graph{std::string(kAppName)};
  auto source = graph.op<app::TestSourceOp>(
      "source",
      app::TestSourceConfig{.width = opts.width,
                            .height = opts.height,
                            .count = frames,
                            .period = std::chrono::milliseconds(opts.period_ms),
                            .done = finite ? std::shared_ptr<const std::atomic<std::uint32_t>>(
                                                 report, &report->frames)
                                           : nullptr});
  auto encoder =
      graph.op<hc::VideoEncoderOp>("encoder", hc::VideoEncoderConfig{.settings = opts.settings,
                                                                     .max_width = opts.width,
                                                                     .max_height = opts.height,
                                                                     .output_path = opts.output});
  auto decoder = graph.op<hc::VideoDecoderOp>(
      "decoder", hc::VideoDecoderConfig{.backend = opts.decoder_backend,
                                        .output_format = hc::PixelFormat::kRgb8,
                                        .max_width = opts.width,
                                        .max_height = opts.height,
                                        .low_latency = opts.settings.low_latency});
  auto reporter =
      graph.op<app::CodecReportOp>("report", app::CodecReportConfig{.report = report,
                                                                    .width = opts.width,
                                                                    .height = opts.height,
                                                                    .codec = opts.settings.codec,
                                                                    .backend = backend,
                                                                    .expected_frames = frames,
                                                                    .min_psnr = opts.min_psnr,
                                                                    .lossless = finite,
                                                                    .text = render});
  // Frames may be skipped before the encoder in real time; never after it, where every access
  // unit is needed to decode the next.
  graph.add_flow(source->image, encoder->image, finite ? lossless() : newest());
  graph.add_flow(encoder->bitstream, decoder->bitstream, lossless());
  graph.add_flow(encoder->bitstream, reporter->bitstream, finite ? lossless() : recent());
  graph.add_flow(decoder->image, reporter->decoded, finite ? lossless() : recent());

  if (render) {
    const auto width = static_cast<std::uint32_t>(opts.width);
    const auto height = static_cast<std::uint32_t>(opts.height);
    viz::LayerStack layers;
    // The decoded frame drives each render; it goes on the right half, the original on the left.
    layers.add_image_layer("decoded", viz::ImageLayerDesc{.format = viz::ImageFormat::kR8G8B8Unorm,
                                                          .max_width = width,
                                                          .max_height = height,
                                                          .view = viz::ViewTransform{
                                                              .offset_x = 0.5F, .scale_x = 0.5F}});
    layers.add_image_layer("original",
                           viz::ImageLayerDesc{.format = viz::ImageFormat::kR8G8B8Unorm,
                                               .max_width = width,
                                               .max_height = height,
                                               .view = viz::ViewTransform{.scale_x = 0.5F}});
    layers.add_text_layer("stats",
                          viz::TextLayerDesc{.max_glyphs = 512, .normalized_coordinates = true});
    const auto max_input_age =
        std::max(8 * std::chrono::milliseconds(opts.period_ms), std::chrono::milliseconds{500});
    viz::SinkDesc sink = viz::WindowSinkDesc{
        .title = "Codec: original | decoded", .width = 2 * width, .height = height};
    if (opts.display == Display::kOffscreen) {
      sink = viz::CudaSinkDesc{
          .width = 2 * width, .height = height, .format = viz::ImageFormat::kR8G8B8A8Unorm};
    }
    auto visualizer =
        graph.op<viz::HolovizOp>("holoviz", std::move(layers), std::move(sink),
                                 viz::FrameTrigger{viz::LayerTrigger{0}}, max_input_age);
    graph.add_flow(decoder->image, visualizer->image_layer_input(0), newest());
    // HoloViz shows the newest original no newer than the decoded frame: keep a few, so that the
    // decoded frame's own original is still there when it arrives.
    graph.add_flow(source->image, visualizer->image_layer_input(1), recent());
    graph.add_flow(reporter->stats, visualizer->tensor_layer_input(2), newest());
    if (opts.display == Display::kOffscreen) {
      auto checker = graph.op<app::RenderCheckOp>(
          "render_check", app::RenderCheckConfig{.report = report,
                                                 .width = opts.width,
                                                 .height = opts.height,
                                                 .min_psnr = std::max(opts.min_psnr - 3.0, 20.0),
                                                 .image_file = opts.render_file});
      graph.add_flow(visualizer->output(), checker->rendered, newest());
    }
  }
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  // GPU outputs are on device 0.
  holoscan::CompileOptions compile_options;
  const auto on_gpu = [&compile_options](std::string op, std::string port) {
    compile_options.deployment.bind_tensor_output_device(
        holoscan::TensorOutputDevicePlacement{.operator_path = std::move(op),
                                              .output_port = std::move(port),
                                              .device = holoscan::DeviceId{0}});
  };
  on_gpu("source", "image");
  on_gpu("decoder", "image");
  if (opts.display == Display::kOffscreen) {
    on_gpu("holoviz", "output");
  }
  const auto plan = holoscan::compile(graph, std::move(compile_options));
  if (!plan.diagnostics().empty()) {
    std::cerr << holoscan::format_diagnostics(plan.diagnostics(), plan.graph()) << '\n';
  }
  if (!plan.ok()) {
    return 1;
  }
  auto session = signals ? holoscan::run_async(plan, *signals) : holoscan::run_async(plan);
  // The run fails when no frame comes for this long; the first one also waits for the sessions.
  const auto patience = std::chrono::steady_clock::duration(
      std::chrono::seconds(opts.timeout_seconds) + 2 * std::chrono::milliseconds(opts.period_ms));
  std::uint32_t seen = 0;
  auto last_progress = std::chrono::steady_clock::now();
  bool timed_out = false;
  while (session.is_running()) {
    std::this_thread::sleep_for(10ms);
    const auto now = std::chrono::steady_clock::now();
    if (const std::uint32_t reported = report->frames.load(); reported != seen) {
      seen = reported;
      last_progress = now;
    } else if (now - last_progress >= patience) {
      timed_out = true;
      break;
    }
  }
  if (timed_out) {
    std::cerr << kAppName << ": timed out: no frame for " << opts.timeout_seconds << " s\n";
    session.request_stop();
  }
  session.wait();
  const holoscan::RunTermination termination = session.termination();
  const bool sdk_reported = termination == holoscan::RunTermination::kFailed &&
                            holoscan::automatic_diagnostic_logging_enabled() &&
                            holoscan::log_level() <= holoscan::LogLevel::ERROR;
  if (!session.diagnostics().empty() && !sdk_reported) {
    std::cerr << holoscan::format_diagnostics(session.diagnostics(), plan.graph()) << '\n';
  }
  if (signals) {
    (void)signals->close();
  }

  const std::scoped_lock lock(report->mutex);
  const std::uint32_t done = report->frames.load();
  for (const std::string& failure : report->failures) {
    std::cerr << kAppName << ": " << failure << '\n';
  }
  const bool stopped = !timed_out && termination == holoscan::RunTermination::kStopped;
  // Nearly every rendered frame shows both halves right (see RenderCheckOp).
  const bool rendered_ok =
      opts.display != Display::kOffscreen ||
      (report->rendered > 0 && report->rendered_matched * 10U >= report->rendered * 8U);
  const bool passed = !timed_out && session.diagnostics().empty() &&
                      termination == holoscan::RunTermination::kRequestedCompleted &&
                      done == frames && report->failures.empty() && rendered_ok;
  // At the frame rate the encoder plans for, whatever the source's period.
  const double seconds =
      static_cast<double>(std::max<std::uint32_t>(done, 1U)) / opts.settings.frame_rate;
  std::cout << std::fixed << std::setprecision(2) << kAppName
            << ": codec=" << hc::codec_name(opts.settings.codec) << " backend=" << backend
            << " frames=" << done;
  if (frames != 0) std::cout << '/' << frames;
  std::cout << " keyframes=" << report->keyframes;
  if (done != 0) {
    std::cout << " psnr_db: mean=" << report->psnr_sum / done << " min=" << report->psnr_min
              << " bitrate_mbps=" << static_cast<double>(report->bytes) * 8.0 / seconds / 1e6;
  }
  if (opts.display == Display::kOffscreen) {
    std::cout << " rendered=" << report->rendered
              << " rendered_matched=" << report->rendered_matched;
  }
  std::cout << '\n'
            << kAppName << ": verdict="
            << (passed    ? "PASS"
                : stopped ? "STOPPED"
                          : "FAIL")
            << '\n';
  return passed || (stopped && frames == 0) ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  // Before any other thread starts, so that the SDK can take over SIGINT and SIGTERM.
  SignalDomain signals = holoscan::ManagedShutdownSignals::create();
  Options opts;
  try {
    opts = parse_args(argc, argv);
    if (opts.help) {
      usage(argv[0]);
      return 0;
    }
    if (opts.benchmark == 0) {
      validate(opts);
    }
  } catch (const std::exception& error) {
    std::cerr << kAppName << ": " << error.what() << "\n(--help lists the options)\n";
    return 2;
  }
  try {
    if (opts.benchmark != 0) {
      return app::run_benchmark(opts.benchmark);
    }
    return execute(opts, signals);
  } catch (const std::exception& error) {
    std::cerr << kAppName << ": " << error.what() << '\n';
    return 1;
  }
}
