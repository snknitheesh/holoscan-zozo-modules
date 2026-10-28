// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Rectify: a stereo source (a synthetic distorted pair with known geometry, or
// a recording) fans out to a StereoRectifyOp per backend, each feeding a
// reporter that checks the rectified pair: on the synthetic pair, that every
// dot lies on the same row in both images, where the geometry puts it; on a
// recording, which the camera has rectified already, that nothing changes.
// The first backend's raw and rectified pairs, with epipolar lines, can also
// be shown on a HoloViz display.

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <functional>
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

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/rectify/backend.hpp>
#include <holoscan/rectify/camera_model.hpp>
#include <holoscan/rectify/rectification.hpp>
#include <holoscan/rectify/stereo_rectify_op.hpp>
#include <holoscan/rectify/synthetic_stereo.hpp>
#include <holoscan/time/realtime_clock.hpp>
#if RECTIFY_ENABLE_VISUALIZATION
#include <holoscan/visualization/holoviz_op.hpp>
#endif

#include "recording.hpp"
#include "report_op.hpp"
#include "stereo_source_op.hpp"

namespace {

using std::chrono_literals::operator""ms;
namespace rc = holoscan::rectify;
namespace fs = std::filesystem;

constexpr std::string_view kAppName = "rectify";

enum class DisplayMode { kAuto, kOn, kOff };

// How long a finite run may take beyond its expected time.
constexpr auto kTimeoutAllowance = std::chrono::seconds{120};

struct Options {
  std::vector<rc::Backend> backends{rc::Backend::kAuto};
  rc::Interpolation interpolation{rc::Interpolation::kLinear};
  double alpha{0.0};
  rc::DistortionModel distortion{rc::DistortionModel::kBrownConrady};
  std::optional<fs::path> recording;
  int width{1280};
  int height{720};
  std::optional<std::uint32_t> count;
  std::uint32_t period_ms{33};
  bool device_input{false};
  DisplayMode display{DisplayMode::kAuto};
  bool compile_only{false};
  bool help{false};
  // Filled in by validate().
  rc::StereoCalibration calibration;
  std::optional<rc::SyntheticStereo> scene;
  std::uint32_t frames{};
};

bool env_set(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && *value != '\0';
}

bool parse_bool(std::string_view value, std::string_view option) {
  if (value == "true") return true;
  if (value == "false") return false;
  throw std::invalid_argument(std::string(option) + " requires true or false");
}

template <typename T>
void parse_number(std::string_view value, std::string_view option, T& result) {
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
  if (error != std::errc() || end != value.data() + value.size()) {
    throw std::invalid_argument(std::string(option) + " requires a number, got '" +
                                std::string(value) + "'");
  }
}

// The backends that can rectify `width` x `height` MONO8 pairs on this machine.
std::vector<rc::Backend> runnable_backends(rc::Interpolation interpolation, int width, int height) {
  const rc::PlatformFacts facts = rc::probe_platform();
  std::vector<rc::Backend> result;
  for (const rc::Backend backend : rc::available_backends()) {
    if (!rc::backend_limitation(backend, facts, rc::PixelFormat::kMono8, interpolation, width,
                                height)) {
      result.push_back(backend);
    }
  }
  return result;
}

// "all" stays a word here: it expands once the image size is known.
std::vector<std::string> split(std::string_view value) {
  std::vector<std::string> names;
  while (!value.empty()) {
    const std::size_t comma = value.find(',');
    names.emplace_back(value.substr(0, comma));
    value = comma == std::string_view::npos ? std::string_view{} : value.substr(comma + 1);
  }
  return names;
}

Options parse_args(int argc, char** argv, std::vector<std::string>& backend_names) {
  Options opts;
  backend_names = {"auto"};
  using Parse = std::function<void(std::string_view value, std::string_view option)>;
  const auto number = [](auto& target) -> Parse {
    return [&target](std::string_view value, std::string_view option) {
      parse_number(value, option, target);
    };
  };
  const auto flag = [](bool& target) -> Parse {
    return [&target](std::string_view value, std::string_view option) {
      target = parse_bool(value, option);
    };
  };
  const std::pair<std::string_view, Parse> kOptions[] = {
      {"--backend",
       [&](std::string_view value, std::string_view) { backend_names = split(value); }},
      {"--interpolation",
       [&](std::string_view value, std::string_view) {
         const auto parsed = rc::parse_interpolation(value);
         if (!parsed) throw std::invalid_argument("--interpolation takes linear or nearest");
         opts.interpolation = *parsed;
       }},
      {"--alpha", number(opts.alpha)},
      {"--distortion",
       [&](std::string_view value, std::string_view) {
         const auto parsed = rc::parse_distortion_model(value);
         if (!parsed || *parsed == rc::DistortionModel::kNone) {
           throw std::invalid_argument(
               "--distortion takes brown_conrady, rational_polynomial or kannala_brandt4");
         }
         opts.distortion = *parsed;
       }},
      {"--recording", [&](std::string_view value, std::string_view) { opts.recording = value; }},
      {"--width", number(opts.width)},
      {"--height", number(opts.height)},
      {"--count",
       [&](std::string_view value, std::string_view option) {
         std::uint32_t count = 0;
         parse_number(value, option, count);
         opts.count = count;
       }},
      {"--period-ms", number(opts.period_ms)},
      {"--device-input", flag(opts.device_input)},
      {"--display",
       [&](std::string_view value, std::string_view option) {
         opts.display = value == "auto"             ? DisplayMode::kAuto
                        : parse_bool(value, option) ? DisplayMode::kOn
                                                    : DisplayMode::kOff;
       }},
      {"--compile-only", flag(opts.compile_only)},
  };
  for (int i = 1; i < argc; ++i) {
    const std::string_view option{argv[i]};
    if (option == "--help" || option == "-h") {
      opts.help = true;
      return opts;
    }
    if (i + 1 >= argc) throw std::invalid_argument("missing value for " + std::string(option));
    const auto* entry = std::find_if(std::begin(kOptions), std::end(kOptions),
                                     [option](const auto& known) { return known.first == option; });
    if (entry == std::end(kOptions)) {
      throw std::invalid_argument("unknown option: " + std::string(option));
    }
    entry->second(argv[++i], option);
  }
  return opts;
}

void usage(const char* argv0) {
  std::cout
      << "Usage: " << argv0
      << " [options]\n"
         "Input (default: a synthetic distorted pair with known geometry):\n"
         "  --distortion MODEL         brown_conrady, rational_polynomial or kannala_brandt4\n"
         "                             (default: brown_conrady)\n"
         "  --width N --height N       Synthetic image size (default: 1280x720)\n"
         "  --recording DIR            A stereo recording (calibration.json, index.csv, PGM\n"
         "                             eyes) instead\n"
         "  --count N                  Pairs (default: 300; a recording's frame count)\n"
         "  --period-ms N              Pair period, 1..1000 ms (default: 33)\n"
         "  --device-input true|false  Pairs in GPU memory (default: false)\n"
         "Rectification:\n"
         "  --backend LIST             Comma-separated: cuda, vpi-cuda, vpi-vic, vpi-pva, auto\n"
         "                             (vpi-vic on a Jetson, else cuda), or all that this\n"
         "                             machine can run (default: auto)\n"
         "  --interpolation MODE       linear or nearest (default: linear)\n"
         "  --alpha X                  0 keeps valid pixels only, 1 every pixel (default: 0)\n"
         "Output:\n"
         "  --display auto|true|false  Show the first backend's raw and rectified pairs with\n"
         "                             epipolar lines (default: auto, on with a display)\n"
         "  --compile-only true|false  Build and compile the graph, then exit (default: false)\n";
}

void validate(Options& opts, const std::vector<std::string>& backend_names) {
  if (opts.period_ms == 0 || opts.period_ms > 1000) {
    throw std::invalid_argument("--period-ms must be 1..1000");
  }
  if (opts.count && *opts.count == 0) throw std::invalid_argument("--count must be positive");
  int width = opts.width;
  int height = opts.height;
  if (opts.recording) {
    opts.calibration = rc::read_stereo_calibration(*opts.recording / "calibration.json");
    const auto frames = rectify_app::read_index(*opts.recording);
    opts.frames = opts.count.value_or(static_cast<std::uint32_t>(frames.size()));
    width = opts.calibration.left.width;
    height = opts.calibration.left.height;
  } else {
    opts.scene = rc::synthetic_stereo(opts.width, opts.height, opts.distortion);
    opts.calibration = opts.scene->calibration;
    opts.frames = opts.count.value_or(300U);
  }
  opts.backends.clear();
  const auto add = [&opts](rc::Backend backend) {
    if (std::find(opts.backends.begin(), opts.backends.end(), backend) != opts.backends.end()) {
      throw std::invalid_argument("backend '" + std::string(rc::backend_name(backend)) +
                                  "' is listed twice");
    }
    opts.backends.push_back(backend);
  };
  for (const std::string& name : backend_names) {
    if (name == "all") {
      for (const rc::Backend b : runnable_backends(opts.interpolation, width, height)) add(b);
    } else if (const auto backend = rc::parse_backend(name)) {
      rc::check_available(*backend);
      add(*backend);
    } else {
      throw std::invalid_argument("unknown backend '" + name +
                                  "'; use cuda, vpi-cuda, vpi-vic, vpi-pva, auto or all");
    }
  }
  if (opts.backends.empty()) throw std::invalid_argument("--backend needs at least one backend");
}

// What auto must resolve to on a CI runner, from HOLOSCAN_CI_PLATFORM, for
// MONO8 pairs with linear interpolation: the VIC on a Jetson, cuda on x86_64.
// (Thor's VIC has no nearest interpolation, so there auto may pick cuda.)
std::optional<std::string> expected_auto_backend(rc::Interpolation interpolation) {
  if (interpolation != rc::Interpolation::kLinear) return std::nullopt;
  const char* platform = std::getenv("HOLOSCAN_CI_PLATFORM");
  if (platform == nullptr) return std::nullopt;
  if (std::string_view(platform) == "jetson") return "vpi-vic";
  if (std::string_view(platform) == "x86_64") return "cuda";
  return std::nullopt;
}

// A recording is rectified already: rectifying it must keep its camera.
bool keeps_camera(const rc::StereoCalibration& c, const rc::StereoRectification& r) {
  return std::abs(r.fx - c.left.fy) < 1e-6 && std::abs(r.cx_left - c.left.cx) < 1e-3 &&
         std::abs(r.cy - c.left.cy) < 1e-3 && r.width == c.left.width && r.height == c.left.height;
}

using SignalDomain = holoscan::expected<holoscan::ManagedShutdownSignals, holoscan::Error>;

int execute(const Options& opts, SignalDomain& signals) {
  holoscan::Graph graph{std::string(kAppName)};
  const auto period = std::chrono::milliseconds(opts.period_ms);
  // One report per backend, made now so that the source can count what they finished.
  std::vector<std::shared_ptr<rectify_app::Report>> reports;
  for (std::size_t i = 0; i < opts.backends.size(); ++i) {
    reports.push_back(std::make_shared<rectify_app::Report>());
  }
  auto source = graph.op<rectify_app::StereoSourceOp>(
      "source", rectify_app::StereoSourceConfig{.scene = opts.scene,
                                                .recording = opts.recording.value_or(fs::path{}),
                                                .count = opts.frames,
                                                .period = period,
                                                .device_output = opts.device_input,
                                                .done = [reports] {
                                                  std::uint32_t done = UINT32_MAX;
                                                  for (const auto& report : reports)
                                                    done = std::min(done, report->frames.load());
                                                  return done;
                                                }});

  const bool display_available =
      env_set("DISPLAY") || (env_set("WAYLAND_DISPLAY") && env_set("XDG_RUNTIME_DIR"));
  const bool display =
      opts.display == DisplayMode::kOn ||
      (opts.display == DisplayMode::kAuto && display_available && RECTIFY_ENABLE_VISUALIZATION);
  if (display && !RECTIFY_ENABLE_VISUALIZATION) {
    throw std::invalid_argument("--display needs a build with visualization enabled");
  }
  if (display && !display_available && !opts.compile_only) {
    throw std::invalid_argument(
        "the display needs a display server; set DISPLAY or add --display false");
  }

  // Lossless edges: every rectifier and reporter sees every pair, in order, for the checks.
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  const rc::StereoRectifyOptions rectify_options{.alpha = opts.alpha};
  const rc::StereoRectification rectification =
      rc::compute_stereo_rectification(opts.calibration, rectify_options);
  std::cout << std::setprecision(6) << kAppName << ": geometry " << rectification.width << "x"
            << rectification.height << " fx=" << rectification.fx
            << " cx_left=" << rectification.cx_left << " cx_right=" << rectification.cx_right
            << " cy=" << rectification.cy << " baseline_m=" << rectification.baseline
            << " doffs=" << rectification.doffs << " valid_left=" << rectification.valid_left.x
            << "," << rectification.valid_left.y << "," << rectification.valid_left.width << "x"
            << rectification.valid_left.height << '\n';
  auto remaining = std::make_shared<std::atomic<int>>(static_cast<int>(opts.backends.size()));
  holoscan::CompileOptions compile_options;
  for (std::size_t i = 0; i < opts.backends.size(); ++i) {
    const std::string name = std::string(rc::backend_name(opts.backends[i]));
    std::string suffix = name;
    std::replace(suffix.begin(), suffix.end(), '-', '_');
    auto rectify = graph.op<rc::StereoRectifyOp>(
        "rectify_" + suffix, rc::StereoRectifyConfig{.calibration = opts.calibration,
                                                     .options = rectify_options,
                                                     .backend = opts.backends[i],
                                                     .interpolation = opts.interpolation,
                                                     .geometry_output = true});
    const std::shared_ptr<rectify_app::Report>& report = reports[i];
    auto reporter = graph.op<rectify_app::ReportOp>(
        "report_" + suffix, rectify_app::ReportConfig{.report = report,
                                                      .remaining = remaining,
                                                      .expected_frames = opts.frames,
                                                      .scene = opts.scene,
                                                      .rectification = rectification,
                                                      .view_output = display && i == 0});
    graph.add_flow(source->left, rectify->left, lossless);
    graph.add_flow(source->right, rectify->right, lossless);
    graph.add_flow(source->left, reporter->raw_left, lossless);
    graph.add_flow(source->right, reporter->raw_right, lossless);
    graph.add_flow(rectify->rectified_left, reporter->rectified_left, lossless);
    graph.add_flow(rectify->rectified_right, reporter->rectified_right, lossless);
    graph.add_flow(rectify->geometry, reporter->geometry, lossless);
    // The rectified images are in device memory, on GPU 0.
    for (const char* port : {"rectified_left", "rectified_right"}) {
      compile_options.deployment.bind_tensor_output_device(
          holoscan::TensorOutputDevicePlacement{.operator_path = "rectify_" + suffix,
                                                .output_port = port,
                                                .device = holoscan::DeviceId{0}});
    }

#if RECTIFY_ENABLE_VISUALIZATION
    if (display && i == 0) {
      namespace viz = holoscan::visualization;
      const auto view_width = static_cast<std::uint32_t>(2 * rectification.width);
      const auto view_height = static_cast<std::uint32_t>(2 * rectification.height);
      viz::LayerStack layers;
      layers.add_image_layer("view", viz::ImageLayerDesc{.format = viz::ImageFormat::kR8G8B8A8Unorm,
                                                         .max_width = view_width,
                                                         .max_height = view_height});
      // Half size for large pairs, so that the window fits a screen.
      const std::uint32_t scale = view_width > 1920 ? 2U : 1U;
      auto visualizer = graph.op<viz::HolovizOp>(
          "holoviz", std::move(layers),
          viz::WindowSinkDesc{.title = "Rectify (" + name + "): raw above, rectified below",
                              .width = view_width / scale,
                              .height = view_height / scale});
      // Live: the display keeps the newest view and never holds back the reporter.
      graph.add_flow(reporter->view, visualizer->image_layer_input(0),
                     holoscan::ConnectionOptions{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                                 .queue_depth = 1U});
    }
#endif
  }

  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  if (opts.device_input) {
    for (const char* port : {"left", "right"}) {
      compile_options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
          .operator_path = "source", .output_port = port, .device = holoscan::DeviceId{0}});
    }
  }
  const auto plan = holoscan::compile(graph, std::move(compile_options));
  if (!plan.diagnostics().empty()) {
    std::cerr << holoscan::format_diagnostics(plan.diagnostics(), plan.graph()) << '\n';
  }
  if (!plan.ok()) return 1;
  if (opts.compile_only) {
    std::cout << kAppName << ": plan=OK\n";
    return 0;
  }
  if (!signals) {
    // A thread started before main() (profilers, sanitizers) makes the SDK refuse.
    std::cerr << kAppName << ": warning: running without managed SIGINT/SIGTERM handling: "
              << holoscan::runtime_error_message(signals.error())
              << "; an interrupt ends the process without a report\n";
  }
  auto session = signals ? holoscan::run_async(plan, *signals) : holoscan::run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + kTimeoutAllowance +
                        std::chrono::milliseconds(3LL * opts.frames * opts.period_ms);
  while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(10ms);
  }
  const bool timed_out = session.is_running();
  if (timed_out) {
    std::cerr << kAppName << ": timed out\n";
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
    if (auto closed = signals->close(); !closed) {
      std::cerr << kAppName << ": warning: closing the signal domain failed: "
                << holoscan::runtime_error_message(closed.error()) << '\n';
    }
  }

  const bool stopped = !timed_out && termination == holoscan::RunTermination::kStopped;
  bool passed = !timed_out && session.diagnostics().empty() &&
                termination == holoscan::RunTermination::kRequestedCompleted;
  if (opts.recording && !keeps_camera(opts.calibration, rectification)) {
    std::cout << kAppName << ": the recording is rectified, but rectifying it changes its camera\n";
    passed = false;
  }
  const auto expected_auto = expected_auto_backend(opts.interpolation);
  for (std::size_t i = 0; i < reports.size(); ++i) {
    const rectify_app::Report& report = *reports[i];
    const std::string& chosen = report.backend;
    std::cout << std::fixed << std::setprecision(3) << kAppName << ": backend=";
    if (opts.backends[i] == rc::Backend::kAuto) std::cout << "auto:";
    std::cout << chosen << " frames=" << report.frames.load() << '/' << opts.frames
              << " geometry=" << (report.geometry_ok ? "ok" : "WRONG");
    bool ok = report.frames == opts.frames && report.geometry_ok;
    if (opts.scene) {
      std::cout << " dots=" << report.dots << " missing=" << report.missing
                << " row_error_px(max/rms)=" << report.max_row_error << '/'
                << report.rms_row_error()
                << " position_error_px(max)=" << report.max_position_error;
      ok = ok && report.dots > 0 && report.missing == 0 &&
           report.max_row_error < rectify_app::kMaxRowError &&
           report.max_position_error < rectify_app::kMaxPositionError;
    } else {
      std::cout << " max_pixel_change=" << report.max_pixel_change;
      ok = ok && report.max_pixel_change <= rectify_app::kMaxPixelChange;
    }
    std::cout << '\n' << std::defaultfloat;
    if (opts.backends[i] == rc::Backend::kAuto && expected_auto) {
      std::cout << kAppName << ": auto=" << chosen << " expected=" << *expected_auto << " ("
                << std::getenv("HOLOSCAN_CI_PLATFORM") << ")\n";
      ok = ok && chosen == *expected_auto;
    }
    passed = passed && ok;
  }
  std::cout << kAppName << ": verdict=" << (passed ? "PASS" : stopped ? "STOPPED" : "FAIL") << '\n';
  return passed ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) try {
  // Take over SIGINT and SIGTERM first: the SDK refuses once other threads run.
  SignalDomain signals = holoscan::ManagedShutdownSignals::create();
  std::vector<std::string> backend_names;
  auto opts = parse_args(argc, argv, backend_names);
  if (opts.help) {
    usage(argv[0]);
    return 0;
  }
  validate(opts, backend_names);
  return execute(opts, signals);
} catch (const std::invalid_argument& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 2;
} catch (const std::exception& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 1;
}
