// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Stitching: four synthetic fisheye cameras on a rig that drives around a
// checkered world; one StitchOp makes a bird's-eye surround view of the
// ground, another a 360-degree panorama of the wall around the rig. HoloViz
// shows the cameras and both outputs, in a window or rendered offscreen.
// Headless and offscreen runs check every output against the ground truth
// rendered directly from the world: PSNR and the checkerboards' corners.

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
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
#include <holoscan/stitching/camera_rig.hpp>
#include <holoscan/stitching/stitch_op.hpp>
#include <holoscan/stitching/synthetic_world.hpp>
#include <holoscan/time/realtime_clock.hpp>
#if STITCHING_APP_ENABLE_VISUALIZATION
#include <holoscan/visualization/holoviz_op.hpp>
#endif

#include "check_ops.hpp"
#include "rig_source_op.hpp"

namespace {

using std::chrono_literals::operator""ms;
namespace app = stitching_app;
namespace st = holoscan::stitching;

constexpr std::string_view kAppName = "stitching";
// The render: the bird's-eye view at the left, the cameras at the right, the
// panorama below.
constexpr int kRenderWidth = 2048;
constexpr int kRenderHeight = 1280;
constexpr int kTopHeight = 768;

enum class Display { kAuto, kWindow, kOffscreen, kNone };

struct Options {
  std::filesystem::path rig_file;
  int camera_width{1280};
  int camera_height{720};
  int samples{3};
  int panorama_width{4096};
  int panorama_height{1024};
  int birds_eye_size{1024};
  double birds_eye_extent{5.5};
  st::BlendMode blend{st::BlendMode::kFeather};
  float feather_width{0.0F};
  bool exposure{false};
  std::vector<float> camera_gains;
  std::optional<std::uint32_t> count;
  std::uint32_t period_ms{33};
  Display display{Display::kAuto};
  std::filesystem::path image_dir;
  std::uint32_t timeout_seconds{60};
  bool compile_only{false};
  bool help{false};
};

bool env_set(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && *value != '\0';
}

bool display_available() {
  return env_set("DISPLAY") || (env_set("WAYLAND_DISPLAY") && env_set("XDG_RUNTIME_DIR"));
}

bool parse_bool(std::string_view value, std::string_view option) {
  if (value == "true") return true;
  if (value == "false") return false;
  throw std::invalid_argument(std::string(option) + " requires true or false");
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

std::vector<float> parse_list(std::string_view value, std::string_view option) {
  std::vector<float> result;
  while (!value.empty()) {
    const std::size_t comma = value.find(',');
    result.push_back(parse_number<float>(value.substr(0, comma), option));
    value = comma == std::string_view::npos ? std::string_view{} : value.substr(comma + 1);
  }
  return result;
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
    if (option == "--rig") {
      opts.rig_file = value;
    } else if (option == "--camera-width") {
      opts.camera_width = parse_number<int>(value, option);
    } else if (option == "--camera-height") {
      opts.camera_height = parse_number<int>(value, option);
    } else if (option == "--samples") {
      opts.samples = parse_number<int>(value, option);
    } else if (option == "--panorama-width") {
      opts.panorama_width = parse_number<int>(value, option);
    } else if (option == "--panorama-height") {
      opts.panorama_height = parse_number<int>(value, option);
    } else if (option == "--birds-eye-size") {
      opts.birds_eye_size = parse_number<int>(value, option);
    } else if (option == "--birds-eye-extent") {
      opts.birds_eye_extent = parse_number<double>(value, option);
    } else if (option == "--blend") {
      const auto blend = st::parse_blend_mode(value);
      if (!blend) throw std::invalid_argument("--blend must be feather or hard_seam");
      opts.blend = *blend;
    } else if (option == "--feather-width") {
      opts.feather_width = parse_number<float>(value, option);
    } else if (option == "--exposure") {
      opts.exposure = parse_bool(value, option);
    } else if (option == "--camera-gains") {
      opts.camera_gains = parse_list(value, option);
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
    } else if (option == "--image-dir") {
      opts.image_dir = value;
    } else if (option == "--timeout-seconds") {
      opts.timeout_seconds = parse_number<std::uint32_t>(value, option);
    } else if (option == "--compile-only") {
      opts.compile_only = parse_bool(value, option);
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
         "Rig and world:\n"
         "  --rig FILE                 A rig file (see the module's README) to render instead\n"
         "                             of the built-in four-fisheye surround rig\n"
         "  --camera-width N --camera-height N\n"
         "                             The built-in rig's image size (default: 1280x720)\n"
         "  --samples N                Rays per pixel along each side, 1..8 (default: 3)\n"
         "  --camera-gains G,G,...     Each camera's exposure (default: all 1)\n"
         "  --count N                  Frames, then stop (default: 60; with a window, until it\n"
         "                             is closed)\n"
         "  --period-ms N              The frame period, 1..1000 ms (default: 33)\n"
         "Stitching:\n"
         "  --panorama-width N --panorama-height N\n"
         "                             The cylindrical panorama's size (default: 4096x1024)\n"
         "  --birds-eye-size N         The bird's-eye view's side (default: 1024)\n"
         "  --birds-eye-extent M       The ground it shows, M metres each way (default: 5.5)\n"
         "  --blend feather|hard_seam  How overlaps blend (default: feather)\n"
         "  --feather-width PX         Cap of the feathering weight, 0 for none (default: 0)\n"
         "  --exposure true|false      Per-camera exposure compensation (default: false)\n"
         "Display:\n"
         "  --display auto|window|offscreen|none\n"
         "                             A HoloViz window (live: the newest frame only), HoloViz\n"
         "                             rendering offscreen, every render checked, or no display;\n"
         "                             auto is window with a display, else none\n"
         "  --image-dir DIR            With offscreen, also write each render as PPM there\n"
         "  --timeout-seconds N        Fail when no frame finishes for N seconds (default: 60)\n"
         "  --compile-only true|false  Build and check the graph without running it\n";
}

st::CameraRig make_rig(const Options& opts) {
  if (!opts.rig_file.empty()) {
    return st::read_rig(opts.rig_file);
  }
  return st::surround_rig(opts.camera_width, opts.camera_height);
}

void validate(Options& opts) {
  if (opts.display == Display::kAuto) {
    opts.display = display_available() ? Display::kWindow : Display::kNone;
  }
#if !STITCHING_APP_ENABLE_VISUALIZATION
  if (opts.display != Display::kNone) {
    throw std::invalid_argument("--display needs a build with visualization enabled");
  }
#endif
  if (opts.display == Display::kWindow && !display_available() && !opts.compile_only) {
    throw std::invalid_argument(
        "the window needs a display server; set DISPLAY, or use --display offscreen or none");
  }
  if (opts.display == Display::kWindow) {
    opts.count = opts.count.value_or(0);
  } else {
    opts.count = opts.count.value_or(60);
    if (*opts.count == 0) {
      throw std::invalid_argument("--count must be positive without a window");
    }
  }
  if (opts.timeout_seconds == 0) {
    throw std::invalid_argument("--timeout-seconds must be positive");
  }
  if (!opts.image_dir.empty()) {
    if (opts.display != Display::kOffscreen) {
      throw std::invalid_argument("--image-dir writes offscreen renders");
    }
    std::error_code error;
    if (!std::filesystem::is_directory(opts.image_dir, error)) {
      throw std::invalid_argument("--image-dir is not a directory: " + opts.image_dir.string());
    }
  }
  const st::CameraRig rig = make_rig(opts);
  if (opts.display != Display::kNone && rig.cameras.size() > 6) {
    throw std::invalid_argument("HoloViz shows up to 6 cameras with the two outputs");
  }
  app::validate_config(app::RigSourceConfig{.rig = rig,
                                            .gains = opts.camera_gains,
                                            .period = std::chrono::milliseconds(opts.period_ms),
                                            .samples = opts.samples});
  if (!(opts.birds_eye_extent > 0.0 && opts.birds_eye_extent < 1000.0)) {
    throw std::invalid_argument("--birds-eye-extent must be in (0, 1000) metres");
  }
}

// The largest rectangle of `aspect` (width over height) centred in a box.
app::Tile fit(int x, int y, int width, int height, double aspect) {
  int w = width;
  int h = static_cast<int>(width / aspect);
  if (h > height) {
    h = height;
    w = static_cast<int>(height * aspect);
  }
  return app::Tile{x + (width - w) / 2, y + (height - h) / 2, w, h};
}

// Where the render shows the bird's-eye view, the panorama and each camera.
[[maybe_unused]] std::vector<app::Tile> layout(const st::OutputGeometry& birds_eye,
                                               const st::OutputGeometry& panorama,
                                               const st::CameraRig& rig) {
  std::vector<app::Tile> tiles;
  tiles.push_back(
      fit(0, 0, kTopHeight, kTopHeight, static_cast<double>(birds_eye.width) / birds_eye.height));
  tiles.push_back(fit(0, kTopHeight, kRenderWidth, kRenderHeight - kTopHeight,
                      static_cast<double>(panorama.width) / panorama.height));
  const int n = static_cast<int>(rig.cameras.size());
  const int rows = (n + 1) / 2;
  const int cell_width = (kRenderWidth - kTopHeight) / 2;
  const int cell_height = kTopHeight / rows;
  for (int i = 0; i < n; ++i) {
    const st::Camera& camera = rig.cameras[i];
    tiles.push_back(fit(kTopHeight + (i % 2) * cell_width, (i / 2) * cell_height, cell_width,
                        cell_height, static_cast<double>(camera.width) / camera.height));
  }
  return tiles;
}

using SignalDomain = holoscan::expected<holoscan::ManagedShutdownSignals, holoscan::Error>;

int execute(const Options& opts, SignalDomain& signals) {
  holoscan::Graph graph{std::string(kAppName)};
  const st::CameraRig rig = make_rig(opts);
  const st::SyntheticWorld world;
  const st::OutputGeometry birds_eye =
      st::ground_view(opts.birds_eye_size, opts.birds_eye_size, opts.birds_eye_extent);
  const st::OutputGeometry panorama = st::wall_panorama(world, st::Projection::kCylindrical,
                                                        opts.panorama_width, opts.panorama_height);
  auto report = std::make_shared<app::Report>();
  const bool live = opts.display == Display::kWindow;
  const bool offscreen = opts.display == Display::kOffscreen;

  // Lossless (no window): one frame in flight, sent once its outputs are
  // checked; every frame is stitched and checked. Live (a window): every
  // edge keeps only the newest frame.
  app::RigSourceConfig source_config{.rig = rig,
                                     .world = world,
                                     .gains = opts.camera_gains,
                                     .count = *opts.count,
                                     .period = std::chrono::milliseconds(opts.period_ms),
                                     .samples = opts.samples};
  if (!live) {
    source_config.done = [report] { return report->frames.load(); };
  }
  auto source = graph.op<app::RigSourceOp>("cameras", std::move(source_config));
  st::BlendSettings blend;
  blend.mode = opts.blend;
  blend.feather_width = opts.feather_width;
  blend.exposure_compensation = opts.exposure;
  auto stitch_birds_eye = graph.op<st::StitchOp>(
      "birds_eye",
      st::StitchConfig{.rig = rig, .output = birds_eye, .blend = blend, .frame_id = "rig"});
  auto stitch_panorama = graph.op<st::StitchOp>(
      "panorama",
      st::StitchConfig{.rig = rig, .output = panorama, .blend = blend, .frame_id = "rig"});

  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  const holoscan::ConnectionOptions newest{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                           .queue_depth = 1U};
  const holoscan::ConnectionOptions& edge = live ? newest : lossless;
  for (std::size_t c = 0; c < rig.cameras.size(); ++c) {
    graph.add_flow(source->images[c], stitch_birds_eye->images[c], edge);
    graph.add_flow(source->images[c], stitch_panorama->images[c], edge);
  }
  if (!live) {
    auto check = graph.op<app::StitchCheckOp>(
        "check", app::StitchCheckConfig{
                     .report = report,
                     .rig = rig,
                     .world = world,
                     .birds_eye = birds_eye,
                     .panorama = panorama,
                     // A hard seam through a corner joins two cameras' differently
                     // blurred halves; compensation may leave all cameras a little
                     // brighter or darker.
                     .thresholds = app::Thresholds{
                         .corner_error_px = opts.blend == st::BlendMode::kHardSeam ? 1.5 : 1.0,
                         .gain_error = opts.exposure ? 0.1 : 0.02}});
    graph.add_flow(stitch_birds_eye->stitched, check->birds_eye, lossless);
    graph.add_flow(stitch_panorama->stitched, check->panorama, lossless);
  }

  holoscan::CompileOptions compile_options;
  const auto on_gpu = [&compile_options](std::string op, std::string port) {
    compile_options.deployment.bind_tensor_output_device(
        holoscan::TensorOutputDevicePlacement{.operator_path = std::move(op),
                                              .output_port = std::move(port),
                                              .device = holoscan::DeviceId{0}});
  };
  for (const st::Camera& camera : rig.cameras) {
    on_gpu("cameras", camera.name);
  }
  on_gpu("birds_eye", "stitched");
  on_gpu("panorama", "stitched");

#if STITCHING_APP_ENABLE_VISUALIZATION
  if (opts.display != Display::kNone) {
    namespace viz = holoscan::visualization;
    const std::vector<app::Tile> tiles = layout(birds_eye, panorama, rig);
    viz::LayerStack layers;
    const auto add_layer = [&](const std::string& name, int width, int height,
                               const app::Tile& tile) {
      layers.add_image_layer(
          name,
          viz::ImageLayerDesc{.format = viz::ImageFormat::kR8G8B8A8Unorm,
                              .max_width = static_cast<std::uint32_t>(width),
                              .max_height = static_cast<std::uint32_t>(height),
                              .view = viz::ViewTransform{
                                  .offset_x = static_cast<float>(tile.x) / kRenderWidth,
                                  .offset_y = static_cast<float>(tile.y) / kRenderHeight,
                                  .scale_x = static_cast<float>(tile.width) / kRenderWidth,
                                  .scale_y = static_cast<float>(tile.height) / kRenderHeight}});
    };
    add_layer("birds_eye", birds_eye.width, birds_eye.height, tiles[0]);
    add_layer("panorama", panorama.width, panorama.height, tiles[1]);
    for (std::size_t c = 0; c < rig.cameras.size(); ++c) {
      add_layer(rig.cameras[c].name, rig.cameras[c].width, rig.cameras[c].height, tiles[2 + c]);
    }
    viz::SinkDesc sink = viz::WindowSinkDesc{
        .title = "Stitching", .width = kRenderWidth / 2, .height = kRenderHeight / 2};
    if (offscreen) {
      sink = viz::CudaSinkDesc{.width = kRenderWidth,
                               .height = kRenderHeight,
                               .format = viz::ImageFormat::kR8G8B8A8Unorm};
    }
    // Each bird's-eye view draws a render, over the newest of the others.
    auto visualizer = graph.op<viz::HolovizOp>("holoviz", std::move(layers), std::move(sink),
                                               viz::FrameTrigger{viz::LayerTrigger{0}},
                                               std::chrono::milliseconds(1000));
    graph.add_flow(stitch_birds_eye->stitched, visualizer->image_layer_input(0), newest);
    graph.add_flow(stitch_panorama->stitched, visualizer->image_layer_input(1), newest);
    for (std::size_t c = 0; c < rig.cameras.size(); ++c) {
      graph.add_flow(source->images[c], visualizer->image_layer_input(2 + c), newest);
    }
    if (offscreen) {
      auto render_check = graph.op<app::RenderCheckOp>(
          "render_check", app::RenderCheckConfig{.report = report,
                                                 .width = kRenderWidth,
                                                 .height = kRenderHeight,
                                                 .tiles = tiles,
                                                 .image_dir = opts.image_dir});
      graph.add_flow(visualizer->output(), render_check->rendered, newest);
      on_gpu("holoviz", "output");
    }
  }
#endif
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const auto plan = holoscan::compile(graph, std::move(compile_options));
  if (!plan.diagnostics().empty()) {
    std::cerr << holoscan::format_diagnostics(plan.diagnostics(), plan.graph()) << '\n';
  }
  if (!plan.ok()) {
    return 1;
  }
  if (opts.compile_only) {
    std::cout << kAppName << ": plan=OK\n";
    return 0;
  }
  if (!signals) {
    std::cerr << kAppName << ": warning: running without managed SIGINT/SIGTERM handling: "
              << holoscan::runtime_error_message(signals.error()) << '\n';
  }
  auto session = signals ? holoscan::run_async(plan, *signals) : holoscan::run_async(plan);
  const std::uint32_t frames = *opts.count;
  const auto finished = [&] { return report->frames.load(); };
  std::uint32_t seen = 0;
  auto last_progress = std::chrono::steady_clock::now();
  bool timed_out = false;
  while (session.is_running()) {
    std::this_thread::sleep_for(10ms);
    const auto now = std::chrono::steady_clock::now();
    if (!live && finished() >= frames) {
      // HoloViz renders the last frames a little later: wait until the
      // renders stop coming, for at most two seconds.
      if (offscreen) {
        std::uint32_t renders = report->rendered.load();
        for (int i = 0; i < 20; ++i) {
          std::this_thread::sleep_for(100ms);
          const std::uint32_t now_rendered = report->rendered.load();
          if (now_rendered == renders && renders != 0) break;
          renders = now_rendered;
        }
      }
      session.request_stop();
      break;
    }
    if (live) continue;
    if (const std::uint32_t done = finished(); done != seen) {
      seen = done;
      last_progress = now;
    } else if (now - last_progress >= std::chrono::seconds(opts.timeout_seconds)) {
      timed_out = true;
      std::cerr << kAppName << ": timed out: no frame finished for " << opts.timeout_seconds
                << " s\n";
      session.request_stop();
      break;
    }
  }
  session.wait();
  const holoscan::RunTermination termination = session.termination();
  const bool sdk_reported = termination == holoscan::RunTermination::kFailed &&
                            holoscan::automatic_diagnostic_logging_enabled() &&
                            holoscan::log_level() <= holoscan::LogLevel::ERROR;
  const bool all_checked = !live && finished() >= frames;
  if (!session.diagnostics().empty() && !sdk_reported && !all_checked) {
    std::cerr << holoscan::format_diagnostics(session.diagnostics(), plan.graph()) << '\n';
  }
  if (signals) {
    if (auto closed = signals->close(); !closed) {
      std::cerr << kAppName << ": warning: closing the signal domain failed: "
                << holoscan::runtime_error_message(closed.error()) << '\n';
    }
  }

  std::cout << kAppName << ": " << rig.cameras.size() << " cameras, birds_eye " << birds_eye.width
            << 'x' << birds_eye.height << ", " << st::projection_name(panorama.projection) << ' '
            << panorama.width << 'x' << panorama.height << ", blend "
            << st::blend_mode_name(opts.blend)
            << (opts.exposure ? " with exposure compensation" : "") << '\n';
  if (live) {
    const bool stopped = termination == holoscan::RunTermination::kStopped ||
                         termination == holoscan::RunTermination::kRequestedCompleted;
    std::cout << kAppName << ": verdict=" << (stopped ? "STOPPED" : "FAIL") << '\n';
    return stopped ? 0 : 1;
  }
  bool passed = !timed_out && all_checked && termination != holoscan::RunTermination::kFailed;
  {
    const std::lock_guard lock(report->mutex);
    passed = passed && report->failed_frames == 0 && report->failed_renders == 0;
    passed = passed && (!offscreen || report->rendered.load() != 0);
    const auto print = [](const char* name, const app::Worst& w) {
      std::cout << ' ' << name << ": psnr>=" << w.psnr_db << " coverage>=" << w.coverage
                << " corners>=" << w.corners << " corner_error<=" << w.max_corner_error
                << " gain_error<=" << w.gain_error;
    };
    std::cout << std::fixed << std::setprecision(3) << kAppName
              << ": frames=" << report->frames.load() << '/' << frames;
    print("birds_eye", report->birds_eye);
    print("panorama", report->panorama);
    if (offscreen) {
      std::cout << " renders=" << report->rendered.load();
    }
    std::cout << '\n';
    if (!report->first_failure.empty()) {
      std::cout << kAppName << ": first failure: " << report->first_failure << '\n';
    }
  }
  std::cout << kAppName << ": verdict=" << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  // Before any other thread, so that the SDK can own SIGINT and SIGTERM.
  SignalDomain signals = holoscan::ManagedShutdownSignals::create();
  Options opts;
  try {
    opts = parse_args(argc, argv);
    if (opts.help) {
      usage(argv[0]);
      return 0;
    }
    validate(opts);
  } catch (const std::invalid_argument& error) {
    std::cerr << "error: " << error.what() << "\nSee " << argv[0] << " --help\n";
    return 2;
  }
  try {
    return execute(opts, signals);
  } catch (const std::exception& error) {
    std::cerr << kAppName << ": error: " << error.what() << '\n';
    return 1;
  }
}
