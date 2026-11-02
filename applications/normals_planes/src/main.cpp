// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Normals and Planes: a synthetic depth camera in a known room, deprojected
// by holoscan-point-cloud's DepthToPointCloudOp; holoscan-normals'
// NormalEstimationOp estimates every point's normal and curvature, and
// PlaneSegmentationOp finds the planes and labels every pixel with its plane.
// A reporter checks every frame against the room's truth, and with a display
// HoloViz shows the normals color-mapped beside the plane labels.

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/normals/normal_estimation_op.hpp>
#include <holoscan/normals/plane_segmentation_op.hpp>
#include <holoscan/normals/synthetic_scene.hpp>
#include <holoscan/point_cloud/depth_to_point_cloud_op.hpp>
#include <holoscan/time/realtime_clock.hpp>
#if NORMALS_PLANES_ENABLE_VISUALIZATION
#include <holoscan/visualization/holoviz_op.hpp>
#endif

#include "render_check_op.hpp"
#include "report_op.hpp"
#include "scene_source_op.hpp"
#include "view_op.hpp"

namespace {

using std::chrono_literals::operator""ms;
namespace app = normals_planes_app;
namespace pc = holoscan::point_cloud;
namespace hn = holoscan::normals;

constexpr std::string_view kAppName = "normals_planes";

enum class DisplayMode { kAuto, kOn, kOff, kOffscreen };

struct Options {
  std::uint32_t count{300};
  std::uint32_t period_ms{33};
  int width{640};
  int height{480};
  int window_radius{3};
  hn::NeighborSearch search{hn::NeighborSearch::kWindow};
  bool merge{false};
  DisplayMode display{DisplayMode::kAuto};
  std::uint32_t timeout_seconds{120};
  bool compile_only{false};
  bool help{false};
};

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
    if (option == "--count") {
      opts.count = parse_number<std::uint32_t>(value, option);
    } else if (option == "--period-ms") {
      opts.period_ms = parse_number<std::uint32_t>(value, option);
    } else if (option == "--width") {
      opts.width = parse_number<int>(value, option);
    } else if (option == "--height") {
      opts.height = parse_number<int>(value, option);
    } else if (option == "--window-radius") {
      opts.window_radius = parse_number<int>(value, option);
    } else if (option == "--search") {
      if (value == "window") {
        opts.search = hn::NeighborSearch::kWindow;
      } else if (value == "knn") {
        opts.search = hn::NeighborSearch::kKnn;
      } else if (value == "radius") {
        opts.search = hn::NeighborSearch::kRadius;
      } else {
        throw std::invalid_argument("--search must be window, knn or radius");
      }
    } else if (option == "--merge") {
      if (value != "true" && value != "false") {
        throw std::invalid_argument("--merge requires true or false");
      }
      opts.merge = value == "true";
    } else if (option == "--display") {
      if (value == "auto") {
        opts.display = DisplayMode::kAuto;
      } else if (value == "true") {
        opts.display = DisplayMode::kOn;
      } else if (value == "false") {
        opts.display = DisplayMode::kOff;
      } else if (value == "offscreen") {
        opts.display = DisplayMode::kOffscreen;
      } else {
        throw std::invalid_argument("--display must be auto, true, false or offscreen");
      }
    } else if (option == "--timeout-seconds") {
      opts.timeout_seconds = parse_number<std::uint32_t>(value, option);
    } else if (option == "--compile-only") {
      if (value != "true" && value != "false") {
        throw std::invalid_argument("--compile-only requires true or false");
      }
      opts.compile_only = value == "true";
    } else {
      throw std::invalid_argument("unknown option: " + std::string(option));
    }
  }
  return opts;
}

void usage(const char* argv0) {
  std::cout << "Usage: " << argv0
            << " [options]\n"
               "  --count N                Frames (default: 300)\n"
               "  --period-ms N            Camera period, 1..1000 ms (default: 33)\n"
               "  --width N --height N     Image size, 64x48 to 4096x4096 (default: 640x480)\n"
               "  --search window|knn|radius  Neighbors of the normals (default: window)\n"
               "  --window-radius N        Window of (2N + 1)^2 pixels, 1..15 (default: 3)\n"
               "  --merge true|false       Merge coplanar planes (default: false)\n"
               "  --display auto|true|false|offscreen\n"
               "                           Show normals and planes in HoloViz (default: auto, on\n"
               "                           when a display is available)\n"
               "  --timeout-seconds N      Allowance beyond the expected run time (default: 120)\n"
               "  --compile-only true|false  Build and check the graph without running it\n";
}

bool env_set(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && *value != '\0';
}

using SignalDomain = holoscan::expected<holoscan::ManagedShutdownSignals, holoscan::Error>;

int execute(const Options& opts, SignalDomain& signals) {
  const bool display_available =
      env_set("DISPLAY") || (env_set("WAYLAND_DISPLAY") && env_set("XDG_RUNTIME_DIR"));
  const bool display = opts.display == DisplayMode::kOn ||
                       (opts.display == DisplayMode::kAuto && display_available &&
                        NORMALS_PLANES_ENABLE_VISUALIZATION);
  const bool offscreen = opts.display == DisplayMode::kOffscreen;
  if (offscreen && !NORMALS_PLANES_ENABLE_VISUALIZATION) {
    throw std::invalid_argument("--display needs a build with visualization enabled");
  }
  if (display && !NORMALS_PLANES_ENABLE_VISUALIZATION) {
    throw std::invalid_argument("--display needs a build with visualization enabled");
  }
  if (display && !display_available && !opts.compile_only) {
    throw std::invalid_argument(
        "the display needs a display server; set DISPLAY or add --display false");
  }

  const hn::SyntheticScene scene = hn::synthetic_scene(opts.width, opts.height, 0);
  const pc::CameraIntrinsics intrinsics{scene.fx, scene.fy, scene.cx, scene.cy};
  const auto points =
      static_cast<std::uint32_t>(opts.width) * static_cast<std::uint32_t>(opts.height);

  holoscan::Graph graph{std::string(kAppName)};
  auto finished = std::make_shared<std::atomic<std::uint32_t>>(0U);
  auto source = graph.op<app::SceneSourceOp>(
      "source", app::SceneSourceConfig{.width = opts.width,
                                       .height = opts.height,
                                       .count = opts.count,
                                       .period = std::chrono::milliseconds(opts.period_ms),
                                       .finished = finished});
  auto deproject = graph.op<pc::DepthToPointCloudOp>(
      "deproject", pc::DepthToPointCloudConfig{.intrinsics = intrinsics,
                                               .depth_scale = 1.0F,
                                               .depth_min = 0.1F,
                                               .depth_max = 10.0F,
                                               .max_width = opts.width,
                                               .max_height = opts.height});
  hn::NormalEstimationConfig normal_config{.max_points = points};
  normal_config.normals.search = opts.search;
  normal_config.normals.window_radius = opts.window_radius;
  normal_config.normals.radius = opts.search == hn::NeighborSearch::kKnn ? 0.1F : 0.04F;
  auto normals = graph.op<hn::NormalEstimationOp>("normals", normal_config);
  hn::PlaneSegmentationConfig plane_config{.max_points = points};
  plane_config.planes.merge_coplanar = opts.merge;
  // A plane needs a share of the image: 500 points at 640x480.
  plane_config.planes.min_inliers = std::max(100U, points / 600U);
  auto planes = graph.op<hn::PlaneSegmentationOp>("planes", plane_config);
  auto report = std::make_shared<app::Report>();
  if (opts.search == hn::NeighborSearch::kKnn) {
    report->max_median_deg = 3.0;
    report->max_p95_deg = 9.0;
  }
  auto reporter = graph.op<app::ReportOp>(
      "report", app::ReportConfig{.report = report,
                                  .width = opts.width,
                                  .height = opts.height,
                                  .window_radius = opts.window_radius,
                                  .max_planes = plane_config.planes.max_planes,
                                  .expected_frames = opts.count,
                                  .finished = finished});

  // Lossless edges: every frame is estimated, segmented and checked, in
  // order. The source keeps one frame in flight, so no queue fills.
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(source->depth, deproject->depth, lossless);
  graph.add_flow(deproject->point_cloud, normals->points, lossless);
  graph.add_flow(normals->filtered_points, planes->points, lossless);
  graph.add_flow(normals->normals, planes->normals, lossless);
  graph.add_flow(normals->normals, reporter->normals, lossless);
  graph.add_flow(normals->curvature, reporter->curvature, lossless);
  graph.add_flow(planes->labels, reporter->labels, lossless);
  graph.add_flow(planes->plane_coefficients, reporter->coefficients, lossless);
  graph.add_flow(planes->plane_centroids, reporter->centroids, lossless);
  graph.add_flow(planes->plane_inliers, reporter->inliers, lossless);

  holoscan::CompileOptions compile_options;
  const auto on_gpu = [&compile_options](std::string op, std::string port) {
    compile_options.deployment.bind_tensor_output_device(
        holoscan::TensorOutputDevicePlacement{.operator_path = std::move(op),
                                              .output_port = std::move(port),
                                              .device = holoscan::DeviceId{0}});
  };
  on_gpu("source", "depth");
  on_gpu("deproject", "point_cloud");
  for (const char* port : {"filtered_points", "normals", "curvature"}) {
    on_gpu("normals", port);
  }
  for (const char* port : {"labels", "plane_coefficients", "plane_centroids", "plane_inliers"}) {
    on_gpu("planes", port);
  }

#if NORMALS_PLANES_ENABLE_VISUALIZATION
  auto rendered = std::make_shared<app::RenderTally>();
  if (display || offscreen) {
    namespace viz = holoscan::visualization;
    // The normals on the left, the planes on the right.
    auto view =
        graph.op<app::ViewOp>("view", app::ViewConfig{.width = opts.width, .height = opts.height});
    viz::LayerStack layers;
    layers.add_image_layer(
        "view", viz::ImageLayerDesc{.format = viz::ImageFormat::kR8G8B8Unorm,
                                    .max_width = static_cast<std::uint32_t>(2 * opts.width),
                                    .max_height = static_cast<std::uint32_t>(opts.height)});
    // On screen, or offscreen into CUDA memory, where RenderCheckOp looks at each frame.
    viz::SinkDesc sink = viz::WindowSinkDesc{.title = "Normals and Planes: normals | planes",
                                             .width = static_cast<std::uint32_t>(2 * opts.width),
                                             .height = static_cast<std::uint32_t>(opts.height)};
    if (offscreen) {
      sink = viz::CudaSinkDesc{.width = static_cast<std::uint32_t>(2 * opts.width),
                               .height = static_cast<std::uint32_t>(opts.height)};
    }
    auto visualizer = graph.op<viz::HolovizOp>("holoviz", std::move(layers), std::move(sink));
    // Best effort: the display keeps the newest frame and never holds back the pipeline.
    const holoscan::ConnectionOptions latest{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                             .queue_depth = 1U};
    graph.add_flow(normals->normals, view->normals, latest);
    graph.add_flow(planes->labels, view->labels, latest);
    graph.add_flow(view->image, visualizer->image_layer_input(0), latest);
    on_gpu("view", "image");
    if (offscreen) {
      auto check = graph.op<app::RenderCheckOp>("render_check", rendered);
      graph.add_flow(visualizer->output(), check->frame, lossless);
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
              << holoscan::runtime_error_message(signals.error())
              << "; an interrupt ends the process without a report\n";
  }
  auto session = signals ? holoscan::run_async(plan, *signals) : holoscan::run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(opts.timeout_seconds) +
                        std::chrono::milliseconds(2LL * opts.count * opts.period_ms);
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
  // Offscreen, HoloViz must have shown both views; it may skip frames, never all.
  const bool shown = !offscreen || rendered->showing != 0;
  const bool passed = !timed_out && session.diagnostics().empty() &&
                      termination == holoscan::RunTermination::kRequestedCompleted &&
                      report->frames == opts.count && report->passed() && shown;
  const app::Report& r = *report;
  std::cout << std::fixed << std::setprecision(3) << kAppName << ": frames=" << r.frames << '/'
            << opts.count << " normals: worst_median_deg=" << r.worst_median_deg
            << " worst_p95_deg=" << r.worst_p95_deg << " flipped=" << r.flipped
            << " planes_per_frame="
            << (r.frames == 0 ? 0.0 : static_cast<double>(r.planes) / r.frames)
            << " planes: worst_angle_deg=" << r.worst_plane_angle_deg
            << " worst_offset_mm=" << 1000.0 * r.worst_plane_offset_m
            << " worst_recall=" << r.worst_recall << " worst_precision=" << r.worst_precision
            << " wrong_planes=" << r.wrong_planes << " flat_sphere=" << r.flat_sphere;
  if (r.frame_gaps != 0) {
    std::cout << " frame_gaps=" << r.frame_gaps;
  }
  if (offscreen) {
    std::cout << " rendered=" << rendered->frames << " showing=" << rendered->showing;
  }
  std::cout << '\n'
            << kAppName << ": verdict="
            << (passed    ? "PASS"
                : stopped ? "STOPPED"
                          : "FAIL")
            << '\n';
  return passed ? 0 : 1;
}

void validate(const Options& opts) {
  if (opts.period_ms == 0 || opts.period_ms > 1000 || opts.timeout_seconds == 0) {
    throw std::invalid_argument("--period-ms must be 1..1000 and --timeout-seconds positive");
  }
  app::validate_config(app::SceneSourceConfig{.width = opts.width,
                                              .height = opts.height,
                                              .count = opts.count,
                                              .period = std::chrono::milliseconds(opts.period_ms)});
  hn::NormalEstimationConfig normals;
  normals.normals.window_radius = opts.window_radius;
  hn::validate_config(normals);
}

}  // namespace

int main(int argc, char** argv) try {
  // Take over SIGINT and SIGTERM while this is the only thread; the SDK
  // refuses a process that already runs others.
  SignalDomain signals = holoscan::ManagedShutdownSignals::create();
  const Options opts = parse_args(argc, argv);
  if (opts.help) {
    usage(argv[0]);
    return 0;
  }
  validate(opts);
  return execute(opts, signals);
} catch (const std::invalid_argument& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 2;
} catch (const std::exception& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 1;
}
