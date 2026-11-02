// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Clustering: a synthetic depth camera over objects on a removed ground,
// deprojected by holoscan-point-cloud's DepthToPointCloudOp and split into
// objects on the GPU by holoscan-clustering's EuclideanClusterOp. A reporter
// checks every frame against the scene's truth, and with a display the
// clusters are drawn in their colors, with their boxes, in HoloViz.

#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <holoscan/clustering/euclidean_cluster_op.hpp>
#include <holoscan/clustering/synthetic_scene.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/point_cloud/depth_to_point_cloud_op.hpp>
#include <holoscan/time/realtime_clock.hpp>
#if CLUSTERING_ENABLE_VISUALIZATION
#include <holoscan/visualization/holoviz_op.hpp>

#include "cluster_view_op.hpp"
#endif

#include "cluster_report_op.hpp"
#include "scene_source_op.hpp"

namespace {

using std::chrono_literals::operator""ms;
namespace app = clustering_app;
namespace hc = holoscan::clustering;
namespace pc = holoscan::point_cloud;

constexpr std::string_view kAppName = "clustering";

enum class DisplayMode { kAuto, kOn, kOff };

struct Options {
  std::uint32_t count{300};
  std::uint32_t period_ms{33};
  int width{640};
  int height{480};
  float tolerance{0.025F};
  std::uint32_t min_size{300};
  hc::Connectivity connectivity{hc::Connectivity::kEuclidean};
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
    } else if (option == "--tolerance") {
      opts.tolerance = parse_number<float>(value, option);
    } else if (option == "--min-size") {
      opts.min_size = parse_number<std::uint32_t>(value, option);
    } else if (option == "--connectivity") {
      if (value == "euclidean") {
        opts.connectivity = hc::Connectivity::kEuclidean;
      } else if (value == "image") {
        opts.connectivity = hc::Connectivity::kImage;
      } else {
        throw std::invalid_argument("--connectivity must be euclidean or image");
      }
    } else if (option == "--display") {
      if (value == "auto") {
        opts.display = DisplayMode::kAuto;
      } else if (value == "true") {
        opts.display = DisplayMode::kOn;
      } else if (value == "false") {
        opts.display = DisplayMode::kOff;
      } else {
        throw std::invalid_argument("--display must be auto, true or false");
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
  std::cout
      << "Usage: " << argv0
      << " [options]\n"
         "  --count N                Frames (default: 300)\n"
         "  --period-ms N            Camera period, 1..1000 ms (default: 33)\n"
         "  --width N --height N     Image size, 64x48 to 4096x4096 (default: 640x480)\n"
         "  --tolerance M            Largest distance within a cluster, meters\n"
         "                           (default: 0.025)\n"
         "  --min-size N             Smallest cluster, points (default: 300)\n"
         "  --connectivity euclidean|image\n"
         "                           Neighbors in space or in the image (default: euclidean)\n"
         "  --display auto|true|false  Show the clusters in HoloViz (default: auto, on\n"
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
  const bool display =
      opts.display == DisplayMode::kOn ||
      (opts.display == DisplayMode::kAuto && display_available && CLUSTERING_ENABLE_VISUALIZATION);
  if (display && !CLUSTERING_ENABLE_VISUALIZATION) {
    throw std::invalid_argument("--display needs a build with visualization enabled");
  }
  if (display && !display_available && !opts.compile_only) {
    throw std::invalid_argument(
        "the display needs a display server; set DISPLAY or add --display false");
  }

  const hc::SyntheticScene scene = hc::synthetic_scene(opts.width, opts.height, 0);
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
  auto cluster = graph.op<hc::EuclideanClusterOp>(
      "cluster", hc::EuclideanClusterConfig{.clustering = {.tolerance = opts.tolerance,
                                                           .min_cluster_size = opts.min_size,
                                                           .connectivity = opts.connectivity},
                                            .oriented_boxes = true,
                                            .max_points = points});
  auto report = std::make_shared<app::ClusterReport>();
  auto reporter = graph.op<app::ClusterReportOp>(
      "report", app::ClusterReportConfig{.report = report,
                                         .width = opts.width,
                                         .height = opts.height,
                                         .min_cluster_size = opts.min_size,
                                         .expected_frames = opts.count,
                                         .finished = finished});

  // Lossless edges: every frame is clustered and checked, in order. The
  // source keeps one frame in flight, so no queue fills and no output is
  // refused.
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(source->depth, deproject->depth, lossless);
  graph.add_flow(deproject->point_cloud, cluster->points, lossless);
  graph.add_flow(deproject->point_cloud, reporter->points, lossless);
  graph.add_flow(cluster->labels, reporter->labels, lossless);
  graph.add_flow(cluster->clusters, reporter->clusters, lossless);

  holoscan::CompileOptions compile_options;
  const auto on_gpu = [&compile_options](std::string op, std::string port) {
    compile_options.deployment.bind_tensor_output_device(
        holoscan::TensorOutputDevicePlacement{.operator_path = std::move(op),
                                              .output_port = std::move(port),
                                              .device = holoscan::DeviceId{0}});
  };
  on_gpu("source", "depth");
  on_gpu("deproject", "point_cloud");
  on_gpu("cluster", "labels");
  on_gpu("cluster", "clusters");

#if CLUSTERING_ENABLE_VISUALIZATION
  if (display) {
    namespace viz = holoscan::visualization;
    auto view = graph.op<app::ClusterViewOp>("view", app::ClusterViewConfig{.width = opts.width,
                                                                            .height = opts.height,
                                                                            .fx = scene.fx,
                                                                            .fy = scene.fy,
                                                                            .cx = scene.cx,
                                                                            .cy = scene.cy});
    viz::LayerStack layers;
    layers.add_image_layer(
        "clusters", viz::ImageLayerDesc{.format = viz::ImageFormat::kR8G8B8Unorm,
                                        .max_width = static_cast<std::uint32_t>(opts.width),
                                        .max_height = static_cast<std::uint32_t>(opts.height)});
    auto visualizer = graph.op<viz::HolovizOp>(
        "holoviz", std::move(layers),
        viz::WindowSinkDesc{.title = "Clustering: objects and their boxes",
                            .width = static_cast<std::uint32_t>(opts.width),
                            .height = static_cast<std::uint32_t>(opts.height)});
    // Best effort: the display keeps the newest frame and never holds back the clustering.
    const holoscan::ConnectionOptions latest{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                             .queue_depth = 1U};
    graph.add_flow(cluster->labels, view->labels, latest);
    graph.add_flow(cluster->clusters, view->clusters, latest);
    graph.add_flow(view->image, visualizer->image_layer_input(0), latest);
    on_gpu("view", "image");
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
  const bool passed = !timed_out && session.diagnostics().empty() &&
                      termination == holoscan::RunTermination::kRequestedCompleted &&
                      report->frames == opts.count && report->passed();
  const app::ClusterReport& r = *report;
  std::cout << std::fixed << std::setprecision(4) << kAppName << ": frames=" << r.frames << '/'
            << opts.count << " clusters_per_frame="
            << (r.frames == 0 ? 0.0 : static_cast<double>(r.clusters) / r.frames)
            << " wrong_counts=" << r.wrong_counts << " min_rand_index=" << r.min_rand_index
            << " min_iou=" << r.min_iou << " max_centroid_error_m=" << r.max_centroid_error_m
            << " max_box_error_m=" << r.max_box_error_m << " wrong_objects=" << r.wrong_objects
            << " inconsistent=" << r.inconsistent;
  if (r.frame_gaps != 0) {
    std::cout << " frame_gaps=" << r.frame_gaps;
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
  hc::validate_config(hc::EuclideanClusterConfig{
      .clustering = {.tolerance = opts.tolerance,
                     .min_cluster_size = opts.min_size,
                     .connectivity = opts.connectivity},
      .max_points = static_cast<std::uint32_t>(opts.width * opts.height)});
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
