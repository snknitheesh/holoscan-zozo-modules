// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// ICP Odometry: depth odometry from a moving depth camera. A synthetic depth
// camera moves through a room along a known, closed path;
// holoscan-point-cloud's DepthToPointCloudOp turns each depth image into a
// point cloud, and holoscan-icp's IcpOp registers each cloud to the one before
// and accumulates the camera's pose. A reporter checks the estimated
// trajectory against the true one; a HoloViz window can show both, and the
// current alignment.

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
#include <holoscan/icp/icp_op.hpp>
#include <holoscan/icp/synthetic_scene.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/point_cloud/depth_to_point_cloud_op.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "depth_camera_op.hpp"
#include "odometry_report_op.hpp"

#if ICP_ODOMETRY_ENABLE_VISUALIZATION
#include <holoscan/visualization/holoviz_op.hpp>

#include "trajectory_view_op.hpp"
#endif

namespace {

using std::chrono_literals::operator""ms;
namespace app = icp_odometry_app;
namespace icp = holoscan::icp;
namespace pc = holoscan::point_cloud;

constexpr std::string_view kAppName = "icp_odometry";

enum class DisplayMode { kAuto, kOn, kOff };

struct Options {
  int width{640};
  int height{480};
  std::uint32_t count{300};
  std::optional<std::uint32_t> path_frames;
  std::uint32_t period_ms{33};
  bool noise{true};
  icp::IcpOptions icp;
  bool max_iterations_given{false};
  app::OdometryLimits limits;
  DisplayMode display{DisplayMode::kAuto};
  std::uint32_t timeout_seconds{120};
  bool compile_only{false};
  bool help{false};
};

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

bool env_set(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && *value != '\0';
}

bool display_available() {
  return env_set("DISPLAY") || (env_set("WAYLAND_DISPLAY") && env_set("XDG_RUNTIME_DIR"));
}

Options parse_args(int argc, char** argv) {
  Options opts;
  // The demo's ICP settings: coarse to fine from 10 cm to 3 cm, on a 2 cm
  // voxel subsampling, with normals within 6 cm.
  opts.icp.max_correspondence_distance = 0.10F;
  opts.icp.min_correspondence_distance = 0.03F;
  opts.icp.voxel_size = 0.02F;
  opts.icp.normal_radius = 0.06F;
  opts.icp.max_iterations = 100;
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
    if (option == "--width") {
      opts.width = parse_number<int>(value, option);
    } else if (option == "--height") {
      opts.height = parse_number<int>(value, option);
    } else if (option == "--count") {
      opts.count = parse_number<std::uint32_t>(value, option);
    } else if (option == "--path-frames") {
      opts.path_frames = parse_number<std::uint32_t>(value, option);
    } else if (option == "--period-ms") {
      opts.period_ms = parse_number<std::uint32_t>(value, option);
    } else if (option == "--noise") {
      opts.noise = parse_bool(value, option);
    } else if (option == "--method") {
      if (value == "point-to-plane") {
        opts.icp.method = icp::IcpMethod::kPointToPlane;
      } else if (value == "point-to-point") {
        opts.icp.method = icp::IcpMethod::kPointToPoint;
      } else {
        throw std::invalid_argument("--method must be point-to-plane or point-to-point");
      }
    } else if (option == "--voxel-size") {
      opts.icp.voxel_size = parse_number<float>(value, option);
    } else if (option == "--max-distance") {
      opts.icp.max_correspondence_distance = parse_number<float>(value, option);
    } else if (option == "--min-distance") {
      opts.icp.min_correspondence_distance = parse_number<float>(value, option);
    } else if (option == "--normal-radius") {
      opts.icp.normal_radius = parse_number<float>(value, option);
    } else if (option == "--max-iterations") {
      opts.icp.max_iterations = parse_number<int>(value, option);
      opts.max_iterations_given = true;
    } else if (option == "--robust") {
      if (value == "none") {
        opts.icp.robust_kernel = icp::RobustKernel::kNone;
      } else if (value == "huber") {
        opts.icp.robust_kernel = icp::RobustKernel::kHuber;
      } else if (value == "tukey") {
        opts.icp.robust_kernel = icp::RobustKernel::kTukey;
      } else {
        throw std::invalid_argument("--robust must be none, huber or tukey");
      }
    } else if (option == "--robust-scale") {
      opts.icp.robust_scale = parse_number<float>(value, option);
    } else if (option == "--max-drift-percent") {
      opts.limits.max_drift_percent = parse_number<double>(value, option);
    } else if (option == "--max-rotation-error-deg") {
      opts.limits.max_rotation_error_deg = parse_number<double>(value, option);
    } else if (option == "--max-unconverged") {
      opts.limits.max_unconverged = parse_number<std::uint32_t>(value, option);
    } else if (option == "--display") {
      opts.display = value == "auto"             ? DisplayMode::kAuto
                     : parse_bool(value, option) ? DisplayMode::kOn
                                                 : DisplayMode::kOff;
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
         "Camera:\n"
         "  --width N --height N       Depth image size, 64..4096 x 48..4096 (default: 640x480)\n"
         "  --count N                  Frames (default: 300)\n"
         "  --path-frames N            Frames the closed path takes (default: --count)\n"
         "  --period-ms N              Camera period, 1..1000 ms (default: 33)\n"
         "  --noise true|false         Time-of-flight depth noise (default: true)\n"
         "ICP:\n"
         "  --method point-to-plane|point-to-point  (default: point-to-plane)\n"
         "  --voxel-size M             Subsampling voxel, 0 for none (default: 0.02)\n"
         "  --max-distance M --min-distance M\n"
         "                             Correspondence distance, coarse to fine\n"
         "                             (default: 0.10 to 0.03)\n"
         "  --normal-radius M          (default: 0.06)\n"
         "  --max-iterations N         (default: 100; 600 for point-to-point)\n"
         "  --robust none|huber|tukey  --robust-scale M  (default: none, 0.01)\n"
         "Check, against the camera's true path:\n"
         "  --max-drift-percent P      Final position error over path length (default: 2)\n"
         "  --max-rotation-error-deg D Final rotation error (default: 1.5)\n"
         "  --max-unconverged N        Frames whose registration may fail (default: 0)\n"
         "Run:\n"
         "  --display auto|true|false  Show the trajectories and the alignment in a HoloViz\n"
         "                             window (default: auto, on when a display is available)\n"
         "  --timeout-seconds N        Allowance beyond the expected run time (default: 120)\n"
         "  --compile-only true|false  Build and check the graph without running it\n";
}

app::DepthCameraConfig camera_config(const Options& opts) {
  return app::DepthCameraConfig{.width = opts.width,
                                .height = opts.height,
                                .count = opts.count,
                                .path_frames = opts.path_frames.value_or(opts.count),
                                .period = std::chrono::milliseconds(opts.period_ms),
                                .noise = opts.noise};
}

pc::DepthToPointCloudConfig deproject_config(const Options& opts) {
  const icp::PinholeCamera k = icp::synthetic_camera(opts.width, opts.height);
  return pc::DepthToPointCloudConfig{.intrinsics = pc::CameraIntrinsics{k.fx, k.fy, k.cx, k.cy},
                                     .depth_scale = 1.0F,
                                     .depth_min = 0.1F,
                                     .depth_max = 8.0F,
                                     .max_width = opts.width,
                                     .max_height = opts.height};
}

icp::IcpConfig icp_config(const Options& opts, bool display) {
  icp::IcpConfig config;
  config.options = opts.icp;
  config.mode = icp::IcpMode::kOdometry;
  // The pose and the statistics; the frame-to-frame transform is not used,
  // and an output must be connected if it exists. The display also shows the
  // current cloud moved into the previous frame.
  config.transform_output = false;
  config.aligned_output = display;
  config.max_points =
      static_cast<std::uint32_t>(opts.width) * static_cast<std::uint32_t>(opts.height);
  return config;
}

void validate(Options& opts) {
  if (opts.period_ms == 0 || opts.period_ms > 1000 || opts.timeout_seconds == 0) {
    throw std::invalid_argument("--period-ms must be 1..1000 and --timeout-seconds positive");
  }
  if (opts.icp.method == icp::IcpMethod::kPointToPoint && !opts.max_iterations_given) {
    opts.icp.max_iterations = 600;  // It converges linearly.
  }
  if (!(opts.limits.max_drift_percent >= 0.0) || !(opts.limits.max_rotation_error_deg >= 0.0)) {
    throw std::invalid_argument("the limits must not be negative");
  }
  app::validate_config(camera_config(opts));
  pc::validate_config(deproject_config(opts));
  icp::validate_config(icp_config(opts, false));
  // The display: on when one is available, unless set.
  if (opts.display == DisplayMode::kAuto) {
    opts.display = ICP_ODOMETRY_ENABLE_VISUALIZATION && display_available() ? DisplayMode::kOn
                                                                            : DisplayMode::kOff;
  }
  if (opts.display == DisplayMode::kOn) {
    if (!ICP_ODOMETRY_ENABLE_VISUALIZATION) {
      throw std::invalid_argument("--display needs a build with visualization enabled");
    }
    if (!display_available() && !opts.compile_only) {
      throw std::invalid_argument(
          "the display needs a display server; set DISPLAY or add --display false");
    }
  }
}

using SignalDomain = holoscan::expected<holoscan::ManagedShutdownSignals, holoscan::Error>;

int execute(const Options& opts, SignalDomain& signals) {
  holoscan::Graph graph{std::string(kAppName)};
  const bool display = opts.display == DisplayMode::kOn;
  auto report = std::make_shared<app::OdometryReport>();
  // The camera sends a frame only once the reporter is done with the one
  // before, so the lossless chain never meets a full queue: an operator in
  // the middle cannot send a refused output again. The count is the signal,
  // read by the camera; an edge back from the reporter would be a cycle,
  // which the SDK refuses.
  app::DepthCameraConfig camera = camera_config(opts);
  camera.done = std::shared_ptr<const std::atomic<std::uint32_t>>(report, &report->frames);
  auto source = graph.op<app::SyntheticDepthCameraOp>("camera", camera);
  auto deproject = graph.op<pc::DepthToPointCloudOp>("deproject", deproject_config(opts));
  auto odometry = graph.op<icp::IcpOp>("odometry", icp_config(opts, display));
  auto reporter = graph.op<app::OdometryReportOp>(
      "report", app::OdometryReportConfig{.report = report, .expected_frames = opts.count});

  // Lossless edges: every frame is registered and checked, in order.
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(source->depth, deproject->depth, lossless);
  graph.add_flow(deproject->point_cloud, odometry->source, lossless);
  graph.add_flow(odometry->pose, reporter->pose, lossless);
  graph.add_flow(odometry->stats, reporter->stats, lossless);
  graph.add_flow(source->true_pose, reporter->true_pose, lossless);

#if ICP_ODOMETRY_ENABLE_VISUALIZATION
  if (display) {
    namespace viz = holoscan::visualization;
    const icp::PinholeCamera k = icp::synthetic_camera(opts.width, opts.height);
    const auto width = static_cast<std::uint32_t>(app::view_width(k));
    const auto height = static_cast<std::uint32_t>(app::view_height(k));
    auto view = graph.op<app::TrajectoryViewOp>(
        "view", app::TrajectoryViewConfig{.camera = k,
                                          .path_frames = opts.path_frames.value_or(opts.count),
                                          .count = opts.count});
    viz::LayerStack layers;
    layers.add_image_layer("view", viz::ImageLayerDesc{.format = viz::ImageFormat::kR8G8B8A8Unorm,
                                                       .max_width = width,
                                                       .max_height = height});
    layers.add_text_layer(
        "labels", viz::TextLayerDesc{.max_glyphs = static_cast<std::uint32_t>(app::kViewLabelRows *
                                                                              app::kViewLabelBytes),
                                     .normalized_coordinates = true});
    // A window at least 1280 pixels wide.
    const std::uint32_t zoom = std::max(1U, 1280U / width);
    auto visualizer = graph.op<viz::HolovizOp>(
        "holoviz", std::move(layers),
        viz::WindowSinkDesc{
            .title = "ICP Odometry", .width = width * zoom, .height = height * zoom},
        viz::FrameTrigger{viz::LayerTrigger{0}},
        std::max(4 * std::chrono::milliseconds(opts.period_ms), std::chrono::milliseconds{250}));
    // The display keeps the newest frames and never holds back the odometry:
    // two slots, so that the view's five inputs of one frame meet.
    const holoscan::ConnectionOptions recent{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                             .queue_depth = 2U};
    const holoscan::ConnectionOptions newest{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                             .queue_depth = 1U};
    graph.add_flow(odometry->pose, view->pose, recent);
    graph.add_flow(odometry->stats, view->stats, recent);
    graph.add_flow(odometry->aligned, view->aligned, recent);
    graph.add_flow(deproject->point_cloud, view->point_cloud, recent);
    graph.add_flow(source->true_pose, view->true_pose, recent);
    graph.add_flow(view->view, visualizer->image_layer_input(0), newest);
    graph.add_flow(view->labels, visualizer->tensor_layer_input(1), newest);
  }
#endif
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  // GPU outputs are on device 0.
  holoscan::CompileOptions compile_options;
  const auto on_gpu = [&compile_options](std::string op, std::string port) {
    compile_options.deployment.bind_tensor_output_device(
        holoscan::TensorOutputDevicePlacement{.operator_path = std::move(op),
                                              .output_port = std::move(port),
                                              .device = holoscan::DeviceId{0}});
  };
  on_gpu("camera", "depth");
  on_gpu("deproject", "point_cloud");
  if (display) {
    on_gpu("odometry", "aligned");
  }
  const auto plan = holoscan::compile(graph, std::move(compile_options));
  // Warnings on an accepted plan, as well as the reasons for a rejected one.
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
    // Profilers, sanitizers and preloaded libraries can start a thread before
    // main(); the SDK then refuses to take over SIGINT and SIGTERM.
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
  // The SDK logs a failed run's diagnostics itself, unless its report or its
  // logger is turned down; show those of any other run.
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

  // A stop request (SIGINT, SIGTERM) ends the run early; that is not a
  // failure, but not a pass either.
  const bool stopped = !timed_out && termination == holoscan::RunTermination::kStopped;
  const bool passed = !timed_out && session.diagnostics().empty() &&
                      termination == holoscan::RunTermination::kRequestedCompleted &&
                      app::passed(*report, opts.count, opts.limits);
  const std::uint32_t frames = report->frames.load();
  const double registered = frames > 1U ? static_cast<double>(frames - 1U) : 1.0;
  std::cout << std::fixed << std::setprecision(2) << kAppName << ": frames=" << frames << '/'
            << opts.count << ' ' << opts.width << 'x' << opts.height << " path_m=" << report->path_m
            << " final_error_mm=" << report->final_error_m * 1000.0 << " final_rotation_error_deg="
            << report->final_rotation_error_rad * 180.0 / 3.14159265358979323846
            << " drift_percent=" << report->drift_percent()
            << " ate_rmse_mm=" << report->ate_rmse_m() * 1000.0
            << " max_error_mm=" << report->max_error_m * 1000.0
            << " unconverged=" << report->unconverged
            << " mean_fitness=" << report->sum_fitness / registered
            << " mean_rmse_mm=" << report->sum_rmse_m * 1000.0 / registered
            << " mean_iterations=" << static_cast<double>(report->iterations) / registered;
  if (report->frame_gaps != 0U) {
    std::cout << " frame_gaps=" << report->frame_gaps;
  }
  std::cout << '\n'
            << kAppName << ": verdict="
            << (passed    ? "PASS"
                : stopped ? "STOPPED"
                          : "FAIL")
            << '\n';
  return passed ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) try {
  // Take over SIGINT and SIGTERM while this is the only thread; the SDK
  // refuses a process that already runs others.
  SignalDomain signals = holoscan::ManagedShutdownSignals::create();
  auto opts = parse_args(argc, argv);
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
