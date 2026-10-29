// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// PCL Filters: a synthetic depth camera over a known scene, deprojected by
// holoscan-point-cloud's DepthToPointCloudOp, then filtered on the GPU by
// holoscan-pcl-filters: a crop box, outlier removal, ground removal and a
// voxel grid. A reporter checks every frame against the scene's truth, and
// with a display the voxels are projected back into the camera by
// PointCloudToDepthOp and shown in HoloViz.

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
#include <holoscan/pcl_filters/crop_box_op.hpp>
#include <holoscan/pcl_filters/ground_removal_op.hpp>
#include <holoscan/pcl_filters/outlier_removal_op.hpp>
#include <holoscan/pcl_filters/synthetic_scene.hpp>
#include <holoscan/pcl_filters/voxel_downsample_op.hpp>
#include <holoscan/point_cloud/depth_to_point_cloud_op.hpp>
#include <holoscan/point_cloud/point_cloud_to_depth_op.hpp>
#include <holoscan/time/realtime_clock.hpp>
#if PCL_FILTERS_ENABLE_VISUALIZATION
#include <holoscan/visualization/holoviz_op.hpp>
#endif

#include "filter_report_op.hpp"
#include "scene_source_op.hpp"

namespace {

using std::chrono_literals::operator""ms;
namespace app = pcl_filters_app;
namespace pc = holoscan::point_cloud;
namespace pf = holoscan::pcl_filters;

constexpr std::string_view kAppName = "pcl_filters";

enum class DisplayMode { kAuto, kOn, kOff };

struct Options {
  std::uint32_t count{300};
  std::uint32_t period_ms{33};
  int width{640};
  int height{480};
  float voxel_size{0.02F};
  pf::OutlierMode outliers{pf::OutlierMode::kRadius};
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
    } else if (option == "--voxel-size") {
      opts.voxel_size = parse_number<float>(value, option);
    } else if (option == "--outliers") {
      if (value == "radius") {
        opts.outliers = pf::OutlierMode::kRadius;
      } else if (value == "statistical") {
        opts.outliers = pf::OutlierMode::kStatistical;
      } else {
        throw std::invalid_argument("--outliers must be radius or statistical");
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
  std::cout << "Usage: " << argv0
            << " [options]\n"
               "  --count N                Frames (default: 300)\n"
               "  --period-ms N            Camera period, 1..1000 ms (default: 33)\n"
               "  --width N --height N     Image size, 64x48 to 4096x4096 (default: 640x480)\n"
               "  --voxel-size M           Voxel side in meters (default: 0.02)\n"
               "  --outliers radius|statistical\n"
               "                           Outlier removal (default: radius)\n"
               "  --display auto|true|false  Show the voxels in HoloViz (default: auto, on\n"
               "                           when a display is available)\n"
               "  --timeout-seconds N      Allowance beyond the expected run time (default: 120)\n"
               "  --compile-only true|false  Build and check the graph without running it\n";
}

bool env_set(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && *value != '\0';
}

// Drops what it receives: the projected depth image, which only the display
// branch makes and nothing uses.
class DiscardOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(image, "image"); }
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{image});
    return result;
  }
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto sample = image.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    return {};
  }
  holoscan::Input<holoscan::schema::ImageT> image;
};

using SignalDomain = holoscan::expected<holoscan::ManagedShutdownSignals, holoscan::Error>;

int execute(const Options& opts, SignalDomain& signals) {
  const bool display_available =
      env_set("DISPLAY") || (env_set("WAYLAND_DISPLAY") && env_set("XDG_RUNTIME_DIR"));
  const bool display =
      opts.display == DisplayMode::kOn ||
      (opts.display == DisplayMode::kAuto && display_available && PCL_FILTERS_ENABLE_VISUALIZATION);
  if (display && !PCL_FILTERS_ENABLE_VISUALIZATION) {
    throw std::invalid_argument("--display needs a build with visualization enabled");
  }
  if (display && !display_available && !opts.compile_only) {
    throw std::invalid_argument(
        "the display needs a display server; set DISPLAY or add --display false");
  }

  const pf::SyntheticScene scene = pf::synthetic_scene(opts.width, opts.height, 0);
  const pc::CameraIntrinsics intrinsics{scene.fx, scene.fy, scene.cx, scene.cy};
  const auto points =
      static_cast<std::uint32_t>(opts.width) * static_cast<std::uint32_t>(opts.height);
  float truth[4];
  pf::synthetic_ground_plane(scene, truth);

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
                                               .color = true,
                                               .max_width = opts.width,
                                               .max_height = opts.height});
  // Everything within 6 m of the camera.
  pf::CropBoxConfig crop{.color = true, .max_points = points};
  crop.box.max_distance = 6.0F;
  auto cropper = graph.op<pf::CropBoxOp>("crop", crop);
  pf::OutlierRemovalConfig outliers{.color = true, .max_points = points};
  outliers.outliers.mode = opts.outliers;
  outliers.outliers.radius = 0.03F;
  outliers.outliers.min_neighbors = 3;
  outliers.outliers.k = 8;
  outliers.outliers.search_radius = 0.1F;
  outliers.outliers.std_multiplier = 2.0F;
  auto outlier_filter = graph.op<pf::OutlierRemovalOp>("outliers", outliers);
  // The camera is mounted pitched down; up is the true ground normal.
  pf::GroundRemovalConfig ground{.color = true, .max_points = points};
  std::copy(truth, truth + 3, ground.ground.up);
  ground.ground.distance_threshold = 0.03F;
  auto ground_filter = graph.op<pf::GroundRemovalOp>("ground", ground);
  auto voxels = graph.op<pf::VoxelDownsampleOp>(
      "voxels", pf::VoxelDownsampleConfig{
                    .voxel_size = opts.voxel_size, .color = true, .max_points = points});
  auto report = std::make_shared<app::FilterReport>();
  auto reporter =
      graph.op<app::FilterReportOp>("report", app::FilterReportConfig{.report = report,
                                                                      .width = opts.width,
                                                                      .height = opts.height,
                                                                      .voxel_size = opts.voxel_size,
                                                                      .expected_frames = opts.count,
                                                                      .finished = finished});

  // Lossless edges: every frame is filtered and checked, in order. The source
  // keeps one frame in flight, so no queue fills and no filter is refused.
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(source->depth, deproject->depth, lossless);
  graph.add_flow(source->color, deproject->color, lossless);
  graph.add_flow(deproject->point_cloud, cropper->points, lossless);
  graph.add_flow(deproject->colors, cropper->colors, lossless);
  graph.add_flow(cropper->filtered_points, outlier_filter->points, lossless);
  graph.add_flow(cropper->filtered_colors, outlier_filter->colors, lossless);
  graph.add_flow(outlier_filter->filtered_points, ground_filter->points, lossless);
  graph.add_flow(outlier_filter->filtered_colors, ground_filter->colors, lossless);
  graph.add_flow(ground_filter->filtered_points, voxels->points, lossless);
  graph.add_flow(ground_filter->filtered_colors, voxels->colors, lossless);
  graph.add_flow(ground_filter->filtered_points, reporter->points, lossless);
  graph.add_flow(ground_filter->plane, reporter->plane, lossless);
  graph.add_flow(voxels->filtered_points, reporter->voxels, lossless);
  graph.add_flow(voxels->filtered_colors, reporter->voxel_colors, lossless);

  holoscan::CompileOptions compile_options;
  const auto on_gpu = [&compile_options](std::string op, std::string port) {
    compile_options.deployment.bind_tensor_output_device(
        holoscan::TensorOutputDevicePlacement{.operator_path = std::move(op),
                                              .output_port = std::move(port),
                                              .device = holoscan::DeviceId{0}});
  };
  on_gpu("source", "depth");
  on_gpu("source", "color");
  on_gpu("deproject", "point_cloud");
  on_gpu("deproject", "colors");
  for (const char* op : {"crop", "outliers", "ground", "voxels"}) {
    on_gpu(op, "filtered_points");
    on_gpu(op, "filtered_colors");
  }
  on_gpu("ground", "plane");

#if PCL_FILTERS_ENABLE_VISUALIZATION
  if (display) {
    namespace viz = holoscan::visualization;
    // The voxels, back in the camera that saw them, each a 5-pixel square.
    auto project = graph.op<pc::PointCloudToDepthOp>(
        "project", pc::PointCloudToDepthConfig{.width = opts.width,
                                               .height = opts.height,
                                               .intrinsics = intrinsics,
                                               .depth_min = 0.1F,
                                               .depth_max = 10.0F,
                                               .point_size = 5,
                                               .color = true,
                                               .max_points = points});
    auto discard = graph.op<DiscardOp>("discard");
    viz::LayerStack layers;
    layers.add_image_layer(
        "voxels", viz::ImageLayerDesc{.format = viz::ImageFormat::kR8G8B8Unorm,
                                      .max_width = static_cast<std::uint32_t>(opts.width),
                                      .max_height = static_cast<std::uint32_t>(opts.height)});
    auto visualizer = graph.op<viz::HolovizOp>(
        "holoviz", std::move(layers),
        viz::WindowSinkDesc{.title = "PCL Filters: voxels above the ground",
                            .width = static_cast<std::uint32_t>(opts.width),
                            .height = static_cast<std::uint32_t>(opts.height)});
    // Best effort: the display keeps the newest frame and never holds back the filters.
    const holoscan::ConnectionOptions latest{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                             .queue_depth = 1U};
    graph.add_flow(voxels->filtered_points, project->points, latest);
    graph.add_flow(voxels->filtered_colors, project->colors, latest);
    graph.add_flow(project->color, visualizer->image_layer_input(0), latest);
    graph.add_flow(project->depth, discard->image, latest);
    on_gpu("project", "depth");
    on_gpu("project", "color");
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
  const app::FilterReport& r = *report;
  std::cout << std::fixed << std::setprecision(2) << kAppName << ": frames=" << r.frames << '/'
            << opts.count << " ground_left=" << 100.0 * r.fraction(r.ground_left, r.ground)
            << "% objects_kept=" << 100.0 * r.fraction(r.objects_kept, r.objects)
            << "% outliers_left=" << 100.0 * r.fraction(r.outliers_left, r.outliers)
            << "% voxels_per_frame=" << (r.frames == 0 ? 0U : r.voxels / r.frames)
            << " wrong_voxel_counts=" << r.wrong_voxel_counts << std::setprecision(4)
            << " plane: max_angle_deg=" << r.max_plane_angle_deg
            << " max_offset_m=" << r.max_plane_offset_m << " wrong_planes=" << r.wrong_planes;
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
  pf::validate_config(pf::VoxelDownsampleConfig{.voxel_size = opts.voxel_size});
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
