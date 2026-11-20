// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// Height Map: a synthetic depth camera drives over a course with a ramp,
// steps, a box and a pit. holoscan-point-cloud's DepthToPointCloudOp
// deprojects each depth image with the camera's info, which
// holoscan-messages' CameraInfoSourceOp reads from a camera_info file written
// at run time; holoscan-height-map's HeightMapOp maps the PointCloud
// messages, with the camera's pose, into an elevation map that follows the
// camera, and TraversabilityOp turns it into slope, step height and a cost.
// A reporter checks every map against the course's truth, and with a display
// HoloViz shows the elevation beside the cost. The graph runs on one worker.

#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/height_map/height_map_op.hpp>
#include <holoscan/height_map/traversability_op.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/messages/camera_info_input.hpp>
#include <holoscan/messages/camera_info_source_op.hpp>
#include <holoscan/messages/camera_info_yaml.hpp>
#include <holoscan/point_cloud/depth_to_point_cloud_op.hpp>
#include <holoscan/time/realtime_clock.hpp>
#if HEIGHT_MAP_ENABLE_VISUALIZATION
#include <holoscan/visualization/holoviz_op.hpp>
#endif

#include "render_check_op.hpp"
#include "report_op.hpp"
#include "terrain_camera_op.hpp"

namespace {

namespace app = height_map_app;
namespace pc = holoscan::point_cloud;
namespace hm = holoscan::height_map;
namespace msg = holoscan::messages;

constexpr std::string_view kAppName = "height_map";

enum class DisplayMode { kAuto, kOn, kOff, kOffscreen };

struct Options {
  std::uint32_t count{300};
  std::uint32_t period_ms{33};
  int width{640};
  int height{480};
  int map_cells{200};
  float resolution{0.04F};
  hm::ElevationMode elevation{hm::ElevationMode::kFused};
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
    } else if (option == "--map-cells") {
      opts.map_cells = parse_number<int>(value, option);
    } else if (option == "--resolution") {
      opts.resolution = parse_number<float>(value, option);
    } else if (option == "--elevation") {
      if (value == "fused") {
        opts.elevation = hm::ElevationMode::kFused;
      } else if (value == "mean") {
        opts.elevation = hm::ElevationMode::kMean;
      } else if (value == "max") {
        opts.elevation = hm::ElevationMode::kMax;
      } else {
        throw std::invalid_argument("--elevation must be fused, mean or max");
      }
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
  std::cout
      << "Usage: " << argv0
      << " [options]\n"
         "  --count N                Frames (default: 300)\n"
         "  --period-ms N            Camera period, 1..1000 ms (default: 33)\n"
         "  --width N --height N     Depth image size, 64x48 to 4096x4096 (default: 640x480)\n"
         "  --map-cells N            Map side in cells, 32..1024 (default: 200)\n"
         "  --resolution M           Cell side in meters, 0.01..0.2 (default: 0.04)\n"
         "  --elevation fused|mean|max  The elevation layer (default: fused)\n"
         "  --display auto|true|false|offscreen\n"
         "                           Show the elevation and cost in HoloViz (default: auto,\n"
         "                           on when a display is available)\n"
         "  --timeout-seconds N      Allowance beyond the expected run time (default: 120)\n"
         "  --compile-only true|false  Build and check the graph without running it\n";
}

bool env_set(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && *value != '\0';
}

using SignalDomain = holoscan::expected<holoscan::ManagedShutdownSignals, holoscan::Error>;

// Closes the signal domain, if there is one, on every way out of main().
class CloseSignals {
 public:
  explicit CloseSignals(SignalDomain& signals) noexcept : signals_(signals) {}
  ~CloseSignals() {
    if (!signals_) {
      return;
    }
    if (auto closed = signals_->close(); !closed) {
      std::cerr << kAppName << ": warning: closing the signal domain failed: "
                << holoscan::runtime_error_message(closed.error()) << '\n';
    }
  }
  CloseSignals(const CloseSignals&) = delete;
  CloseSignals& operator=(const CloseSignals&) = delete;

 private:
  SignalDomain& signals_;
};

// The synthetic camera's camera_info file, for CameraInfoSourceOp to read, in
// a folder of its own that mkdtemp() names unpredictably, so that nothing can
// be planted where the file goes. Removed with this object, on every way out
// of the run: a pass, a failure, an error, or a stop request (SIGINT,
// SIGTERM), which ends the run normally.
class CameraInfoFile {
 public:
  explicit CameraInfoFile(const msg::CameraInfo& camera) {
    std::string folder =
        (std::filesystem::temp_directory_path() / (std::string(kAppName) + "_XXXXXX")).string();
    if (::mkdtemp(folder.data()) == nullptr) {
      throw std::system_error(errno, std::generic_category(),
                              "cannot make a temporary folder for the camera_info file");
    }
    folder_ = folder;
    path_ = folder_ / "camera_info.yaml";
    try {
      msg::write_camera_info_yaml(camera, path_);
    } catch (...) {
      remove();
      throw;
    }
  }
  ~CameraInfoFile() { remove(); }
  CameraInfoFile(const CameraInfoFile&) = delete;
  CameraInfoFile& operator=(const CameraInfoFile&) = delete;

  [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

 private:
  void remove() noexcept {
    std::error_code ignored;
    std::filesystem::remove_all(folder_, ignored);
  }

  std::filesystem::path folder_;
  std::filesystem::path path_;
};

int execute(const Options& opts, SignalDomain& signals) {
  const bool display_available =
      env_set("DISPLAY") || (env_set("WAYLAND_DISPLAY") && env_set("XDG_RUNTIME_DIR"));
  const bool display =
      opts.display == DisplayMode::kOn ||
      (opts.display == DisplayMode::kAuto && display_available && HEIGHT_MAP_ENABLE_VISUALIZATION);
  const bool offscreen = opts.display == DisplayMode::kOffscreen;
  if ((display || offscreen) && !HEIGHT_MAP_ENABLE_VISUALIZATION) {
    throw std::invalid_argument("--display needs a build with visualization enabled");
  }
  if (display && !display_available && !opts.compile_only) {
    throw std::invalid_argument(
        "the display needs a display server; set DISPLAY or add --display false");
  }
  const bool images = display || offscreen;

  const hm::TerrainCamera camera = app::camera_for_frame(opts.width, opts.height, 0);
  const auto points =
      static_cast<std::uint32_t>(opts.width) * static_cast<std::uint32_t>(opts.height);
  // A window of map_cells x map_cells around the camera, cells on the lattice through the
  // map frame's origin.
  const hm::MapGeometry geometry{.resolution = opts.resolution,
                                 .rows = opts.map_cells,
                                 .cols = opts.map_cells,
                                 .origin_x = 0.0,
                                 .origin_y = 0.0};

  holoscan::Graph graph{std::string(kAppName)};
  auto finished = std::make_shared<std::atomic<std::uint32_t>>(0U);
  auto source = graph.op<app::TerrainCameraOp>(
      "camera", app::TerrainCameraConfig{.width = opts.width,
                                         .height = opts.height,
                                         .count = opts.count,
                                         .period = std::chrono::milliseconds(opts.period_ms),
                                         .finished = finished});
  // The camera's calibration, from the file CameraInfoSourceOp reads; the run
  // reads it again when it starts, so the file stays until the run is over.
  const CameraInfoFile camera_info_file(hm::camera_info_of(camera, app::kCameraFrame));
  auto camera_info = graph.op<msg::CameraInfoSourceOp>(
      "camera_info", msg::CameraInfoSourceConfig{.camera_info_path = camera_info_file.path()});
  auto deproject = graph.op<pc::DepthToPointCloudOp>(
      "deproject", pc::DepthToPointCloudConfig{.depth_scale = 1.0F,
                                               .depth_min = 0.1F,
                                               .depth_max = camera.max_depth,
                                               .max_width = opts.width,
                                               .max_height = opts.height});
  hm::HeightMapConfig map_config;
  map_config.map.geometry = geometry;
  map_config.map.mode = hm::MapMode::kFollow;
  map_config.map.elevation = opts.elevation;
  // The camera's noise: 1 mm plus 0.2 % of the squared range.
  map_config.map.noise_base = 0.001F;
  map_config.map.noise_quadratic = 0.002F;
  map_config.map.min_height = -1.0F;
  map_config.map.max_height = 1.0F;
  map_config.pose_input = true;
  map_config.cloud_frame_id = app::kCameraFrame;
  map_config.outputs.variance = false;
  map_config.outputs.image = images;
  map_config.frame_id = "map";
  // The elevation's colors: the course's range, the pit's floor to the ramp's top.
  map_config.image_low = -0.2F;
  map_config.image_high = 0.4F;
  map_config.max_points = points;
  auto mapper = graph.op<hm::HeightMapOp>("height_map", map_config);
  hm::TraversabilityConfig cost_config;
  cost_config.components = true;
  cost_config.image = images;
  cost_config.frame_id = "map";
  cost_config.max_rows = opts.map_cells;
  cost_config.max_cols = opts.map_cells;
  auto traversability = graph.op<hm::TraversabilityOp>("traversability", cost_config);
  auto report = std::make_shared<app::Report>();
  if (opts.elevation == hm::ElevationMode::kMax) {
    report->max_median_error_m = 0.02;
    report->max_p95_error_m = 0.04;
  }
  if (opts.elevation != hm::ElevationMode::kFused) {
    report->min_lethal_rim = 0.5;
  }
  auto reporter = graph.op<app::ReportOp>("report", app::ReportConfig{.report = report,
                                                                      .camera_width = opts.width,
                                                                      .camera_height = opts.height,
                                                                      .geometry = geometry,
                                                                      .expected_frames = opts.count,
                                                                      .finished = finished});

  // Lossless edges: every frame is mapped, costed and checked, in order. The
  // camera keeps one frame in flight, so no queue fills. The camera info is
  // the newest one, kept by the deprojection; its edge never holds back the
  // source.
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(source->depth, deproject->depth, lossless);
  graph.add_flow(camera_info->camera_info, deproject->camera_info,
                 msg::latest_camera_info_connection());
  graph.add_flow(deproject->point_cloud, mapper->point_cloud, lossless);
  graph.add_flow(source->pose, mapper->pose, lossless);
  graph.add_flow(mapper->elevation, traversability->elevation, lossless);
  graph.add_flow(mapper->map_info, traversability->map_info, lossless);
  graph.add_flow(mapper->elevation, reporter->elevation, lossless);
  graph.add_flow(mapper->map_info, reporter->map_info, lossless);
  graph.add_flow(traversability->cost, reporter->cost, lossless);
  graph.add_flow(traversability->unknown, reporter->unknown, lossless);
  graph.add_flow(traversability->slope, reporter->slope, lossless);
  graph.add_flow(traversability->step_height, reporter->step_height, lossless);
  graph.add_flow(traversability->roughness, reporter->roughness, lossless);

  holoscan::CompileOptions compile_options;
  const auto on_gpu = [&compile_options](std::string op, std::string port) {
    compile_options.deployment.bind_tensor_output_device(
        holoscan::TensorOutputDevicePlacement{.operator_path = std::move(op),
                                              .output_port = std::move(port),
                                              .device = holoscan::DeviceId{0}});
  };
  on_gpu("camera", "depth");
  on_gpu("deproject", "point_cloud");
  on_gpu("height_map", "elevation");
  for (const char* port : {"cost", "unknown", "slope", "step_height", "roughness"}) {
    on_gpu("traversability", port);
  }

#if HEIGHT_MAP_ENABLE_VISUALIZATION
  auto rendered = std::make_shared<app::RenderTally>();
  if (images) {
    namespace viz = holoscan::visualization;
    const auto side = static_cast<std::uint32_t>(opts.map_cells);
    viz::LayerStack layers;
    // The elevation on the left half, the cost on the right.
    layers.add_image_layer(
        "elevation",
        viz::ImageLayerDesc{
            .format = viz::ImageFormat::kR8G8B8Unorm,
            .max_width = side,
            .max_height = side,
            .view = viz::ViewTransform{
                .offset_x = 0.0F, .offset_y = 1.0F, .scale_x = 0.5F, .scale_y = -1.0F}});
    layers.add_image_layer(
        "cost", viz::ImageLayerDesc{
                    .format = viz::ImageFormat::kR8G8B8Unorm,
                    .max_width = side,
                    .max_height = side,
                    .view = viz::ViewTransform{
                        .offset_x = 0.5F, .offset_y = 1.0F, .scale_x = 0.5F, .scale_y = -1.0F}});
    constexpr std::uint32_t kViewSide = 512U;
    viz::SinkDesc sink = viz::WindowSinkDesc{
        .title = "Height Map: elevation | cost", .width = 2U * kViewSide, .height = kViewSide};
    if (offscreen) {
      sink = viz::CudaSinkDesc{.width = 2U * kViewSide, .height = kViewSide};
    }
    auto visualizer = graph.op<viz::HolovizOp>("holoviz", std::move(layers), std::move(sink));
    // Best effort: the display keeps the newest frame and never holds back the pipeline.
    const holoscan::ConnectionOptions latest{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                             .queue_depth = 1U};
    graph.add_flow(mapper->elevation_image, visualizer->image_layer_input(0), latest);
    graph.add_flow(traversability->cost_image, visualizer->image_layer_input(1), latest);
    on_gpu("height_map", "elevation_image");
    on_gpu("traversability", "cost_image");
    if (offscreen) {
      auto check = graph.op<app::RenderCheckOp>("render_check", rendered);
      graph.add_flow(visualizer->output(), check->frame, lossless);
      on_gpu("holoviz", "output");
    }
  }
#endif

  // One worker for the whole graph, as holoscan-messages' README advises for a
  // lossless chain: with the camera info source the graph has two sources,
  // forks and joins, and the automatic partitioning would split it onto six
  // workers (seven with the display). There a producer runs while its
  // consumer's queue or its output's pool is full, and an operator in the
  // middle cannot send the refused output again: the run fails.
  graph.partition(std::string(kAppName)).include_all();
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  const auto plan = holoscan::compile(graph, std::move(compile_options));
  // Warnings on an accepted plan, as well as the reasons for a rejected one.
  if (!plan.diagnostics().empty()) {
    std::cerr << holoscan::format_diagnostics(plan.diagnostics(), plan.graph()) << '\n';
  }
  if (!plan.ok()) {
    return 1;
  }
  if (opts.compile_only) {
    // The workers the graph runs on: one per partition that runs operators.
    std::cout << kAppName << ": plan=OK workers=" << plan.scheduler_plan().worker_count << '\n';
    return 0;
  }
  if (!signals) {
    std::cerr << kAppName << ": warning: running without managed SIGINT/SIGTERM handling: "
              << holoscan::runtime_error_message(signals.error())
              << "; an interrupt ends the process without a report\n";
  }
  auto session = signals ? holoscan::run_async(plan, *signals) : holoscan::run_async(plan);
  const bool timed_out =
      !session.wait_for(std::chrono::seconds(opts.timeout_seconds) +
                        std::chrono::milliseconds(2LL * (opts.count + 1U) * opts.period_ms));
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

  const bool stopped = !timed_out && termination == holoscan::RunTermination::kStopped;
#if HEIGHT_MAP_ENABLE_VISUALIZATION
  // Offscreen, HoloViz must have shown both maps; it may skip frames, never all.
  const bool shown = !offscreen || rendered->showing != 0;
#else
  const bool shown = true;
#endif
  const bool passed = !timed_out && session.diagnostics().empty() &&
                      termination == holoscan::RunTermination::kRequestedCompleted &&
                      report->frames == opts.count && report->passed() && shown;
  const app::Report& r = *report;
  std::cout << std::fixed << std::setprecision(4) << kAppName << ": frames=" << r.frames << '/'
            << opts.count << " elevation_median_m=" << r.worst_median_error_m
            << " elevation_p95_m=" << r.worst_p95_error_m
            << " slope_error_deg=" << r.worst_slope_error_deg
            << " step_error_m=" << r.worst_step_error_m
            << " flat_cost_p95=" << r.worst_flat_cost_p95 << " lethal_rim=" << r.worst_lethal_rim
            << " known_share=" << r.least_known_share << " window_wrong=" << r.window_wrong
            << " mask_wrong=" << r.mask_wrong;
  if (r.frame_gaps != 0) {
    std::cout << " frame_gaps=" << r.frame_gaps;
  }
#if HEIGHT_MAP_ENABLE_VISUALIZATION
  if (offscreen) {
    std::cout << " rendered=" << rendered->frames << " showing=" << rendered->showing;
  }
#endif
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
  if (opts.map_cells < 32 || opts.map_cells > 1024) {
    throw std::invalid_argument("--map-cells must be 32..1024");
  }
  if (!(opts.resolution >= 0.01F && opts.resolution <= 0.2F)) {
    throw std::invalid_argument("--resolution must be 0.01..0.2");
  }
  app::validate_config(
      app::TerrainCameraConfig{.width = opts.width,
                               .height = opts.height,
                               .count = opts.count,
                               .period = std::chrono::milliseconds(opts.period_ms)});
}

}  // namespace

int main(int argc, char** argv) try {
  // Take over SIGINT and SIGTERM while this is the only thread; the SDK
  // refuses a process that already runs others.
  SignalDomain signals = holoscan::ManagedShutdownSignals::create();
  const CloseSignals close_signals(signals);
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
