// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// TSDF Reconstruction: a synthetic RGB-D camera circles a known tabletop
// scene with known poses; holoscan-tsdf's TsdfIntegrateOp integrates every
// frame into a TSDF volume, and TsdfMeshOp ray casts the volume from each
// pose and extracts its mesh every N frames. Reporters check every ray cast
// against the scene's exact depth and every mesh against its exact surfaces,
// and with a display HoloViz shows the input depth beside the shaded ray
// cast. At the end the mesh can be saved as a PLY file.

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
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/tsdf/synthetic_scene.hpp>
#include <holoscan/tsdf/tsdf_integrate_op.hpp>
#include <holoscan/tsdf/tsdf_mesh_op.hpp>
#include <holoscan/tsdf/tsdf_volume.hpp>
#if TSDF_RECONSTRUCTION_ENABLE_VISUALIZATION
#include <holoscan/visualization/holoviz_op.hpp>
#endif

#include "render_check_op.hpp"
#include "report_op.hpp"
#include "scene_camera_op.hpp"
#include "view_op.hpp"

namespace {

using std::chrono_literals::operator""ms;
namespace app = tsdf_reconstruction_app;
namespace ht = holoscan::tsdf;

constexpr std::string_view kAppName = "tsdf_reconstruction";
constexpr float kDepthMin = 0.1F;
constexpr float kDepthMax = 3.0F;

enum class DisplayMode { kAuto, kOn, kOff, kOffscreen };

struct Options {
  std::uint32_t count{240};
  std::uint32_t orbit{120};
  std::uint32_t period_ms{33};
  int width{640};
  int height{480};
  int dims{256};
  std::uint32_t mesh_every{30};
  bool millimeters{true};
  std::string ply;
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

bool parse_bool(std::string_view value, std::string_view option) {
  if (value != "true" && value != "false") {
    throw std::invalid_argument(std::string(option) + " requires true or false");
  }
  return value == "true";
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
    } else if (option == "--orbit") {
      opts.orbit = parse_number<std::uint32_t>(value, option);
    } else if (option == "--period-ms") {
      opts.period_ms = parse_number<std::uint32_t>(value, option);
    } else if (option == "--width") {
      opts.width = parse_number<int>(value, option);
    } else if (option == "--height") {
      opts.height = parse_number<int>(value, option);
    } else if (option == "--dims") {
      opts.dims = parse_number<int>(value, option);
    } else if (option == "--mesh-every") {
      opts.mesh_every = parse_number<std::uint32_t>(value, option);
    } else if (option == "--depth") {
      if (value != "mm" && value != "m") {
        throw std::invalid_argument("--depth must be mm (uint16) or m (float32)");
      }
      opts.millimeters = value == "mm";
    } else if (option == "--ply") {
      opts.ply = std::string(value);
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
         "  --count N                Frames (default: 240)\n"
         "  --orbit N                Frames per orbit of the scene (default: 120)\n"
         "  --period-ms N            Camera period, 1..1000 ms (default: 33)\n"
         "  --width N --height N     Image size, 64x48 to 4096x4096 (default: 640x480)\n"
         "  --dims N                 Volume of N^3 voxels over 1 m, 32..512 (default: 256)\n"
         "  --mesh-every N           Extract the mesh every N frames, 1..count (default: 30)\n"
         "  --depth mm|m             uint16 millimeters or float32 meters (default: mm)\n"
         "  --ply PATH               Save the final mesh as a PLY file\n"
         "  --display auto|true|false|offscreen\n"
         "                           Show the depth and the ray cast in HoloViz (default: auto,\n"
         "                           on when a display is available)\n"
         "  --timeout-seconds N      Allowance beyond the expected run time (default: 120)\n"
         "  --compile-only true|false  Build and check the graph without running it\n";
}

bool env_set(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && *value != '\0';
}

void validate(const Options& opts) {
  if (opts.period_ms == 0 || opts.period_ms > 1000 || opts.timeout_seconds == 0) {
    throw std::invalid_argument("--period-ms must be 1..1000 and --timeout-seconds positive");
  }
  if (opts.dims < 32 || opts.dims > 512) {
    throw std::invalid_argument("--dims must be 32..512");
  }
  if (opts.mesh_every == 0 || opts.mesh_every > opts.count) {
    throw std::invalid_argument("--mesh-every must be 1..count");
  }
  app::validate_config(app::SceneCameraConfig{.width = opts.width,
                                              .height = opts.height,
                                              .count = opts.count,
                                              .orbit = opts.orbit,
                                              .period = std::chrono::milliseconds(opts.period_ms)});
}

using SignalDomain = holoscan::expected<holoscan::ManagedShutdownSignals, holoscan::Error>;

int execute(const Options& opts, SignalDomain& signals) {
  const bool display_available =
      env_set("DISPLAY") || (env_set("WAYLAND_DISPLAY") && env_set("XDG_RUNTIME_DIR"));
  const bool display = opts.display == DisplayMode::kOn ||
                       (opts.display == DisplayMode::kAuto && display_available &&
                        TSDF_RECONSTRUCTION_ENABLE_VISUALIZATION);
  const bool offscreen = opts.display == DisplayMode::kOffscreen;
  if ((display || offscreen) && !TSDF_RECONSTRUCTION_ENABLE_VISUALIZATION) {
    throw std::invalid_argument("--display needs a build with visualization enabled");
  }
  if (display && !display_available && !opts.compile_only) {
    throw std::invalid_argument(
        "the display needs a display server; set DISPLAY or add --display false");
  }

  // The volume, shared by the two TSDF operators: N^3 voxels over the
  // tabletop scene's metre, with color.
  const ht::TsdfVolumeConfig volume_config =
      ht::scene_volume(ht::SceneKind::kTabletop, opts.dims, true);
  auto volume = std::make_shared<ht::TsdfVolume>(volume_config);
  std::cout << kAppName << ": volume " << opts.dims << "^3, " << std::fixed << std::setprecision(2)
            << 1000.0F * volume_config.voxel_size << " mm voxels, " << (volume->bytes() >> 20)
            << " MiB on the GPU\n";
  const ht::CameraIntrinsics intrinsics = ht::synthetic_intrinsics(opts.width, opts.height);
  // The mesh's capacity: the scene's mesh has about 3.5 dims^2 vertices (900
  // thousand at 512^3).
  const std::uint32_t max_vertices = std::max(
      1U << 16, 6U * static_cast<std::uint32_t>(opts.dims) * static_cast<std::uint32_t>(opts.dims));
  const std::uint32_t max_triangles = 2U * max_vertices;

  holoscan::Graph graph{std::string(kAppName)};
  auto finished = std::make_shared<std::atomic<std::uint32_t>>(0U);
  auto camera = graph.op<app::SceneCameraOp>(
      "camera", app::SceneCameraConfig{.width = opts.width,
                                       .height = opts.height,
                                       .count = opts.count,
                                       .orbit = opts.orbit,
                                       .millimeters = opts.millimeters,
                                       .period = std::chrono::milliseconds(opts.period_ms),
                                       .finished = finished});
  auto integrate = graph.op<ht::TsdfIntegrateOp>(
      "integrate", ht::TsdfIntegrateConfig{.volume = volume,
                                           .intrinsics = intrinsics,
                                           .depth_scale = opts.millimeters ? 0.001F : 1.0F,
                                           .depth_min = kDepthMin,
                                           .depth_max = kDepthMax,
                                           .color = true,
                                           .max_width = opts.width,
                                           .max_height = opts.height});
  auto mesh = graph.op<ht::TsdfMeshOp>(
      "mesh", ht::TsdfMeshConfig{.volume = volume,
                                 .mesh = true,
                                 .mesh_every = opts.mesh_every,
                                 .max_vertices = max_vertices,
                                 .max_triangles = max_triangles,
                                 .raycast = true,
                                 .raycast_params = ht::RaycastParams{.intrinsics = intrinsics,
                                                                     .width = opts.width,
                                                                     .height = opts.height,
                                                                     .depth_min = kDepthMin,
                                                                     .depth_max = kDepthMax}});
  auto report = std::make_shared<app::Report>();
  report->expected_frames = opts.count;
  report->expected_meshes = opts.count / opts.mesh_every;
  report->voxel = volume_config.voxel_size;
  const app::ReportConfig report_config{.report = report,
                                        .volume = volume_config,
                                        .width = opts.width,
                                        .height = opts.height,
                                        .finished = finished};
  auto raycast_report = graph.op<app::RaycastReportOp>("raycast_report", report_config);
  auto mesh_report = graph.op<app::MeshReportOp>("mesh_report", report_config);

  // Lossless edges: every frame is integrated, ray cast and checked, and
  // every mesh checked, in order. The camera keeps one frame in flight, so no
  // queue fills.
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  graph.add_flow(camera->depth, integrate->depth, lossless);
  graph.add_flow(camera->color, integrate->color, lossless);
  graph.add_flow(camera->pose, integrate->pose, lossless);
  graph.add_flow(integrate->integrated_pose, mesh->pose, lossless);
  graph.add_flow(integrate->integrated_pose, raycast_report->pose, lossless);
  graph.add_flow(mesh->raycast_depth, raycast_report->depth, lossless);
  graph.add_flow(mesh->raycast_normals, raycast_report->normals, lossless);
  graph.add_flow(mesh->raycast_colors, raycast_report->colors, lossless);
  graph.add_flow(mesh->vertices, mesh_report->vertices, lossless);
  graph.add_flow(mesh->normals, mesh_report->normals, lossless);
  graph.add_flow(mesh->colors, mesh_report->colors, lossless);
  graph.add_flow(mesh->triangles, mesh_report->triangles, lossless);
  graph.add_flow(mesh->mesh_counts, mesh_report->counts, lossless);

  holoscan::CompileOptions compile_options;
  const auto on_gpu = [&compile_options, &volume](std::string op, std::string port) {
    compile_options.deployment.bind_tensor_output_device(
        holoscan::TensorOutputDevicePlacement{.operator_path = std::move(op),
                                              .output_port = std::move(port),
                                              .device = holoscan::DeviceId{volume->device()}});
  };
  on_gpu("camera", "depth");
  on_gpu("camera", "color");
  for (const char* port : {"vertices", "normals", "colors", "triangles", "mesh_counts",
                           "raycast_depth", "raycast_normals", "raycast_colors"}) {
    on_gpu("mesh", port);
  }

  auto rendered = std::make_shared<app::RenderTally>();
#if TSDF_RECONSTRUCTION_ENABLE_VISUALIZATION
  if (display || offscreen) {
    namespace viz = holoscan::visualization;
    // The input depth on the left, the shaded ray cast on the right.
    auto view = graph.op<app::ViewOp>("view", app::ViewConfig{.width = opts.width,
                                                              .height = opts.height,
                                                              .depth_min = kDepthMin,
                                                              .depth_max = kDepthMax});
    viz::LayerStack layers;
    layers.add_image_layer(
        "view", viz::ImageLayerDesc{.format = viz::ImageFormat::kR8G8B8Unorm,
                                    .max_width = static_cast<std::uint32_t>(2 * opts.width),
                                    .max_height = static_cast<std::uint32_t>(opts.height)});
    // On screen, or offscreen into CUDA memory, where RenderCheckOp looks at each frame.
    viz::SinkDesc sink = viz::WindowSinkDesc{.title = "TSDF Reconstruction: input depth | ray cast",
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
    graph.add_flow(camera->depth, view->depth, latest);
    graph.add_flow(mesh->raycast_normals, view->normals, latest);
    graph.add_flow(mesh->raycast_colors, view->colors, latest);
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
  const auto started = std::chrono::steady_clock::now();
  auto session = signals ? holoscan::run_async(plan, *signals) : holoscan::run_async(plan);
  const auto deadline = started + std::chrono::seconds(opts.timeout_seconds) +
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
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
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

  bool saved = true;
  if (!opts.ply.empty()) {
    try {
      volume->export_ply(opts.ply);
      std::cout << kAppName << ": saved the mesh to " << opts.ply << '\n';
    } catch (const std::exception& error) {
      std::cerr << kAppName << ": " << error.what() << '\n';
      saved = false;
    }
  }

  const bool stopped = !timed_out && termination == holoscan::RunTermination::kStopped;
  // Offscreen, HoloViz must have shown both views; it may skip frames, never all.
  const bool shown = !offscreen || rendered->showing != 0;
  const std::lock_guard lock(report->mutex);
  const app::Report& r = *report;
  const bool passed = !timed_out && session.diagnostics().empty() &&
                      termination == holoscan::RunTermination::kRequestedCompleted && r.passed() &&
                      shown && saved;
  const double voxel_mm = 1000.0 * r.voxel;
  std::cout << std::fixed << std::setprecision(2) << kAppName << ": frames=" << r.frames << '/'
            << opts.count << " fps=" << (seconds > 0.0 ? r.frames / seconds : 0.0)
            << " raycast: worst_median_mm=" << 1000.0 * r.worst_median_m
            << " worst_p95_mm=" << 1000.0 * r.worst_p95_m
            << " worst_outliers=" << 100.0 * r.worst_outliers << '%'
            << " worst_coverage=" << 100.0 * r.worst_coverage << "% bad_hits=" << r.bad_hits
            << " meshes=" << r.meshes << '/' << r.expected_meshes
            << " mesh: vertices=" << r.last_vertices << " triangles=" << r.last_triangles
            << " mean_mm=" << 1000.0 * r.last_mean_m << " p95_mm=" << 1000.0 * r.last_p95_m
            << " max_mm=" << 1000.0 * r.last_max_m << " overflows=" << r.overflows
            << " bad_triangles=" << r.bad_triangles << " bad_normals=" << r.bad_normals
            << " voxel_mm=" << voxel_mm;
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
