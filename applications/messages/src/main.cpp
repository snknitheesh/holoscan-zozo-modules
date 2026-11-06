// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The holoscan-messages demo: writes a stereo calibration with known values as
// the two camera_info YAML files a ROS stereo calibration writes, sends it
// with CameraInfoSourceOp to an operator that takes stereo frames and the
// camera info with the recommended contract and checks every field and the
// recovered extrinsics, and sends synthetic point clouds with points, colors
// and normals through the graph, checking every value. Exit status 0 only
// when everything matched.

#include <stdlib.h>

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/messages/camera_info_source_op.hpp>
#include <holoscan/messages/synthetic_calibration.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "demo_ops.hpp"
#include "ros_files.hpp"

namespace {

namespace fs = std::filesystem;
namespace hm = holoscan::messages;
namespace demo = messages_demo;
using std::chrono_literals::operator""ms;

constexpr std::string_view kAppName = "messages";
// How long the run may go without a frame before it counts as hung.
constexpr auto kPatience = std::chrono::seconds{60};

struct Options {
  std::uint32_t count{300};
  std::uint32_t period_ms{10};
  std::uint32_t info_period_ms{100};
  holoscan::MemoryKind cloud_memory{holoscan::MemoryKind::kCudaDevice};
  bool compile_only{};
  bool help{};
};

std::uint32_t parse_number(std::string_view value, std::string_view option, std::uint32_t low,
                           std::uint32_t high) {
  std::uint32_t result = 0;
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
  if (error != std::errc() || end != value.data() + value.size() || result < low || result > high) {
    throw std::invalid_argument(std::string(option) + " takes a number " + std::to_string(low) +
                                ".." + std::to_string(high) + ", not '" + std::string(value) + "'");
  }
  return result;
}

bool parse_bool(std::string_view value, std::string_view option) {
  if (value == "true") return true;
  if (value == "false") return false;
  throw std::invalid_argument(std::string(option) + " takes true or false");
}

Options parse_args(int argc, char** argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    const std::string_view option{argv[i]};
    if (option == "--help" || option == "-h") {
      opts.help = true;
      return opts;
    }
    if (i + 1 >= argc) throw std::invalid_argument("missing value for " + std::string(option));
    const std::string_view value{argv[++i]};
    if (option == "--count") {
      opts.count = parse_number(value, option, 1, 1000000);
    } else if (option == "--period-ms") {
      opts.period_ms = parse_number(value, option, 1, 1000);
    } else if (option == "--info-period-ms") {
      opts.info_period_ms = parse_number(value, option, 1, 10000);
    } else if (option == "--cloud-memory") {
      if (value != "device" && value != "host") {
        throw std::invalid_argument("--cloud-memory takes device or host");
      }
      opts.cloud_memory =
          value == "host" ? holoscan::MemoryKind::kHost : holoscan::MemoryKind::kCudaDevice;
    } else if (option == "--compile-only") {
      opts.compile_only = parse_bool(value, option);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(option));
    }
  }
  return opts;
}

void usage(const char* argv0) {
  std::cout << "Usage: " << argv0
            << " [options]\n"
               "  --count N               Stereo frames and point clouds (default: 300)\n"
               "  --period-ms N           Their period, 1..1000 ms (default: 10)\n"
               "  --info-period-ms N      CameraInfoSourceOp's period, 1..10000 ms (default: 100)\n"
               "  --cloud-memory WHERE    device or host (default: device)\n"
               "  --compile-only BOOL     Build and check the graph, print its workers and exit\n"
               "                          without running it (default: false)\n";
}

// A folder for the calibration files, which mkdtemp() names unpredictably
// and makes private, so that nothing can be planted where the files go;
// removed with everything in it on every path out of the run.
class TemporaryFolder {
 public:
  TemporaryFolder() {
    std::string pattern = (fs::temp_directory_path() / "holoscan_messages_demo_XXXXXX").string();
    if (::mkdtemp(pattern.data()) == nullptr) {
      throw std::system_error(errno, std::generic_category(),
                              "cannot make a temporary folder for the calibration files");
    }
    path_ = pattern;
  }
  ~TemporaryFolder() {
    std::error_code ignored;
    fs::remove_all(path_, ignored);
  }
  TemporaryFolder(const TemporaryFolder&) = delete;
  TemporaryFolder& operator=(const TemporaryFolder&) = delete;
  [[nodiscard]] const fs::path& path() const noexcept { return path_; }

 private:
  fs::path path_;
};

using SignalDomain = holoscan::expected<holoscan::ManagedShutdownSignals, holoscan::Error>;

// Closes the signal domain, if there is one, on every way out of main().
class CloseSignals {
 public:
  explicit CloseSignals(SignalDomain& signals) noexcept : signals_(signals) {}
  ~CloseSignals() {
    if (!signals_) return;
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

int execute(const Options& opts, SignalDomain& signals) {
  // The calibration, written as ROS files: the rectified cameras' focal
  // length and principal point are the left camera's. The files hold K and D
  // with the calibrator's decimals, which the checker expects.
  const hm::StereoCameraInfo truth =
      hm::synthetic_stereo_camera_info(hm::DistortionModel::kPlumbBob);
  const TemporaryFolder folder;
  const fs::path left_file = folder.path() / "left.yaml";
  const fs::path right_file = folder.path() / "right.yaml";
  demo::write_ros_stereo_files(truth, truth.left.fy, truth.left.cx, truth.left.cy, left_file,
                               right_file);
  hm::StereoCameraInfo as_written = truth;
  as_written.left = demo::as_written(truth.left);
  as_written.right = demo::as_written(truth.right);

  auto report = std::make_shared<demo::Report>();
  const auto period = std::chrono::milliseconds(opts.period_ms);
  const holoscan::ConnectionOptions lossless{.queue_policy = holoscan::QueuePolicy::kBlocking};
  holoscan::Graph graph{std::string(kAppName)};
  auto frames =
      graph.op<demo::FrameSourceOp>("frames", opts.count, std::chrono::nanoseconds(period));
  auto camera_info = graph.op<hm::CameraInfoSourceOp>(
      "camera_info",
      hm::CameraInfoSourceConfig{.camera_info_path = left_file,
                                 .right_camera_info_path = right_file,
                                 .period = std::chrono::milliseconds(opts.info_period_ms)});
  auto check = graph.op<demo::CalibrationCheckOp>(
      "calibration_check",
      demo::CalibrationCheckConfig{.truth = as_written, .frames = opts.count, .report = report});
  graph.add_flow(frames->left, check->left, lossless);
  graph.add_flow(frames->right, check->right, lossless);
  graph.add_flow(camera_info->stereo_camera_info, check->camera_info,
                 hm::latest_camera_info_connection());

  const demo::CloudConfig cloud_config{
      .count = opts.count, .period = period, .memory = opts.cloud_memory, .report = report};
  auto clouds = graph.op<demo::CloudSourceOp>("clouds", cloud_config);
  auto cloud_check = graph.op<demo::CloudCheckOp>("cloud_check", cloud_config);
  graph.add_flow(clouds->cloud, cloud_check->cloud, lossless);
  // One worker for the whole graph, as the module's README advises for
  // lossless chains (every frame and cloud here is checked). With three
  // sources (the frames, the camera info and the clouds), the automatic
  // partitioning would give the frame source, the camera info source, the
  // calibration check, and the clouds with their check a worker each, and the
  // sources would then run while their checkers' queues or the clouds' pool
  // slots are full, and have what they send refused (and send it again).
  graph.partition(kAppName).include_all();
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions compile_options;
  if (opts.cloud_memory == holoscan::MemoryKind::kCudaDevice) {
    compile_options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "clouds", .output_port = "cloud", .device = holoscan::DeviceId{0}});
  }
  const auto plan = holoscan::compile(graph, std::move(compile_options));
  if (!plan.diagnostics().empty()) {
    std::cerr << holoscan::format_diagnostics(plan.diagnostics(), plan.graph()) << '\n';
  }
  if (!plan.ok()) return 1;
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
  std::uint32_t seen = 0;
  auto last_progress = std::chrono::steady_clock::now();
  bool timed_out = false;
  while (session.is_running()) {
    std::this_thread::sleep_for(10ms);
    const auto now = std::chrono::steady_clock::now();
    if (const std::uint32_t done = report->frames.load() + report->clouds.load(); done != seen) {
      seen = done;
      last_progress = now;
    } else if (now - last_progress >= kPatience) {
      timed_out = true;
      break;
    }
  }
  if (timed_out) {
    std::cerr << kAppName << ": timed out: nothing arrived for " << kPatience.count() << " s\n";
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

  // CameraInfoSourceOp sends first when the run starts; at most the first
  // frame can come before it.
  const std::uint32_t with_info = report->frames_with_camera_info.load();
  const bool stopped = !timed_out && termination == holoscan::RunTermination::kStopped;
  const bool passed = !timed_out && session.diagnostics().empty() &&
                      termination == holoscan::RunTermination::kRequestedCompleted &&
                      report->frames == opts.count && with_info + 1 >= opts.count &&
                      report->wrong_camera_info == 0 && report->clouds == opts.count &&
                      report->wrong_clouds == 0;
  std::cout << kAppName << ": frames=" << report->frames.load() << '/' << opts.count
            << " with_camera_info=" << with_info
            << " wrong_camera_info=" << report->wrong_camera_info.load() << std::scientific
            << std::setprecision(2) << " extrinsics_error=" << report->max_extrinsics_error.load()
            << " clouds=" << report->clouds.load() << '/' << opts.count
            << " wrong_clouds=" << report->wrong_clouds.load() << '\n'
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
  const CloseSignals close_signals(signals);
  const Options opts = parse_args(argc, argv);
  if (opts.help) {
    usage(argv[0]);
    return 0;
  }
  return execute(opts, signals);
} catch (const std::invalid_argument& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 2;
} catch (const std::exception& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 1;
}
