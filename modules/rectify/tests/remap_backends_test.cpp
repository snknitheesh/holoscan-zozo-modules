// SPDX-FileCopyrightText: Copyright (c) 2026 zozo
// SPDX-License-Identifier: Apache-2.0
//
// The remap backends on their own: every backend this platform has, for every
// format and interpolation it takes, against the CPU reference; synthetic
// distorted pairs rectified by each, measured against the known geometry;
// the border of a wide view and rows with padding; what auto chooses on a CI
// runner; what each refuses; the host waits per pair; and the time each
// takes per image and per pair.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <holoscan/rectify/backend.hpp>
#include <holoscan/rectify/rectification.hpp>
#include <holoscan/rectify/remapper.hpp>
#include <holoscan/rectify/synthetic_stereo.hpp>

#include "reference_remap.hpp"
#include "test_support.hpp"
#if HOLOSCAN_RECTIFY_TEST_WITH_VPI
#include "vpi_status.hpp"
#endif

namespace holoscan::rectify {
namespace {

void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

// A device buffer holding a packed image.
struct DeviceBuffer {
  void* data{};
  std::size_t bytes{};
  explicit DeviceBuffer(std::size_t size) : bytes(size) { check_cuda(cudaMalloc(&data, size)); }
  ~DeviceBuffer() { cudaFree(data); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

std::optional<std::string> ci_platform() {
  const char* platform = std::getenv("HOLOSCAN_CI_PLATFORM");
  if (platform == nullptr) return std::nullopt;
  return std::string(platform);
}

// Remaps `input` (packed, T per channel) with `remapper` through `queue`, or
// a queue of its own, and returns the output, packed. On the device,
// `padding` bytes follow every row of the input and of the output.
template <typename T>
std::vector<T> run(Remapper& remapper, const std::vector<T>& input, int width, int height,
                   int out_width, int out_height, int channels, cudaStream_t stream,
                   std::size_t padding = 0, RemapQueue* queue = nullptr) {
  const std::size_t in_row = static_cast<std::size_t>(width) * channels * sizeof(T);
  const std::size_t out_row = static_cast<std::size_t>(out_width) * channels * sizeof(T);
  const std::size_t in_pitch = in_row + padding;
  const std::size_t out_pitch = out_row + padding;
  DeviceBuffer in(in_pitch * height);
  DeviceBuffer out(out_pitch * out_height);
  check_cuda(cudaMemcpy2DAsync(in.data, in_pitch, input.data(), in_row, in_row,
                               static_cast<std::size_t>(height), cudaMemcpyHostToDevice, stream));
  check_cuda(cudaMemsetAsync(out.data, 0xAB, out.bytes, stream));
  std::optional<RemapQueue> own;
  RemapQueue& used = queue != nullptr ? *queue : own.emplace(stream);
  remapper.submit({in.data, in_pitch, width, height}, {out.data, out_pitch, out_width, out_height},
                  used);
  used.finish();
  std::vector<T> result(out_row * out_height / sizeof(T));
  check_cuda(cudaMemcpy2DAsync(result.data(), out_row, out.data, out_pitch, out_row,
                               static_cast<std::size_t>(out_height), cudaMemcpyDeviceToHost,
                               stream));
  check_cuda(cudaStreamSynchronize(stream));
  return result;
}

// A smooth, textured test image: gradients, rings and fine detail.
template <typename T>
std::vector<T> texture(int width, int height, int channels) {
  std::vector<T> image(static_cast<std::size_t>(width) * height * channels);
  const double max_value = sizeof(T) == 1 ? 255.0 : 4000.0;
  std::mt19937 random(11);
  std::uniform_real_distribution<double> noise(-0.03, 0.03);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < channels; ++c) {
        const double r = std::hypot(x - width * 0.4, y - height * 0.6);
        const double v = 0.5 + 0.25 * std::sin(r / (6.0 + 2 * c)) +
                         0.2 * std::cos((x + 2 * y) / 37.0 + c) + noise(random);
        image[(static_cast<std::size_t>(y) * width + x) * channels + c] =
            static_cast<T>(std::clamp(v, 0.0, 1.0) * max_value);
      }
    }
  }
  return image;
}

struct Difference {
  double max{};
  double mean{};
};

template <typename T>
Difference compare(const std::vector<T>& a, const std::vector<T>& b) {
  Difference d;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double e = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    d.max = std::max(d.max, e);
    d.mean += e;
  }
  d.mean /= static_cast<double>(a.size());
  return d;
}

// For nearest: how many output values are none of the four input pixels around
// their source, nor zero. A nearest remap never blends, which is what depth needs.
template <typename T>
std::size_t blended(const std::vector<T>& output, const std::vector<T>& input, int width,
                    int height, int channels, const RemapTable& table) {
  std::size_t count = 0;
  for (std::size_t p = 0; p < output.size() / channels; ++p) {
    const double sx = table.xy[2 * p];
    const double sy = table.xy[2 * p + 1];
    const int x0 = static_cast<int>(std::floor(sx));
    const int y0 = static_cast<int>(std::floor(sy));
    for (int c = 0; c < channels; ++c) {
      const T value = output[p * channels + c];
      bool found = value == 0;
      for (int k = 0; k < 4 && !found; ++k) {
        const int x = x0 + (k & 1);
        const int y = y0 + (k >> 1);
        found = x >= 0 && y >= 0 && x < width && y < height &&
                input[(static_cast<std::size_t>(y) * width + x) * channels + c] == value;
      }
      count += !found;
    }
  }
  return count;
}

// The backends this platform can run: the compiled ones it has the hardware for.
// (On a Jetson runner, test_platform requires the VIC among them.)
std::vector<Backend> platform_backends() {
  static const std::vector<Backend> backends = [] {
    std::vector<Backend> result;
    for (const Backend b : available_backends()) {
      if (const auto why = try_backend(b, PixelFormat::kMono8, Interpolation::kLinear, 1280, 720)) {
        std::cout << "  " << backend_name(b) << " cannot run here: " << *why << '\n';
      } else {
        result.push_back(b);
      }
    }
    return result;
  }();
  return backends;
}

// Largest difference from the CPU reference each backend may show with
// linear interpolation, in gray levels: the cuda backend rounds as the
// reference does, up to float precision; the VIC and PVA interpolate in fixed
// point, and the map between grid points 4 pixels apart.
double tolerance(Backend backend) { return backend == Backend::kCuda ? 1.0 : 6.0; }

// Every backend against the CPU reference, for every format it takes.
void test_parity() {
  const SyntheticStereo scene = synthetic_stereo(1280, 720);
  const StereoRectification r = compute_stereo_rectification(scene.calibration);
  const RemapModel model = left_remap(scene.calibration, r);
  const RemapTable table = build_remap_table(model, r.width, r.height);
  const PlatformFacts facts = probe_platform();
  cudaStream_t stream = nullptr;
  check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  struct Case {
    PixelFormat format;
    Interpolation interpolation;
  };
  const Case cases[] = {
      {PixelFormat::kMono8, Interpolation::kLinear},
      {PixelFormat::kMono8, Interpolation::kNearest},
      {PixelFormat::kMono16, Interpolation::kNearest},
      {PixelFormat::kRgb8, Interpolation::kLinear},
      {PixelFormat::kRgba8, Interpolation::kLinear},
      {PixelFormat::kBgra8, Interpolation::kNearest},
  };
  for (const Backend backend : platform_backends()) {
    for (const Case& c : cases) {
      const std::string what = std::string(backend_name(backend)) + " " +
                               std::string(pixel_format_name(c.format)) + " " +
                               std::string(interpolation_name(c.interpolation));
      if (backend_limitation(backend, facts, c.format, c.interpolation, r.width, r.height)) {
        continue;
      }
      std::unique_ptr<Remapper> remapper;
      try {
        remapper = create_remapper(backend, model, r.width, r.height, c.format, c.interpolation);
      } catch (const std::invalid_argument& error) {
        // Refused by VPI itself, as Thor's VIC refuses nearest.
        std::cout << "  " << what << ": refused: " << error.what() << '\n';
        continue;
      }
      const bool linear = c.interpolation == Interpolation::kLinear;
      const int n = channels(c.format);
      Difference d;
      std::size_t mismatched = 0;
      std::size_t total = 0;
      std::size_t not_input = 0;
      if (c.format == PixelFormat::kMono16) {
        const auto input = texture<std::uint16_t>(1280, 720, 1);
        const auto out = run(*remapper, input, 1280, 720, r.width, r.height, 1, stream);
        const auto ref = test::reference_remap(input, 1280, 720, 1, table, false);
        not_input = blended(out, input, 1280, 720, 1, table);
        d = compare(out, ref);
        for (std::size_t i = 0; i < out.size(); ++i) mismatched += out[i] != ref[i];
        total = out.size();
      } else {
        const auto input = texture<std::uint8_t>(1280, 720, n);
        const auto out = run(*remapper, input, 1280, 720, r.width, r.height, n, stream);
        const auto ref = test::reference_remap(input, 1280, 720, n, table, linear);
        if (!linear) not_input = blended(out, input, 1280, 720, n, table);
        d = compare(out, ref);
        for (std::size_t i = 0; i < out.size(); ++i) mismatched += out[i] != ref[i];
        total = out.size();
      }
      const double fraction = static_cast<double>(mismatched) / static_cast<double>(total);
      std::cout << "  " << std::left << std::setw(30) << what << " max " << d.max << " mean "
                << std::setprecision(4) << d.mean << " differing " << std::setprecision(3)
                << 100 * fraction << "%\n";
      if (linear) {
        VERIFY_MSG(d.max <= tolerance(backend), what + ": too far from the reference");
        // The VIC rounds its result differently: about 0.5 gray levels on Orin,
        // 0.6 on Thor, on average.
        VERIFY_MSG(d.mean < (backend == Backend::kVpiVic ? 1.0 : 0.5),
                   what + ": too far from the reference on average");
      } else {
        // Nearest never blends: every value is one of the four input pixels
        // around its source. It picks another of them than the reference only
        // near a half-pixel boundary: rarely, except on the VIC, whose
        // coordinates are coarser fixed point.
        VERIFY_MSG(not_input == 0,
                   what + ": " + std::to_string(not_input) + " values are not input pixels");
        VERIFY_MSG(fraction < (backend == Backend::kCuda     ? 1e-4
                               : backend == Backend::kVpiVic ? 0.1
                                                             : 1e-3),
                   what + ": too many pixels differ from the reference");
      }
    }
  }
  cudaStreamDestroy(stream);
}

// A wide view (alpha 1) of a strongly distorted camera, whose output reaches
// past the input's border, remapped by every backend twice through the same
// remapper: from packed rows, from rows with 64 bytes of padding in the input
// and the output, and from packed rows again, each through a queue of its
// own; then packed and padded through one queue. Every output pixel whose source is clearly outside
// the input is black, as on the dense map; those whose source is clearly inside match the CPU
// reference as in test_parity; and the three runs agree. (Within 1.5 px of the border the backends
// differ by design: the edge between the image and black moves with each one's border handling and
// grid.)
void test_border_and_padding() {
  for (const messages::DistortionModel distortion :
       {messages::DistortionModel::kPlumbBob, messages::DistortionModel::kEquidistant}) {
    const SyntheticStereo scene = synthetic_stereo(1280, 720, distortion);
    const StereoRectification r =
        compute_stereo_rectification(scene.calibration, StereoRectifyOptions{.alpha = 1.0});
    const RemapModel model = left_remap(scene.calibration, r);
    const RemapTable table = build_remap_table(model, r.width, r.height);
    const auto input = texture<std::uint8_t>(1280, 720, 1);
    const auto reference = test::reference_remap(input, 1280, 720, 1, table, true);
    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (const Backend backend : platform_backends()) {
      const std::string what = std::string(messages::distortion_model_name(distortion)) + " " +
                               std::string(backend_name(backend));
      const auto remapper = create_remapper(backend, model, r.width, r.height, PixelFormat::kMono8,
                                            Interpolation::kLinear);
      const auto packed = run(*remapper, input, 1280, 720, r.width, r.height, 1, stream);
      const auto padded = run(*remapper, input, 1280, 720, r.width, r.height, 1, stream, 64);
      const auto repacked = run(*remapper, input, 1280, 720, r.width, r.height, 1, stream);
      RemapQueue shared(stream);
      const auto shared_packed =
          run(*remapper, input, 1280, 720, r.width, r.height, 1, stream, 0, &shared);
      const auto shared_padded =
          run(*remapper, input, 1280, 720, r.width, r.height, 1, stream, 64, &shared);
      std::size_t outside = 0;
      std::size_t smeared = 0;
      Difference d;
      std::size_t compared = 0;
      std::size_t border = 0;
      for (int v = 0; v < r.height; ++v) {
        for (int u = 0; u < r.width; ++u) {
          const std::size_t i = static_cast<std::size_t>(v) * r.width + u;
          const std::optional<Point2> source = model.source(u, v);
          if (!source || source->x < -1.5 || source->y < -1.5 || source->x > 1280.5 ||
              source->y > 720.5) {
            ++outside;
            smeared += packed[i] != 0;
          } else if (source->x < 1.5 || source->y < 1.5 || source->x > 1280 - 2.5 ||
                     source->y > 720 - 2.5) {
            ++border;
          } else {
            const double e = std::abs(static_cast<double>(packed[i]) - reference[i]);
            d.max = std::max(d.max, e);
            d.mean += e;
            ++compared;
          }
        }
      }
      d.mean /= static_cast<double>(compared);
      std::cout << "  " << std::left << std::setw(22) << what << " " << outside
                << " pixels see nothing, " << smeared << " of them not black; " << border
                << " at the border; inside max " << d.max << " mean " << std::setprecision(4)
                << d.mean << '\n'
                << std::setprecision(6);
      VERIFY_MSG(outside > 1000, what + ": no border to test");
      VERIFY_MSG(smeared == 0, what + ": pixels outside the input are not black");
      VERIFY_MSG(d.max <= tolerance(backend), what + ": too far from the reference");
      VERIFY_MSG(d.mean < (backend == Backend::kVpiVic ? 1.0 : 0.5),
                 what + ": too far from the reference on average");
      VERIFY_MSG(padded == packed && repacked == packed,
                 what + ": padded rows, or packed ones after them, give another image");
      VERIFY_MSG(shared_packed == packed && shared_padded == packed,
                 what + ": padded rows after packed ones in one queue give another image");
    }
    cudaStreamDestroy(stream);
  }
}

// A distorted synthetic pair, rectified by every backend: each dot lies on
// the same row in both images, where the geometry puts it.
void test_rectified_rows() {
  for (const messages::DistortionModel model :
       {messages::DistortionModel::kPlumbBob, messages::DistortionModel::kEquidistant}) {
    const SyntheticStereo scene = synthetic_stereo(1280, 720, model);
    const StereoRectification r = compute_stereo_rectification(scene.calibration);
    const auto left = left_view(scene).render(5);
    const auto right = right_view(scene).render(5);
    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (const Backend backend : platform_backends()) {
      const auto left_remapper =
          create_remapper(backend, left_remap(scene.calibration, r), r.width, r.height,
                          PixelFormat::kMono8, Interpolation::kLinear);
      const auto right_remapper =
          create_remapper(backend, right_remap(scene.calibration, r), r.width, r.height,
                          PixelFormat::kMono8, Interpolation::kLinear);
      const auto left_out = run(*left_remapper, left, 1280, 720, r.width, r.height, 1, stream);
      const auto right_out = run(*right_remapper, right, 1280, 720, r.width, r.height, 1, stream);
      const AlignmentReport report =
          check_rectified(scene, r, left_out.data(), right_out.data(), r.width, 5);
      std::cout << "  " << messages::distortion_model_name(model) << " " << std::setw(9)
                << backend_name(backend) << ": " << report.dots << " dots, row error max "
                << std::setprecision(3) << report.max_row_error << " rms " << report.rms_row_error
                << " px, position error max " << report.max_position_error << " px\n";
      const std::string what = std::string(messages::distortion_model_name(model)) + " " +
                               std::string(backend_name(backend));
      VERIFY_MSG(report.dots > 100 && report.missing == 0, what + ": dots missing");
      VERIFY_MSG(report.max_row_error < 0.1, what + ": rows differ");
      VERIFY_MSG(report.max_position_error < 0.15, what + ": dots misplaced");
    }
    cudaStreamDestroy(stream);
  }
}

// Undistortion of one camera through every backend.
void test_undistorted_positions() {
  const SyntheticStereo scene =
      synthetic_stereo(1280, 720, messages::DistortionModel::kRationalPolynomial);
  const Undistortion u = compute_undistortion(scene.calibration.left);
  const auto raw = left_view(scene).render(2);
  cudaStream_t stream = nullptr;
  check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  for (const Backend backend : platform_backends()) {
    const int width = static_cast<int>(u.output.width);
    const int height = static_cast<int>(u.output.height);
    const auto remapper = create_remapper(backend, undistortion_remap(u), width, height,
                                          PixelFormat::kMono8, Interpolation::kLinear);
    const auto out = run(*remapper, raw, 1280, 720, width, height, 1, stream);
    const AlignmentReport report = check_undistorted(scene.plane, u, out.data(), u.output.width, 2);
    std::cout << "  " << std::setw(9) << backend_name(backend) << ": " << report.dots
              << " dots, position error max " << std::setprecision(3) << report.max_position_error
              << " rms " << report.rms_position_error << " px\n";
    VERIFY_MSG(report.dots > 100 && report.missing == 0,
               std::string(backend_name(backend)) + ": dots missing");
    VERIFY_MSG(report.max_position_error < 0.15,
               std::string(backend_name(backend)) + ": dots misplaced");
  }
  cudaStreamDestroy(stream);
}

// Auto on a CI runner: the VIC on a Jetson for gray images, cuda on x86_64
// and for color.
void test_auto() {
  const SyntheticStereo scene = synthetic_stereo(1280, 720);
  const StereoRectification r = compute_stereo_rectification(scene.calibration);
  const RemapModel model = left_remap(scene.calibration, r);
  std::string reason;
  const auto gray = create_remapper(Backend::kAuto, model, r.width, r.height, PixelFormat::kMono8,
                                    Interpolation::kLinear, &reason);
  std::cout << "  MONO8: auto chose " << gray->name() << ": " << reason << '\n';
  const auto color = create_remapper(Backend::kAuto, model, r.width, r.height, PixelFormat::kRgb8,
                                     Interpolation::kLinear, &reason);
  std::cout << "  RGB8: auto chose " << color->name() << ": " << reason << '\n';
  VERIFY(color->backend() == Backend::kCuda);
  const auto platform = ci_platform();
  if (platform == "jetson") {
    VERIFY_MSG(gray->backend() == Backend::kVpiVic, "auto must choose the VIC on a Jetson");
  } else if (platform == "x86_64") {
    VERIFY_MSG(gray->backend() == Backend::kCuda, "auto must choose cuda on x86_64");
  }
}

// What the backends refuse, with errors that say why.
void test_refusals() {
  const SyntheticStereo scene = synthetic_stereo(640, 480);
  const StereoRectification r = compute_stereo_rectification(scene.calibration);
  const RemapModel model = left_remap(scene.calibration, r);
  VERIFY(test::throws<std::invalid_argument>([&] {
    (void)create_remapper(Backend::kCuda, model, r.width, r.height, PixelFormat::kMono16,
                          Interpolation::kLinear);
  }));
  VERIFY(test::throws<std::invalid_argument>([&] {
    (void)create_remapper(Backend::kCuda, model, 0, r.height, PixelFormat::kMono8,
                          Interpolation::kLinear);
  }));
  const PlatformFacts facts = probe_platform();
  if (!facts.has_vic) {
    VERIFY(test::throws<std::invalid_argument>([&] {
      (void)create_remapper(Backend::kVpiVic, model, r.width, r.height, PixelFormat::kMono8,
                            Interpolation::kLinear);
    }));
  } else {
    // The VIC takes no color.
    VERIFY(test::throws<std::invalid_argument>([&] {
      (void)create_remapper(Backend::kVpiVic, model, r.width, r.height, PixelFormat::kRgb8,
                            Interpolation::kLinear);
    }));
  }
  // Images of another size than the map's.
  const auto remapper = create_remapper(Backend::kCuda, model, r.width, r.height,
                                        PixelFormat::kMono8, Interpolation::kLinear);
  DeviceBuffer buffer(640 * 480);
  cudaStream_t stream = nullptr;
  check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  {
    RemapQueue queue(stream);
    VERIFY(test::throws<std::invalid_argument>([&] {
      remapper->submit({buffer.data, 320, 320, 240}, {buffer.data, 640, 640, 480}, queue);
    }));
    // A second submit before the queue finished the first.
    DeviceBuffer output(640 * 480);
    const DeviceImage input_image{buffer.data, 640, 640, 480};
    const DeviceImage output_image{output.data, 640, 640, 480};
    remapper->submit(input_image, output_image, queue);
    VERIFY(test::throws<std::logic_error>(
        [&] { remapper->submit(input_image, output_image, queue); }));
    queue.finish();
    remapper->submit(input_image, output_image, queue);
    queue.finish();
    check_cuda(cudaStreamSynchronize(stream));
  }
  cudaStreamDestroy(stream);
  // A second submit, to another queue, before the first queue finished.
  check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  {
    DeviceBuffer output(640 * 480);
    const DeviceImage input_image{buffer.data, 640, 640, 480};
    const DeviceImage output_image{output.data, 640, 640, 480};
    RemapQueue first(stream);
    RemapQueue second(stream);
    remapper->submit(input_image, output_image, first);
    VERIFY(test::throws<std::logic_error>(
        [&] { remapper->submit(input_image, output_image, second); }));
    first.finish();
    remapper->submit(input_image, output_image, second);
    second.finish();
    // A queue that is gone does not hold its remappers back.
    {
      RemapQueue gone(stream);
      remapper->submit(input_image, output_image, gone);
    }
    remapper->submit(input_image, output_image, second);
    second.finish();
    check_cuda(cudaStreamSynchronize(stream));
  }
  // Rows shorter than the image, or 16-bit pixels on an odd byte: refused
  // before any work is queued.
  {
    const auto depth = create_remapper(Backend::kCuda, model, r.width, r.height,
                                       PixelFormat::kMono16, Interpolation::kNearest);
    DeviceBuffer input((2 * 640 + 64) * 480 + 2);
    DeviceBuffer output((2 * 640 + 64) * 480 + 2);
    auto* in = static_cast<std::uint8_t*>(input.data);
    const DeviceImage good_output{output.data, 2 * 640, 640, 480};
    RemapQueue queue(stream);
    VERIFY(test::throws<std::invalid_argument>(
        [&] { depth->submit({in, 640, 640, 480}, good_output, queue); }));
    VERIFY(test::throws<std::invalid_argument>(
        [&] { depth->submit({in, 2 * 640 + 1, 640, 480}, good_output, queue); }));
    VERIFY(test::throws<std::invalid_argument>(
        [&] { depth->submit({in + 1, 2 * 640, 640, 480}, good_output, queue); }));
    VERIFY(test::throws<std::invalid_argument>([&] {
      depth->submit({in, 2 * 640, 640, 480}, {output.data, 2 * 640 - 2, 640, 480}, queue);
    }));
    // Padded, aligned rows are fine.
    depth->submit({in, 2 * 640 + 64, 640, 480}, {output.data, 2 * 640, 640, 480}, queue);
    queue.finish();
    check_cuda(cudaStreamSynchronize(stream));
  }
  cudaStreamDestroy(stream);
  // The legacy default stream, which VPI cannot wrap.
  VERIFY(test::throws<std::invalid_argument>([] { RemapQueue queue(nullptr); }));
}

// A cuda remap submitted to a queue with vpi-* work queued waits for that
// work first: VPI allows no CUDA work on the stream until its own is done.
void test_cuda_after_vpi() {
  const std::vector<Backend> backends = platform_backends();
  const auto vpi =
      std::find_if(backends.begin(), backends.end(), [](Backend b) { return b != Backend::kCuda; });
  if (vpi == backends.end()) {
    std::cout << "  no vpi-* backend runs here\n";
    return;
  }
  const SyntheticStereo scene = synthetic_stereo(1280, 720);
  const StereoRectification r = compute_stereo_rectification(scene.calibration);
  const RemapModel model = left_remap(scene.calibration, r);
  const auto first =
      create_remapper(*vpi, model, r.width, r.height, PixelFormat::kMono8, Interpolation::kLinear);
  const auto second = create_remapper(Backend::kCuda, model, r.width, r.height, PixelFormat::kMono8,
                                      Interpolation::kLinear);
  cudaStream_t stream = nullptr;
  check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  {
    DeviceBuffer in(1280 * 720);
    DeviceBuffer out_vpi(1280 * 720);
    DeviceBuffer out_cuda(1280 * 720);
    RemapQueue queue(stream);
    first->submit({in.data, 1280, 1280, 720}, {out_vpi.data, 1280, r.width, r.height}, queue);
    const std::uint64_t before = remap_statistics().host_waits;
    second->submit({in.data, 1280, 1280, 720}, {out_cuda.data, 1280, r.width, r.height}, queue);
    const std::uint64_t waits = remap_statistics().host_waits - before;
    std::cout << "  " << backend_name(*vpi) << " then cuda: " << waits
              << " host wait(s) before the cuda remap\n";
    VERIFY_MSG(waits == 1, "the cuda remap did not wait for VPI's work");
    queue.finish();
    check_cuda(cudaStreamSynchronize(stream));
  }
  cudaStreamDestroy(stream);
}

#if HOLOSCAN_RECTIFY_TEST_WITH_VPI
// A VPI status that refuses the work becomes std::invalid_argument, which
// auto falls back from; any other is a failure, std::runtime_error.
void test_vpi_status() {
  for (const VPIStatus status :
       {VPI_ERROR_NOT_IMPLEMENTED, VPI_ERROR_INVALID_ARGUMENT, VPI_ERROR_INVALID_IMAGE_FORMAT,
        VPI_ERROR_INVALID_OPERATION, VPI_ERROR_NOT_SUPPORTED}) {
    VERIFY_MSG(test::throws<std::invalid_argument>([&] { detail::refuse_or_fail(status, "x"); }),
               std::string(vpiStatusGetName(status)) + " must be a refusal");
  }
  for (const VPIStatus status : {VPI_ERROR_OUT_OF_MEMORY, VPI_ERROR_DEVICE, VPI_ERROR_INTERNAL,
                                 VPI_ERROR_INVALID_CONTEXT, VPI_ERROR_NOT_READY}) {
    VERIFY_MSG(
        test::throws<std::runtime_error>([&] { detail::refuse_or_fail(status, "x"); }) &&
            !test::throws<std::invalid_argument>([&] { detail::refuse_or_fail(status, "x"); }),
        std::string(vpiStatusGetName(status)) + " must be a failure");
  }
}
#endif

// Host waits per pair through one queue: none for cuda, one for the VIC and
// the PVA, whose six tasks for the pair VPI orders itself.
void test_host_waits() {
  const SyntheticStereo scene = synthetic_stereo(1280, 720);
  const StereoRectification r = compute_stereo_rectification(scene.calibration);
  cudaStream_t stream = nullptr;
  check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceBuffer in(1280 * 720);
  DeviceBuffer left_out(1280 * 720);
  DeviceBuffer right_out(1280 * 720);
  const DeviceImage src{in.data, 1280, 1280, 720};
  for (const Backend backend : platform_backends()) {
    const auto left = create_remapper(backend, left_remap(scene.calibration, r), r.width, r.height,
                                      PixelFormat::kMono8, Interpolation::kLinear);
    const auto right = create_remapper(backend, right_remap(scene.calibration, r), r.width,
                                       r.height, PixelFormat::kMono8, Interpolation::kLinear);
    RemapQueue queue(stream);
    constexpr int kPairs = 20;
    const std::uint64_t before = remap_statistics().host_waits;
    for (int i = 0; i < kPairs; ++i) {
      left->submit(src, {left_out.data, 1280, r.width, r.height}, queue);
      right->submit(src, {right_out.data, 1280, r.width, r.height}, queue);
      queue.finish();
    }
    const double waits =
        static_cast<double>(remap_statistics().host_waits - before) / static_cast<double>(kPairs);
    std::cout << "  " << backend_name(backend) << ": " << waits << " host waits per pair\n";
    VERIFY_MSG(
        waits == (backend == Backend::kCuda ? 0.0 : 1.0),
        std::string(backend_name(backend)) + ": " + std::to_string(waits) + " host waits per pair");
  }
  check_cuda(cudaStreamSynchronize(stream));
  cudaStreamDestroy(stream);
}

// Time per image and per pair, submit to finish with the stream synchronized,
// after a warm-up: the left eye alone, then both eyes through one queue.
void test_timing() {
  const SyntheticStereo scene = synthetic_stereo(1280, 720);
  const StereoRectification r = compute_stereo_rectification(scene.calibration);
  const PlatformFacts facts = probe_platform();
  cudaStream_t stream = nullptr;
  check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceBuffer in(1280 * 720 * 4);
  DeviceBuffer left_out(1280 * 720 * 4);
  DeviceBuffer right_out(1280 * 720 * 4);
  check_cuda(cudaMemsetAsync(in.data, 0x40, in.bytes, stream));
  std::cout << "  ms per 1280x720 image and pair (submit to finish, stream synchronized), "
               "mean of 100:\n";
  for (const Backend backend : platform_backends()) {
    for (const PixelFormat format : {PixelFormat::kMono8, PixelFormat::kRgba8}) {
      if (backend_limitation(backend, facts, format, Interpolation::kLinear, 1280, 720)) continue;
      const auto left = create_remapper(backend, left_remap(scene.calibration, r), r.width,
                                        r.height, format, Interpolation::kLinear);
      const auto right = create_remapper(backend, right_remap(scene.calibration, r), r.width,
                                         r.height, format, Interpolation::kLinear);
      const std::size_t pitch = 1280U * bytes_per_pixel(format);
      const DeviceImage src{in.data, pitch, 1280, 720};
      const DeviceImage left_dst{left_out.data, pitch, r.width, r.height};
      const DeviceImage right_dst{right_out.data, pitch, r.width, r.height};
      RemapQueue queue(stream);
      const auto image = [&] {
        left->submit(src, left_dst, queue);
        queue.finish();
      };
      const auto pair = [&] {
        left->submit(src, left_dst, queue);
        right->submit(src, right_dst, queue);
        queue.finish();
      };
      const auto time_ms = [&](const auto& work) {
        for (int i = 0; i < 10; ++i) work();
        check_cuda(cudaStreamSynchronize(stream));
        constexpr int kRuns = 100;
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < kRuns; ++i) {
          work();
          check_cuda(cudaStreamSynchronize(stream));
        }
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                   .count() /
               kRuns;
      };
      const double image_ms = time_ms(image);
      const double pair_ms = time_ms(pair);
      std::cout << "    " << std::left << std::setw(9) << backend_name(backend) << " "
                << std::setw(6) << pixel_format_name(format) << " image " << std::fixed
                << std::setprecision(3) << image_ms << "  pair " << pair_ms << '\n'
                << std::defaultfloat;
      VERIFY(image_ms > 0 && image_ms < 100 && pair_ms > 0 && pair_ms < 200);
    }
  }
  cudaStreamDestroy(stream);
}

// On a Jetson runner, the hardware backends are there to test.
void test_platform() {
  const PlatformFacts facts = probe_platform();
  std::cout << "  VPI " << (facts.vpi_compiled ? "compiled" : "absent") << ", VIC " << facts.has_vic
            << ", PVA " << facts.has_pva << ", CUDA " << facts.cuda_device << '\n';
  VERIFY(facts.cuda_device);
  const std::vector<Backend> backends = platform_backends();
  std::cout << "  backends:";
  for (const Backend b : backends) std::cout << ' ' << backend_name(b);
  std::cout << '\n';
  if (ci_platform() == "jetson") {
    // Both engines, so that a broken backend fails the run instead of being
    // left out of it.
    VERIFY_MSG(facts.vpi_compiled && facts.has_vic && facts.has_pva,
               "a Jetson runner must have VPI, the VIC and the PVA");
    VERIFY_MSG(std::find(backends.begin(), backends.end(), Backend::kVpiVic) != backends.end(),
               "the VIC must remap gray images on a Jetson runner");
    VERIFY_MSG(std::find(backends.begin(), backends.end(), Backend::kVpiPva) != backends.end(),
               "the PVA must remap gray images on a Jetson runner");
  }
}

}  // namespace
}  // namespace holoscan::rectify

int main() {
  namespace rc = holoscan::rectify;
  return rc::test::run({
      {"platform", rc::test_platform},
      {"backend parity with the CPU reference", rc::test_parity},
      {"border and padded rows on every backend", rc::test_border_and_padding},
      {"rectified rows on every backend", rc::test_rectified_rows},
      {"undistorted positions on every backend", rc::test_undistorted_positions},
      {"auto", rc::test_auto},
      {"refusals", rc::test_refusals},
      {"cuda after vpi-* work in one queue", rc::test_cuda_after_vpi},
#if HOLOSCAN_RECTIFY_TEST_WITH_VPI
      {"VPI statuses", rc::test_vpi_status},
#endif
      {"host waits per pair", rc::test_host_waits},
      {"timing", rc::test_timing},
  });
}
