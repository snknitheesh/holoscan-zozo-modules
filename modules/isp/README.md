# holoscan-isp

A Holoscan SDK 5 Module that turns raw camera frames into display-ready RGB on the GPU: an image
signal processor for cameras that deliver RAW, such as GMSL and CSI sensors.

- `IspOp` takes a Bayer mosaic (any of RGGB, BGGR, GRBG, GBRG; 1 to 16 bits per sample in uint8 or
  uint16, or packed MIPI CSI-2 RAW10 / RAW12) and emits RGB8, RGBA8, RGB16 or float RGB, and on
  request the frame's histograms and statistics.
- `IspPipeline` is the same pipeline without a Holoscan graph, on device pointers and a CUDA
  stream, or on host memory.

The [ISP demo](../../applications/isp/) runs it on a synthetic raw sensor with known ground truth
and shows the raw mosaic beside the processed image in HoloViz.

| | |
| --- | --- |
| CMake package | `find_package(holoscan-isp)` |
| Targets | `holoscan::isp` |
| Headers | `<holoscan/isp/...>` |
| C++ namespace | `holoscan::isp` |
| Debian package | `holoscan-isp` |
| Platforms | x86_64; aarch64 (Jetson Thor with JetPack 7, Jetson Orin with JetPack 7.2) |

## The pipeline

Every stage runs in one fused CUDA kernel over 32 x 16 pixel tiles in shared memory; each raw
sample is read and preprocessed once per tile, with a halo of 1 to 3 pixels reflected at the image's
edges.

1. **Black level**: subtracts `black_level` counts, or `channel_black_levels` per channel (R, Gr,
   Gb, B: red, the green in a red row, the green in a blue row, blue), and scales so that
   `white_level` (by default `2^bits - 1`) is 1.
2. **Lens shading** (optional): multiplies by a gain interpolated bilinearly from a grid
   (`LensShading`, 2x2 to 256x256 points, one gain or four per point, the grid's corners on the
   image's corner pixels).
3. **White balance**: multiplies red, green and blue by gains, then clips to 1, so that clipped
   highlights stay white. `kManual` uses `wb_gains`. `kGrayWorld` makes the frame's channel means
   equal; `kWhitePatch` makes each channel's `white_patch_percentile` (0.99) equal. Both estimate
   from the frame itself on the GPU, over the 2x2 quads with no saturated sample, before the frame
   is processed, keep green at 1, and move the gains `awb_smoothing` (0.25) of the way to each
   frame's estimate; the first frame takes its estimate. A frame with nothing to estimate from
   (black, or all saturated) keeps the last gains.
4. **Demosaic**: `kBilinear` (each missing colour from its nearest neighbours of that colour, 3x3)
   or `kMalvar` (default; Malvar, He and Cutler's gradient-corrected linear interpolation, 5x5,
   2 dB to 6 dB better on edges at about the same cost).
5. **Sharpening** (optional): an unsharp mask of `sharpen` (0 to 4) against a 3x3 binomial blur.
6. **Colour correction**: the 3x3 matrix `ccm` (row-major, camera RGB to output linear RGB), then
   clipped to 0..1.
7. **Tone curve**: `kSrgb` (default), `kGamma` (`gamma`), `kLinear`, or `kLut`, a table of 2 to
   65536 outputs for inputs evenly spaced over 0..1. The built-in curves are evaluated as tables of
   4097 entries, interpolated linearly (within 2e-5 of the formulas).
8. **Statistics** (optional, below).

Black and white levels, white balance gains and the colour matrix are the camera's calibration;
`IspParams` holds them all, and `validate_params()` names the first unusable one.

## `IspOp`

| Port | Type | |
| --- | --- | --- |
| `raw` (input) | `schema::ImageT` | `BAYER_RGGB`, `BAYER_BGGR`, `BAYER_GRBG`, `BAYER_GBRG`, or `MONO8` / `MONO16` with the pattern in the configuration; uint8 or uint16 `[H, W]` or `[H, W, 1]`, rows packed or padded; with `packing`, uint8 `[H, row bytes]` and the descriptor's `width`. In host or CUDA device memory |
| `image` | `schema::ImageT` | `RGB8`, `RGBA8` (alpha 255), `RGB16` or `RGB32F` `[H, W, channels]`, packed; device memory for the cuda backend, host memory for cpu. The descriptor keeps the raw image's header, `exposure_time_ns` and `gain` |
| `histogram` (with `stats`) | `holoscan::Tensor` | uint32 `[4, 256]`: the output's luma, red, green and blue, in 8-bit code values |
| `stats` (with `stats`) | `holoscan::Tensor` | float32 `[12]`, below |

Every output carries the raw frame's capture time and frame id. The operator runs once per raw
frame (`OnEach`). The image is emitted before the statistics; an operator cannot send a refused
output again, so a refused emit fails the frame.

`IspOpConfig`:

| Setting | Default | |
| --- | --- | --- |
| `backend` | `Backend::kCuda` | `kCuda`, `kCpu`, or `kAuto` (below) |
| `params` | `IspParams{}` | The pipeline's settings |
| `output_format` | `OutputFormat::kRgb8` | `kRgb8`, `kRgba8`, `kRgb16` or `kRgb32f` |
| `pattern` | none | The mosaic of `MONO8` / `MONO16` input; with a `BAYER_*` input, it must agree |
| `bit_depth` | `0` | Significant bits per sample; 0 takes the image's `significant_bits`, or the whole container (8, 16; 10 and 12 when packed) |
| `packing` | `Packing::kNone` | `kMipiRaw10` or `kMipiRaw12` for packed CSI-2 rows |
| `stats` | `false` | Declare the `histogram` and `stats` outputs |
| `max_width`, `max_height` | `3840`, `2160` | Largest frame, which sizes the outputs |

`IspParams`:

| Setting | Default | |
| --- | --- | --- |
| `black_level` | `0` | Raw counts, every channel |
| `channel_black_levels` | none | R, Gr, Gb, B counts, in place of `black_level` |
| `white_level` | `0` | Raw counts of saturation; 0 is `2^bits - 1` |
| `lens_shading` | none | The gain grid |
| `white_balance` | `kManual` | `kManual`, `kGrayWorld` or `kWhitePatch` |
| `wb_gains` | `{1, 1, 1}` | R, G, B; the auto modes start from them |
| `awb_smoothing` | `0.25` | Share of the way to each frame's estimate, (0, 1] |
| `white_patch_percentile` | `0.99` | [0.5, 1) |
| `demosaic` | `kMalvar` | `kBilinear` or `kMalvar` |
| `ccm` | identity | 3x3, row-major |
| `tone_curve` | `kSrgb` | `kSrgb`, `kGamma`, `kLinear` or `kLut` |
| `gamma`, `lut` | `2.2`, empty | For `kGamma` and `kLut` |
| `sharpen` | `0` | 0..4 |
| `auto_exposure` | below | The exposure suggestion |

## Statistics

The histograms count the output image's pixels by their 8-bit code values (16-bit and float outputs
rounded to 8 bits): luma (Rec. 709 weights of the coded R, G, B), red, green, blue. Each row sums
to the pixel count. The statistics vector (`stats::` indices in `isp_params.hpp`):

| Index | |
| --- | --- |
| `kMeanR`, `kMeanG`, `kMeanB` | Mean linear red, green, blue, 0..1, after the colour correction, before the tone curve |
| `kMeanLuma` | Their Rec. 709 luminance |
| `kAeErrorEv` | `log2(auto_exposure.target / mean luminance)`: the stops to add, -8..8 |
| `kExposureScale` | `2^(damping * error)`: the factor the next exposure time x gain should change by |
| `kSuggestedExposureNs`, `kSuggestedGain` | The next exposure time and analog gain within the bounds of `AutoExposureParams`, exposure time first; 0 when the raw image carries no `exposure_time_ns` |
| `kWbGainR`, `kWbGainG`, `kWbGainB` | The white balance gains the frame was processed with |
| `kSaturatedFraction` | Share of raw samples at or above the white level |

`AutoExposureParams`: `target` 0.18, `damping` 0.5, `min_exposure_ns` 10 us, `max_exposure_ns`
33 ms, `min_gain` 1, `max_gain` 16. The module only suggests: it never drives a camera.

The sums behind the means are integers (units of 2^-20), and the histograms integer counts, so the
statistics are the same on every run.

## Backends

| Backend | Runs on | Images |
| --- | --- | --- |
| `kCuda` (default) | The GPU: the fused kernel, plus a pass for the white balance estimate and one for the histograms | CUDA device memory |
| `kCpu` | The host, one thread: the reference implementation | Host memory |
| `kAuto` | `kCuda` where a CUDA device is usable, else `kCpu`; chosen when the operator is built, and logged | |

Both backends evaluate the same arithmetic (`src/isp_math.hpp`), compiled without fused
multiply-adds, so they produce the same image, histograms and white balance gains bit for bit (the
auto-exposure fields, which use `log2` and `exp2`, agree to 1e-5). When the raw images are not
where the backend reads them, the operator copies each frame (host to device for cuda, device to
host for cpu), and says so once in the log, with the bytes per frame and how to avoid the copy.

Existing demosaics considered, and why the module does not use them:

- **The Holoscan SDK 5.0 Vision module's `holoscan::vision::BayerDemosaicOp`**
  (`libholoscan-vision5-dev-cuda-13`, not in this repository's CI image) wraps NPP's
  `nppiCFAToRGB` / `nppiCFAToRGBA`: bilinear only, even widths and heights only, RGB of the input's
  element type, and nothing before or after it. Black level, lens shading and white balance must
  come before the demosaic and the colour matrix and tone curve after it, so it would sit between
  two more full-frame passes and their buffers, and the CPU backend could not match it bit for bit.
- **NPP's `nppiCFAToRGB`** itself (CUDA Toolkit), for the same reasons. On its own it takes
  0.023 ms at 1920x1080 and 0.089 ms at 3840x2160 (16-bit, RTX 5090), more than this module's whole
  fused bilinear pipeline (0.017 ms and 0.064 ms).
- **VPI 4.1** has no demosaic or ISP algorithm: its image formats describe Bayer data, but no
  algorithm (`ConvertImageFormat` included) takes it. Its `Histogram` and `ImageStats` could compute
  statistics on the PVA, but only of an image the GPU has produced, adding a second engine for
  numbers the fused kernel gets almost for free.

## Use it in your graph

```cmake
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core sensor_io)
find_package(holoscan-isp 1.0 REQUIRED CONFIG)
target_link_libraries(my_app PRIVATE holoscan::isp)
```

```cpp
#include <holoscan/isp/isp_op.hpp>

namespace isp = holoscan::isp;

isp::IspOpConfig config;
config.params.channel_black_levels = std::array<float, 4>{256, 256, 256, 256};  // 12-bit sensor
config.params.white_balance = isp::WhiteBalanceMode::kGrayWorld;
config.params.ccm = {1.6F, -0.4F, -0.2F, -0.3F, 1.5F, -0.2F, -0.1F, -0.5F, 1.6F};
config.stats = true;
config.max_width = 1920;
config.max_height = 1080;
auto processor = graph.op<isp::IspOp>("isp", config);

graph.add_flow(camera->frame, processor->raw);  // Output<schema::ImageT>, BAYER_RGGB
graph.add_flow(processor->image, viewer->image);
graph.add_flow(processor->histogram, exposure->histogram);
graph.add_flow(processor->stats, exposure->stats);
```

The cuda backend's outputs are in device memory, so bind their GPU when compiling, for each output:

```cpp
holoscan::CompileOptions options;
options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
    .operator_path = "isp", .output_port = "image", .device = holoscan::DeviceId{0}});
```

A lossless (`kBlocking`) edge whose queue is full fails the frame. Keep the camera at most one
frame ahead of the slowest consumer, as the demo's source does, or connect consumers that only need
the newest frame, such as a display, with `QueuePolicy::kDropOldest` and depth 1.

Without a graph, `IspPipeline` in `<holoscan/isp/isp_pipeline.hpp>` runs the same pipeline on
`RawImageView` and `RgbImageView` pointers and a CUDA stream (or host memory with `kCpu`).
`<holoscan/isp/synthetic_raw.hpp>` is the synthetic raw sensor of the tests and the demo: a colour
checker, gray and hue ramps, bars, a slanted edge and a moving disc, softened by a two-pixel lens
blur, seen through a colour matrix, white balance gains and vignetting, mosaicked, offset by black
levels, quantized and packed as a sensor would, with `synthetic_truth()` giving what an ideal ISP
outputs at every pixel, and `synthetic_isp_params()` the settings that undo the camera.
`<holoscan/isp/synthetic_raw_cuda.hpp>` renders it on the GPU, bit for bit.

## Accuracy

All on frames rendered at test time, with per-channel black levels, white balance gains of 1.9 and
1.6, a colour matrix and 40 % vignetting to undo. PSNR is over the whole 8-bit sRGB image against
the truth; CIEDE2000 between the mean colours of the inner half of each of the 24 colour checker
patches. 640x480, RGGB:

| Samples | Bilinear PSNR | Malvar-He-Cutler PSNR | Patch CIEDE2000, mean / worst |
| --- | --- | --- | --- |
| uint8, 8 bits | 27.58 dB | 29.98 dB | 0.34 / 2.63 |
| uint16, 10 bits; RAW10 | 27.68 dB | 30.15 dB | 0.02 / 0.25 |
| uint16, 12 bits; RAW12 | 27.69 dB | 30.17 dB | 0.00 / 0.00 |
| uint16, 16 bits | 27.69 dB | 30.17 dB | 0.00 / 0.00 |

Nearly all the error is at edges (patch borders, bars of 4 to 16 pixels, the disc), where a mosaic
cannot know the colours; flat areas come out within quantization of the truth. In linear light the
gap between the methods is wider: 27.8 dB against 33.8 dB at 320x240. On every pattern and format
at 480x360 the tests require at least 26 dB bilinear and 28.5 dB Malvar-He-Cutler, Malvar-He-Cutler
1.5 dB better, and patch CIEDE2000 under 0.1 mean and 0.5 worst (0.6 and 3.5 for 8 bits, which loses
the dark patches to quantization after the white balance gains).

| What | Result |
| --- | --- |
| Gray world on a gray scene, white patch on the chart, every pattern, both backends | The camera's gains (2.1, 1, 1.45) within 1 % |
| White balance smoothing | Each frame moves the gains `awb_smoothing` of the way; converged within 1 % in 30 frames |
| Colour matrix on a flat field, float output | Within 2e-6 of the arithmetic |
| sRGB, gamma 2.2, linear and a table | Within 3e-5 of the formulas |
| Exposure suggestion on a flat field one stop under | Error 1.000 stop, scale 2.000, 20 ms at gain 1.5 for a 15 ms frame capped at 20 ms |
| CUDA against the CPU backend | Identical images, histograms and gains in every format, method, size, white balance and tone curve |
| Bilinear against an independent reference | Within 1e-6 on random mosaics, every pattern |
| Malvar-He-Cutler on linear gradients | Exact to 2e-6 |
| Saturated frame (4x exposure) | Clipped white patch exactly white, no colour cast |
| Black frame | All black; the white balance keeps its gains |

## Performance

Mean GPU time per frame over 100 frames, 12-bit raw in, RGB8 out, from CUDA events;
`isp_benchmark` measures it. "Plain" is black level, manual white balance, demosaic, colour matrix
and sRGB; "full" adds lens shading, the gray world estimate, sharpening, histograms and statistics.

| Frame | Demosaic | Stages | RTX 5090 | Orin |
| --- | --- | --- | --- | --- |
| 1920x1080 | bilinear | plain | 0.017 ms | ORIN_1 |
| 1920x1080 | bilinear | full | 0.090 ms | ORIN_2 |
| 1920x1080 | Malvar-He-Cutler | plain | 0.019 ms | ORIN_3 |
| 1920x1080 | Malvar-He-Cutler | full | 0.092 ms | ORIN_4 |
| 3840x2160 | bilinear | plain | 0.064 ms | ORIN_5 |
| 3840x2160 | bilinear | full | 0.226 ms | ORIN_6 |
| 3840x2160 | Malvar-He-Cutler | plain | 0.073 ms | ORIN_7 |
| 3840x2160 | Malvar-He-Cutler | full | 0.237 ms | ORIN_8 |

The cpu backend, one thread of the x86_64 machine, takes 24 ms (bilinear) and 38 ms
(Malvar-He-Cutler) for a plain 1920x1080 frame, 52 and 64 ms full. The x86_64 machine has an RTX
5090, shared with other work during the runs; the Orin is a Jetson AGX Orin developer kit (64 GB)
with JetPack 7.2, in the power mode it was set to, also running other jobs. The plain pipeline is
bound by memory: it reads each sample and writes each pixel once. The full one adds a read of the
raw frame for the white balance estimate and of the output for the histograms.

## Requirements

- Holoscan SDK 5.0 (Core and Sensor I/O; Testing for the tests), CUDA 13, CMake 4.0 or later and a
  C++20 compiler.

The module downloads nothing and needs no models or data.

## Build, test and install

From the repository root, in the repository's CI image or with the requirements above:

```bash
cmake -S modules/isp -B build/isp -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build/isp
ctest --test-dir build/isp --output-on-failure
cmake --install build/isp --prefix "$PWD/install"
```

| CMake option | Default | |
| --- | --- | --- |
| `BUILD_TESTING` | `OFF` | Build the tests, when the module is the top-level project |
| `HOLOSCAN_ISP_BUILD_TESTING` | `OFF` | Build the tests when another project includes the module (`add_subdirectory()`, `FetchContent`); its `BUILD_TESTING` and `include(CTest)` do not |
| `CMAKE_CUDA_ARCHITECTURES` | `native` (`all-major` for the package) | GPUs to compile for |

The tests, all on raw frames made at test time:

| Test | Needs | Checks |
| --- | --- | --- |
| `isp_pipeline` | GPU | `IspPipeline`: PSNR and patch CIEDE2000 against the truth for every pattern, sample format (uint8, uint16 at 10, 12 and 16 bits, RAW10, RAW12) and demosaic method; CUDA against CPU bit for bit (sharpening, odd sizes, both auto white balances, four-channel lens shading, every tone curve and output format); the GPU render of the synthetic sensor against the host's; padded rows; white balance recovery and smoothing; the colour matrix; the tone curves; histogram sums, means and the exposure suggestion; determinism; 2x2 to odd sizes, saturation and black frames; rejected settings and views |
| `isp_op` | GPU | `IspOp` through the SDK's operator test harness, against the CPU backend bit for bit: `BAYER_*` from device and host memory, `MONO16` with the configured pattern, packed RAW10, every output format, the statistics ports, the cpu backend from device memory, the auto backend; the capture time, frame id, header, exposure and gain of the outputs; unusable inputs and settings |
| `isp_benchmark` | GPU | Runs the benchmark briefly, so that it keeps working |
| `isp_params`, `isp_cpu_reference` | Nothing | SDK-free (`tests/host/`, also run by the hosted CI job): names, validation, tone tables; the CPU backend's bilinear against an independent reference, Malvar-He-Cutler on gradients, packed formats, and the whole pipeline against the synthetic truth |

[`tests/installed_consumer/`](tests/installed_consumer/) is a separate project that uses the
installed package: a graph from host-memory 12-bit frames of the synthetic sensor through `IspOp`,
checking each image's PSNR and that the histograms count every pixel. Build it with the install
prefix on `CMAKE_PREFIX_PATH` and run `ctest` there.

## Debian package

Inside the repository's CI image, with the module's folder as the Holoscan CLI project root:

```bash
holoscan --project-root modules/isp package holoscan-isp --language cpp \
  --pkg-generator DEB --local
```

`--language cpp` selects the module rather than its package entry, which has the same name. This
builds for every major GPU architecture and writes `holoscan-isp_1.0.0_<arch>.deb` to the current
directory. It installs to `/usr` and depends on `holoscan (>= 5.0.0)`, so
`apt install ./holoscan-isp_1.0.0_<arch>.deb` needs NVIDIA's CUDA repository.

## Licenses

Apache-2.0. See [NOTICE](NOTICE).
