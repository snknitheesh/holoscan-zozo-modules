# holoscan-denoise

A Holoscan SDK 5 Module that cleans up noisy low-light camera frames and extends their dynamic
range, on the GPU, with low latency:

- `DenoiseOp` denoises a video: a bilateral or non-local-means spatial filter and a
  motion-adaptive temporal filter on CUDA, or VPI 4's Temporal Noise Reduction (TNR) on the
  Jetson's VIC or on CUDA. MONO8, RGB8, RGBA8 and NV12.
- `ExposureFusionOp` makes one 8-bit image of a high-dynamic-range scene: Mertens exposure fusion
  of 2 or 3 differently exposed brackets matched by capture time, or Reinhard's global tone map of
  one 16-bit or float frame.
- `Denoiser`, `ExposureFusion` and `ToneMapper` are the same without a Holoscan graph, on device
  images and a CUDA stream.

The [denoise demo](../../applications/denoise/) runs both on synthetic scenes with ground truth and
shows noisy beside denoised, and the brackets beside the fused image, in HoloViz.

| | |
| --- | --- |
| CMake package | `find_package(holoscan-denoise)` |
| Targets | `holoscan::denoise` |
| Headers | `<holoscan/denoise/...>` |
| C++ namespace | `holoscan::denoise` |
| Debian package | `holoscan-denoise` |
| Platforms | x86_64; aarch64 (Jetson Thor and Jetson Orin with JetPack 7.2) |

## What VPI and the SDK cover, and what this adds

| | VPI 4.1 | Holoscan SDK 5.0 | This module |
| --- | --- | --- | --- |
| Temporal denoising | TNR (version 3) on the VIC (Jetson) or CUDA, NV12 only; one strength and a scene preset; no motion threshold, history length or scene-cut control | None | The vpi backend wraps TNR: any format in (converted on the GPU, odd sizes padded), its own scene-cut detection. The cuda backend is a motion-adaptive recursive filter whose history length, motion threshold and scene-cut share are settings, with a GPU noise estimate |
| Spatial denoising | Bilateral filter (8/16-bit and float single channel, NV12 on CUDA) | None | Bilateral and non-local means on colour (MONO8, RGB8, RGBA8, NV12), driven by the measured noise and by what the temporal filter left |
| Exposure fusion, tone mapping | None | None | Mertens exposure fusion on GPU Laplacian pyramids; Reinhard's global tone map with GPU statistics |
| Holoscan operators | | None. Core, Inference and Visualization (the CI image's packages) have no image filters; the Vision module (`libholoscan-vision5-dev-cuda-13` 5.0.0, not in the CI image) has Bayer demosaic, resize, remap, crop, border, reformat, normalize, type conversion and non-maximum suppression, but no denoising, fusion or tone mapping | `DenoiseOp`, `ExposureFusionOp` |

VPI's TNR is used as it is where it fits: it is the `vpi` backend, and `auto` chooses it on a
Jetson whose VIC runs it, which leaves the GPU free. VPI's bilateral filter is not used: it filters
one channel at a time with no knowledge of the temporal filter, and the cuda backend's colour
bilateral costs 0.08 ms at 1920x1080 on an RTX 5090.

## `DenoiseOp`

| Port | Type | |
| --- | --- | --- |
| `image` (input) | `schema::ImageT` | `MONO8`, `RGB8` or `RGBA8`: uint8 `[H, W]` or `[H, W, channels]`, packed pixels, any row pitch; or `NV12` by its two plane layouts, even sizes. Host or CUDA device memory |
| `denoised` | `schema::ImageT` | The input's format and size: uint8 `[H, W, channels]` (`NV12`: `[H * 3 / 2, W, 1]` with plane layouts), packed, in CUDA device memory. The descriptor keeps the input's header, colour space, exposure and gain; the sample its capture time and frame id |

The operator runs once per image (`OnEach`). The temporal filter's history lives in the operator:
a change of size or format, a restart, or a scene cut starts a new sequence. Host images are copied
to the GPU, which is logged once with the bytes per frame.

`DenoiseOpConfig`:

| Setting | Default | |
| --- | --- | --- |
| `backend` | `Backend::kCuda` | `kCuda`, `kVpi`, or `kAuto` (below) |
| `params` | `DenoiseParams{}` | Below |
| `vpi.engine` | `VpiEngine::kAuto` | vpi backend: `kVic`, `kCuda`, or `kAuto` (the VIC where VPI runs TNR on it) |
| `vpi.preset` | `VpiPreset::kIndoorLowLight` | vpi backend: VPI's TNR scene preset; on the tests' low-light scene indoor low light gains the most |
| `max_width`, `max_height` | `3840`, `2160` | Largest image, which sizes the output |

`DenoiseParams`:

| Setting | Default | |
| --- | --- | --- |
| `spatial` | `kBilateral` | `kNone`, `kBilateral` (7x7, on the colour difference), `kNlm` (non-local means: 3x3 patches compared on luma over a 7x7 window) |
| `temporal` | `true` | The motion-adaptive temporal filter |
| `strength` | `0.5` | 0..1. 0 leaves frames unchanged; it scales the spatial filter's range and the frames a still pixel averages (1 + 31 x strength of `max_history` 32); for vpi it is TNR's strength |
| `noise_sigma` | `0` | The noise's standard deviation in code values; 0 estimates it from each frame on the GPU |
| `motion_threshold` | `3` | In standard deviations of the expected difference: above it a pixel starts to count as moving, at twice it its history is dropped. 1..20 |
| `max_history` | `32` | Frames a still pixel averages at strength 1, 2..255 |
| `scene_cut_fraction` | `0.5` | A frame in which more than this share of the pixels moves drops the whole history. (0, 1] |

### How the cuda backend filters

Per frame, all on the operator's CUDA stream, nothing waiting on the host:

1. **Noise level**: the configured `noise_sigma`, or Immerkaer's estimate from the frame's green
   (luma for mono and NV12): the mean absolute response to a 3x3 kernel that cancels linear
   gradients, summed as integers in a fixed order.
2. **Temporal** (with `temporal`): a motion pass compares each pixel's 3x3 luma mean, and the pixel
   itself, with the history (the previous temporal output, kept in float) against the noise
   expected in that difference, and counts the pixels that moved. A blend pass then averages still
   pixels into the history, a running mean up to `history_frames()` frames and a recursive filter
   at that weight after, and replaces moving pixels by the frame, in proportion to how surely they
   moved: a moving object leaves no trail. When more than `scene_cut_fraction` of the pixels moved,
   the whole history is replaced.
3. **Spatial** (with `spatial`): bilateral or non-local means of the temporal output (or of the
   frame), whose noise level is the frame's times what the pixel's temporal averaging left: still,
   well-averaged areas stay sharp, moving and newly uncovered ones are smoothed. The history is
   kept before this step, so spatial smoothing never accumulates over frames.

NV12's UV plane follows its Y plane: the motion of the 2x2 luma pixels it covers, and the smaller
of their histories. RGBA8's alpha passes through.

### Backends

| Backend | Runs on | Spatial filter | Notes |
| --- | --- | --- | --- |
| `kCuda` (default) | The GPU | Any | Deterministic: the same frames give the same bytes |
| `kVpi` | VPI's TNR on the VIC (Jetson Orin, Thor) or CUDA (`vpi.engine`); with `kAuto`, frames under 64x16, which the VIC refuses, on CUDA | None of its own to configure: `spatial` must be `kNone` (TNR filters spatially itself) and `temporal` true | MONO8, RGB8 and RGBA8 are converted to full-range BT.601 NV12 and back on the GPU (chroma subsampled, as TNR needs), odd sizes padded by a pixel. The operator waits on the host for VPI each frame. A scene cut, found as on the cuda backend, restarts TNR |
| `kAuto` | `kVpi` where VPI is built, its VIC runs TNR and `spatial` is `kNone` with `temporal`; `kCuda` otherwise. Chosen when the operator is built, logged at start | | On x86_64 always `kCuda`: VPI's TNR on CUDA gains less PSNR than the cuda backend and costs about as much |

## `ExposureFusionOp`

| Port | Type | |
| --- | --- | --- |
| `bracket0`, `bracket1` (inputs, `kFusion`) | `schema::ImageT` | `MONO8`, `RGB8` or `RGBA8` brackets of one size and format, in any order of exposure; host or CUDA device memory |
| `bracket2` (input, `kFusion` with 3 brackets) | `schema::ImageT` | |
| `hdr` (input, `kToneMap`) | `schema::ImageT` | `MONO16`, `RGB16`, `RGBA16` (code values of `significant_bits`, else 16) or `MONO32F`, `RGB32F`, `RGBA32F` (linear, any scale); host or CUDA device memory |
| `image` | `schema::ImageT` | The brackets' format (alpha 255), or `MONO8`, `RGB8`, `RGBA8` for 1, 3, 4 HDR channels: uint8 `[H, W, channels]` in CUDA device memory, with bracket0's (or the HDR frame's) header, capture time and frame id |

The brackets of one image are matched by capture time (`OnAligned`): equal times by default, or
within `match_window` of each other, as when a camera captures them one after another.

`ExposureFusionOpConfig`:

| Setting | Default | |
| --- | --- | --- |
| `mode` | `FusionMode::kFusion` | `kFusion` or `kToneMap` |
| `brackets` | `3` | 2 or 3 |
| `match_window` | `0` | 0..1 s |
| `fusion.contrast_weight`, `.saturation_weight`, `.exposure_weight` | `1`, `1`, `1` | Mertens' exponents, 0..4 |
| `fusion.exposure_sigma` | `0.2` | Width of the well-exposedness Gaussian around 0.5 |
| `fusion.levels` | `0` | Pyramid levels, 1..16; 0 takes up to 6 (`kAutoFusionLevels`) |
| `tone_map.key` | `0.18` | Middle gray, 0.01..1 |
| `tone_map.white` | `0` | The scaled luminance mapped to white; 0 takes the frame's brightest pixel |
| `tone_map.curve`, `.gamma` | `kSrgb`, `2.2` | `kSrgb`, `kGamma` or `kLinear` |
| `max_width`, `max_height` | `3840`, `2160` | Largest image |

**Fusion** weights each bracket's pixels by contrast (the magnitude of the Laplacian of its gray),
saturation (the standard deviation of R, G and B) and well-exposedness (a Gaussian around mid-gray,
per channel), each raised to its exponent, normalizes the weights across the brackets, and sums
the brackets' Laplacian pyramids weighted by the Gaussian pyramids of their weights (5-tap binomial
kernel, mirrored edges, float). Two departures from the paper, both decisions of this module:
contrast and saturation are floored at 1/255, so that in flat or gray areas, where they are 0 in
every bracket, well-exposedness still decides instead of an even mix of a clipped and a good
bracket; and the automatic depth stops at 6 levels, because a deeper pyramid blends over wider
areas and leaves a halo about 2^(levels - 3) pixels wide at an edge between very different
exposures (35 pixels at 1920x1080 with the full depth of 8 levels, 8 with 6).

**Tone mapping** reduces the frame's log-average and largest luminance (Rec. 709) on the GPU in
double, in a fixed order, then maps each pixel's luminance L, scaled to Lm = key x L / Lavg, to
Lm (1 + Lm / Lw^2) / (1 + Lm), scales its colour by that over L, and codes it with the curve.

## Use it in your graph

```cmake
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core sensor_io)
find_package(holoscan-denoise 1.0 REQUIRED CONFIG)
target_link_libraries(my_app PRIVATE holoscan::denoise)
```

```cpp
#include <holoscan/denoise/denoise_op.hpp>
#include <holoscan/denoise/exposure_fusion_op.hpp>

namespace dn = holoscan::denoise;

dn::DenoiseOpConfig config;
config.params.spatial = dn::SpatialFilter::kBilateral;
config.params.strength = 0.6F;
config.max_width = 1920;
config.max_height = 1080;
auto denoiser = graph.op<dn::DenoiseOp>("denoise", config);
graph.add_flow(camera->image, denoiser->image);  // Output<schema::ImageT>, RGB8 or NV12
graph.add_flow(denoiser->denoised, detector->image);

auto fusion = graph.op<dn::ExposureFusionOp>(
    "fusion", dn::ExposureFusionOpConfig{.match_window = std::chrono::milliseconds(40)});
graph.add_flow(camera->short_exposure, fusion->bracket0);
graph.add_flow(camera->mid_exposure, fusion->bracket1);
graph.add_flow(camera->long_exposure, fusion->bracket2);
```

The outputs are in device memory, so bind their GPU when compiling:

```cpp
holoscan::CompileOptions options;
options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
    .operator_path = "denoise", .output_port = "denoised", .device = holoscan::DeviceId{0}});
```

A lossless (`kBlocking`) edge whose queue is full fails the frame. Keep the camera at most one
frame ahead of the slowest consumer, as the demo's source does, or connect consumers that only need
the newest frame, such as a display, with `QueuePolicy::kDropOldest` and depth 1.

Without a graph, `Denoiser` (`<holoscan/denoise/denoiser.hpp>`), `ExposureFusion` and `ToneMapper`
(`<holoscan/denoise/exposure_fusion.hpp>`) run on `ImageView`s of device memory and a CUDA stream.
`<holoscan/denoise/synthetic_scene.hpp>` is the tests' and the demo's synthetic camera: a low-light
scene (dark gradients, coloured patches, stripes, a checkerboard and a moving disc, with a second
layout after a scene cut), clean on the host and with Poisson shot noise and Gaussian read noise of
known levels on the GPU; and an HDR scene (a striped shadow at 0.002 to 0.006, mid-tones, a striped
window at 1.5 to 6, about 13 stops) in radiance and as 8-bit brackets.

## Quality

All on scenes rendered at test time with Poisson shot noise (1 code value per photon) and Gaussian
read noise of 6 code values, about 7.5 code values in all: 29.0 dB PSNR. Gains on the still RGB8
320x240 scene, the cuda backend, after one frame for the spatial filters alone and 16 with the
temporal filter:

| Method | Strength 0.25 | Strength 0.5 | Strength 1 |
| --- | --- | --- | --- |
| Bilateral | +3.6 dB | +9.1 dB | +9.1 dB |
| Non-local means | +8.5 dB | +9.5 dB | +11.1 dB |
| Temporal | +11.5 dB | +11.9 dB | +11.9 dB (+14.7 at 32 frames, +16.8 at 64) |
| Temporal + bilateral | +17.0 dB | +21.2 dB | +23.8 dB |
| Temporal + non-local means | +21.6 dB | +20.4 dB | +22.0 dB |
| vpi (TNR on CUDA, x86_64, indoor low light) | +7.7 dB | +6.8 dB | +7.5 dB |
| vpi (TNR on the Orin's VIC) | +7.6 dB | +5.9 dB | +6.2 dB |

The scene is mostly flat patches, on which the bilateral filter excels; textured scenes gain less.
The Orin gives the cuda backend's figures to the second decimal.

| What | Result |
| --- | --- |
| Temporal convergence (strength 1) | +3.0, +6.0, +9.0, +11.9 dB at 2, 4, 8, 16 frames: 10 log10(n), the running mean; settled at 64 frames within 2x of the recursive filter's variance |
| No ghosting: a disc moving 6 px a frame | Mean squared error where the disc was in the last 4 frames: 39.6 (temporal), 4.9 (with bilateral), 5.0 (with non-local means), 29.9 (vpi on CUDA), 40.0 (vpi on the VIC), against 71.0 in the noisy input and 1018 for a plain running average |
| Scene cut | The frame after the cut is bit for bit the frame a new sequence would give (cuda); 35.5 dB as against 35.5 dB as a first frame (vpi) |
| First frame | Temporal alone passes it unchanged; with a spatial filter it is that filter's alone |
| Formats | Temporal + bilateral at 0.5, 8 frames: MONO8 +14.9 dB, RGBA8 +18.6 dB (alpha unchanged), NV12 +14.9 dB; vpi +7.7, +6.8 and +4.8 dB |
| Noise estimate | 8.17 for read noise 8.00; 7.95 for 7.81 with shot noise |
| cuda against vpi | Both denoise the still scene (40.9 and 35.8 dB from 29.0), with the same channel means within 0.04 code values; neither trails |
| Determinism | The cuda backend gives the same bytes on every run, every method; so does VPI's TNR |
| Strength 0 | Every method leaves every frame unchanged |
| Odd and tiny sizes (1x1 to 33x17) | Flat frames stay flat to the last row and column, both backends |

Exposure fusion of the HDR scene's brackets at 1/8, 1 and 32 times the exposure (640x360):

| | Shadow texture / crushed samples | Window texture / clipped samples | Structure against the tone-mapped truth, shadow / window |
| --- | --- | --- | --- |
| Bracket x1/8 | 1.9 / 87984 | 95.0 / 0 | 0.00 / 1.00 |
| Bracket x1 | 11.0 / 0 | 0.0 / 178104 | 1.00 / 0.00 |
| Bracket x32 | 46.9 / 0 | 0.0 / 178104 | 1.00 / 0.00 |
| Fused, 3 brackets | 46.4 / 0 | 95.0 / 0 | 0.99 / 1.00 |

Texture is the mean step in code values between neighbouring stripes, away from the regions' edges;
structure the mean normalized cross-correlation, over 16x16 blocks, with Reinhard's tone map of the
radiance. Fusion is not a global tone curve: it lifts the shadow and flattens the mid-tones' gentle
ramp, so its PSNR against the tone-mapped truth is low (12.0 dB) though it keeps the detail every
bracket loses somewhere. Two brackets (x1/8 and x32) keep it too. The GPU tone map matches the
double-precision reference within one code value for float and 16-bit (10, 12, 16 bits) input, with
its log-average within 1e-5.

## Performance

Mean time per frame, from CUDA events around the call, RGB8 in and out (the vpi backend's time
includes its conversions and its waits for VPI); `denoise_benchmark` measures it, over 100 frames on
the x86_64 machine and 300 on the Orin, the better of two runs. The x86_64 machine has an RTX 5090,
shared with other work during the runs; the Orin is a Jetson AGX Orin developer kit (64 GB) with
JetPack 7.2 in MAXN, also running another CI job.

| Method | 1920x1080, RTX 5090 | 3840x2160, RTX 5090 | 1920x1080, Orin | 3840x2160, Orin |
| --- | --- | --- | --- | --- |
| Bilateral | 0.076 ms | 0.294 ms | 1.62 ms | 6.39 ms |
| Non-local means | 0.145 ms | 0.565 ms | 2.51 ms | 9.97 ms |
| Temporal | 0.072 ms | 0.362 ms | 1.26 ms | 4.96 ms |
| Temporal + bilateral | 0.117 ms | 0.521 ms | 2.43 ms | 9.73 ms |
| Temporal + bilateral, NV12 | 0.094 ms | 0.357 ms | 2.07 ms | 8.19 ms |
| Temporal + non-local means | 0.192 ms | 0.848 ms | 3.36 ms | 13.34 ms |
| vpi, TNR on CUDA | 0.103 ms | 0.302 ms | 1.81 ms | 6.54 ms |
| vpi, TNR on CUDA, NV12 | 0.102 ms | 0.281 ms | 1.62 ms | 5.96 ms |
| vpi, TNR on the VIC | | | 3.09 ms | 13.63 ms |
| vpi, TNR on the VIC, NV12 | | | 4.64 ms | 13.51 ms |
| Exposure fusion, 3 brackets | 0.352 ms | 1.854 ms | 6.83 ms | 25.95 ms |
| Tone map, RGB32F | 0.135 ms | 0.549 ms | 2.61 ms | 10.30 ms |

On the VIC the time is latency, not GPU load: the GPU only converts to and from NV12, and the VIC's
own time varied from run to run (3.1 to 5.3 ms at 1920x1080, 9.4 to 21 ms at 3840x2160) while the
other job ran. Where the GPU is busy with inference, the VIC keeps denoising off it; where latency
matters most, the cuda backend is faster on the Orin too.

## Requirements

- Holoscan SDK 5.0 (Core and Sensor I/O; Testing for the tests), CUDA 13, CMake 4.0 or later and a
  C++20 compiler.
- Optional: VPI 4 (`libnvvpi4`, `vpi4-dev`), for the vpi backend; found in `/opt/nvidia/vpi4`.
  Without it the module builds without that backend.

The module downloads nothing and needs no models or data.

## Build, test and install

From the repository root, in the repository's CI image or with the requirements above:

```bash
cmake -S modules/denoise -B build/denoise -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build/denoise
ctest --test-dir build/denoise --output-on-failure
cmake --install build/denoise --prefix "$PWD/install"
```

| CMake option | Default | |
| --- | --- | --- |
| `BUILD_TESTING` | `OFF` | Build the tests, when the module is the top-level project |
| `HOLOSCAN_DENOISE_BUILD_TESTING` | `OFF` | Build the tests when another project includes the module (`add_subdirectory()`, `FetchContent`); its `BUILD_TESTING` and `include(CTest)` do not |
| `HOLOSCAN_DENOISE_WITH_VPI` | `ON` | Build the vpi backend when VPI 4 is installed |
| `CMAKE_CUDA_ARCHITECTURES` | `native` (`all-major` for the package) | GPUs to compile for |

The package sets `holoscan-denoise_WITH_VPI` to whether the vpi backend was built.

The tests, all on scenes rendered at test time:

| Test | Needs | Checks |
| --- | --- | --- |
| `denoise_denoiser` | GPU | `Denoiser`: PSNR gain of every method and strength, strength 0 as the identity, temporal convergence, no ghosting behind a moving disc, the first frame, a scene cut and `reset()`, MONO8, RGBA8 and NV12, odd and 1x1 sizes, the noise estimate, determinism, rejected settings and views; with VPI, the same for the vpi backend and cuda against vpi |
| `denoise_fusion` | GPU | `ExposureFusion` against the tone-mapped truth for RGB and gray brackets, 3 and 2 of them, against each single bracket; the weights; sizes from 1x1 and determinism; `ToneMapper` against its double-precision reference for each float and 16-bit layout, key, white and curve; rejections |
| `denoise_operators` | GPU | `DenoiseOp` through the SDK's operator test harness, byte for byte against `Denoiser`: MONO8, RGB8, RGBA8 and NV12 from device and host memory, non-local means, VPI, the host copy logged once, a change of size, the auto backend, the header, capture time and frame id; `ExposureFusionOp` against `ExposureFusion` with 3 and 2 brackets matched exactly and within a window, unmatched brackets never fused, tone mapping float and 16-bit frames; rejected inputs and settings |
| `denoise_benchmark` | GPU | Runs the benchmark briefly, so that it keeps working |
| `denoise_params` | Nothing | SDK-free (`tests/host/`, also run by the hosted CI job): names, validation, the auto backend's choice, the strength's mapping, pyramid levels, the tone map and fusion weight references |

[`tests/installed_consumer/`](tests/installed_consumer/) is a separate project that uses the
installed package: a graph from host-memory frames through `DenoiseOp` and `ExposureFusionOp`,
checking the PSNR gain and the fused shadow and window. Build it with the install prefix on
`CMAKE_PREFIX_PATH` and run `ctest` there.

## Debian package

Inside the repository's CI image, with the module's folder as the Holoscan CLI project root:

```bash
holoscan --project-root modules/denoise package holoscan-denoise --language cpp \
  --pkg-generator DEB --local
```

This builds for every major GPU architecture and writes `holoscan-denoise_1.0.0_<arch>.deb` to the
current directory. It installs to `/usr` and depends on `holoscan (>= 5.0.0)`, and with the vpi
backend on `libnvvpi4`, so `apt install ./holoscan-denoise_1.0.0_<arch>.deb` needs NVIDIA's CUDA
repository (and on x86_64 for VPI, NVIDIA's Jetson repository).

## Licenses

Apache-2.0. The optional vpi backend links NVIDIA VPI 4, which is distributed under NVIDIA's own
license and not included. See [NOTICE](NOTICE).
