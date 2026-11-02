# holoscan-image-quality

A Holoscan SDK 5 Module that checks every frame of a camera stream, cheaply and on the GPU, for what
makes downstream perception untrustworthy: blur, bad exposure, noise, a frozen stream, a smudged or
blocked lens, and a color cast. `ImageQualityOp` turns each image into a small metrics tensor and a
flags bitfield, so a pipeline can drop or annotate a frame, and optionally a per-tile map and a
HoloViz overlay that show where in the image the problem is. The
[Image Quality demo](../../applications/image_quality/) runs it end to end.

| | |
| --- | --- |
| CMake package | `find_package(holoscan-image-quality)` |
| Targets | `holoscan::image_quality` |
| Headers | `<holoscan/image_quality/...>` |
| C++ namespace | `holoscan::image_quality` |
| Debian package | `holoscan-image-quality` |
| Platforms | x86_64; aarch64 (Jetson Thor with JetPack 7, Jetson Orin with JetPack 7.2) |

## `ImageQualityOp`

| Port | Type | |
| --- | --- | --- |
| `image` (input) | `holoscan::schema::ImageT` | `MONO8`, `RGB8`, `BGR8`, `RGBA8` or `BGRA8`, `[H, W]` or `[H, W, C]`, pixels packed within each row (rows may be padded), in host or CUDA device memory |
| `metrics` (output) | `holoscan::Tensor` | float32 `[22]`, in `Metric` order, in host memory |
| `flags` (output) | `holoscan::Tensor` | uint32 `[1]`, the `Flag` bits, in host memory; zero passes |
| `tiles` (output, with `tiles_output`) | `holoscan::Tensor` | float32 `[tiles_y, tiles_x, 7]`, in `TileChannel` order, in host memory |
| `overlay` (output, with `overlay_output`) | `holoscan::schema::ImageT` | `RGBA8` `[tiles_y * cell, tiles_x * cell, 4]` in host memory: obstructed tiles magenta, blurred tiles red, low-texture tiles faint yellow, the rest clear; HoloViz stretches it over the image |

Every output carries the image's capture time and frame id (an unstamped image's outputs are stamped
with the activation time), so `OnAligned` pairs them downstream. The optional ports exist only when
configured. Images in host memory are copied to the GPU, which the operator logs once. Failed frames
are logged as warnings, at most once per `log_interval` (5 s; zero turns it off), with the flags they
had.

Each frame takes four kernel launches and one copy back of about 30 kB, and the operator waits for
them, so its outputs are on the host when `compute()` returns. All sums are integer and the
histograms integer atomics, so the results do not depend on the GPU or the run: they are the same
bit for bit as the CPU reference's (`<holoscan/image_quality/reference.hpp>`), which the tests check.
VPI has a Laplacian and histograms, but each would be a separate pass over the image; the fused
kernels read the image once and the luma twice, so the module is CUDA only.

### Metrics

All on 8-bit luma, `(77 R + 150 G + 29 B + 128) >> 8` (BT.601), except the color means. Stencil
metrics use the pixels whose 3x3 neighborhood is inside the image.

| Metric | |
| --- | --- |
| `mean_luma`, `luma_p1`, `luma_p5`, `luma_p50`, `luma_p95`, `luma_p99` | Mean and nearest-rank percentiles of luma |
| `underexposed_fraction`, `overexposed_fraction` | Pixels with luma `<= dark_level` (16), `>= bright_level` (239) |
| `dynamic_range` | `luma_p99 - luma_p1` |
| `laplacian_variance` | Variance of the 4-neighbor Laplacian |
| `tenengrad` | Mean squared Sobel gradient, the Sobel scaled by 1/8 so a ramp of slope 1 gives 1 |
| `focus` | Median over the textured tiles of their focus (below) |
| `blurred_tile_fraction`, `textured_tile_fraction`, `obstructed_tile_fraction` | Tiles, out of all, that are blurred, textured, obstructed |
| `noise_sigma` | Noise standard deviation in luma levels: the median absolute value of the high-pass `[1 -2 1; -2 4 -2; 1 -2 1]` over the flat pixels (the `flat_fraction`, 50 %, with the smallest Sobel gradient), over 0.6745 and 6 |
| `frame_difference` | Mean absolute luma difference to the previous frame; NaN for the first frame or after a size change |
| `frozen_frames` | Consecutive frames frozen so far |
| `mean_red`, `mean_green`, `mean_blue` | Means of the color channels (each luma for `MONO8`) |
| `color_cast` | Gray-world deviation: the largest absolute value of `channel mean / gray - 1`, gray the mean of the three; 0 for `MONO8` |

Per tile (16 x 9 by default, fewer when the image has fewer pixels): mean luma, Laplacian variance,
Tenengrad, and:

- **texture**: the RMS Sobel gradient, with the noise's share taken out, over `mean luma + 1`: how
  much structure the tile shows, whatever its exposure;
- **focus**: Laplacian variance over Tenengrad, both with the noise's share taken out (the estimated
  noise, and at least 8-bit rounding's 1/12): about `1 / (2 sigma^2)` for an edge blurred by a
  Gaussian of `sigma` pixels, whatever the edge's contrast. The synthetic scene gives 4.6 sharp,
  0.56 at sigma 1, 0.17 at sigma 2 and 0.086 at sigma 3. A tile is judged only when it is textured
  (texture `>= min_texture`, 0.01, and more gradient energy than the noise's);
- **obstruction**: the fraction of the last `obstruction_frames` frames in which the tile was not
  textured.

### Flags

| Flag | Bit | Set when |
| --- | --- | --- |
| `kBlurred` | 0 | `focus < blur_focus` (0.2) |
| `kLocalBlur` | 1 | Not `kBlurred`, but a textured tile's focus, plus two standard errors of the noise's share, is below `blur_focus` |
| `kUnderexposed` | 2 | `mean_luma < min_mean_luma` (50) or `underexposed_fraction > max_underexposed_fraction` (0.25) |
| `kOverexposed` | 3 | `mean_luma > max_mean_luma` (205) or `overexposed_fraction > max_overexposed_fraction` (0.25) |
| `kNoisy` | 4 | `noise_sigma > max_noise_sigma` (5 luma levels) |
| `kFrozen` | 5 | The frame's bytes hash to the previous frame's (a sum of a 64-bit mix of each pixel's position and bytes), or `frame_difference <= frozen_difference` (0.05), for `frozen_frames` (1) frames in a row |
| `kObstructed` | 6 | A tile was not textured in at least `obstruction_fraction` (all) of the last `obstruction_frames` (30) frames, counted once that many frames have been seen: a smudge or a blocked lens shows after 30 frames, not before |
| `kColorCast` | 7 | `color_cast > max_color_cast` (0.15), color images only |

Every threshold is a field of `QualityThresholds`; the defaults are in parentheses.

Limits worth knowing: a scene that is genuinely flat in places (sky, a blank wall) makes those tiles
low-texture, so a static camera on it flags `kObstructed` after `obstruction_frames`; a static
noiseless scene looks frozen; a scene dominated by one color looks cast; strong noise hides blur in
a tile (hence the standard error) and the noise estimate also counts fine texture that survives the
flat-pixel selection. Noise is estimated on luma: independent noise of `s` on each of R, G and B is
`0.668 s` on luma.

## Use it in your graph

```cmake
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core sensor_io)
find_package(holoscan-image-quality 1.0 REQUIRED CONFIG)
target_link_libraries(my_app PRIVATE holoscan::image_quality)
```

```cpp
#include <holoscan/image_quality/image_quality_op.hpp>

namespace iq = holoscan::image_quality;

iq::ImageQualityConfig config;
config.thresholds.max_noise_sigma = 4.0;  // Any QualityThresholds field.
config.overlay_output = true;             // For a HoloViz image layer.
auto quality = graph.op<iq::ImageQualityOp>("quality", config);
graph.add_flow(camera->image, quality->image);       // Output<holoscan::schema::ImageT>
graph.add_flow(quality->metrics, gate->metrics);     // Input<holoscan::Tensor>, float32 [22]
graph.add_flow(quality->flags, gate->flags);         // Input<holoscan::Tensor>, uint32 [1]
graph.add_flow(quality->overlay, holoviz->image_layer_input(1));
```

A consumer reads `metrics` as `float[iq::kMetricCount]` indexed by `iq::Metric`, and `flags` as one
`uint32_t` of `iq::Flag` bits; `iq::flag_names()` and `iq::metric_name()` print them.

The checks need no graph: `iq::ImageQualityAnalyzer` (`<holoscan/image_quality/analyzer.hpp>`) takes
an image in host or device memory and a CUDA stream and returns a `QualityReport`;
`iq::ReferenceStats` (`reference.hpp`) and `iq::QualityEvaluator` (`quality.hpp`) do the same on the
CPU. `<holoscan/image_quality/synthetic.hpp>` renders the moving test scene and applies the
degradations the tests and the demo use: Gaussian blur (of an area or all), exposure gain, Gaussian
noise, a hazy smudge and channel gains.

## Accuracy

From the tests, on the synthetic scene (cells of 12 pixels, values 40..215), 1920x1080:

| Degradation | Result |
| --- | --- |
| Gaussian blur, sigma 0, 0.5, 1, 1.5, 2, 3, 4 | Laplacian variance 837, 381, 50, 17, 8.9, 4.6, 3.4; Tenengrad 181, 145, 86, 58, 43, 25, 15; focus 4.6, 2.6, 0.56, 0.26, 0.17, 0.086, 0.063: all fall strictly. `kBlurred` from sigma 2 |
| Blur of one area | Exactly the tiles inside it blurred, none clear of it; `kLocalBlur` only |
| Noise, sigma 2, 4, 8, 16 on luma | Estimated 2.03, 4.02, 8.04, 16.06 (within 2 %; the tests allow 20 %) |
| Noise, sigma 2, 4, 8, 16 on each of R, G, B | Estimated 1.38, 2.70, 5.35, 10.70 against 1.34, 2.67, 5.35, 10.69 on luma |
| Exposure x0.25, x2 | `kUnderexposed` only, `kOverexposed` only |
| Repeated frame; 10 pixels off by one | `kFrozen`, `frame_difference` 0; `kFrozen` by the difference |
| Static smudge over the moving scene | `kObstructed` on frame `obstruction_frames` of the smudge and not before, on the tiles under it, gone the frame after it goes |
| Red x1.3, blue x0.75 | `kColorCast` only (0.27; clean scenes stay under 0.03) |
| Clean moving scenes, every format, 333x211 to 1280x720, 35 frames | No flag |

## Performance

Per frame, as `ImageQualityOp` spends it (the kernels, the copy back, the wait and the evaluation),
RGB8, from the `image_quality_benchmark` test, median of 100 frames:

| | 1920x1080, device input | 1920x1080, host input | 3840x2160, device input | 3840x2160, host input |
| --- | --- | --- | --- | --- |
| RTX 5090 (x86_64) | 0.055 ms | 0.32 ms | 0.105 ms | 1.1 ms |
| Jetson AGX Orin (JetPack 7.2) | 0.48 ms | 1.2 ms | 1.9 ms | 6.0 ms |

Under 2 % of a 30 fps frame at 1080p on the Orin, 6 % at 4K. Host input adds the copy to the GPU;
send device images where the camera can.

## Requirements

- Holoscan SDK 5.0 (Core and Sensor I/O; Testing for the tests), CUDA 13 (with its CUB), CMake 4.0
  or later and a C++20 compiler.

The module downloads nothing and needs no models or data.

## Build, test and install

From the repository root, in the repository's CI image or with the requirements above:

```bash
cmake -S modules/image-quality -B build/image-quality -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build/image-quality
ctest --test-dir build/image-quality --output-on-failure
cmake --install build/image-quality --prefix "$PWD/install"
```

| CMake option | Default | |
| --- | --- | --- |
| `BUILD_TESTING` | `OFF` | Build the tests, when the module is the top-level project |
| `HOLOSCAN_IMAGE_QUALITY_BUILD_TESTING` | `OFF` | Build the tests when another project includes the module (`add_subdirectory()`, `FetchContent`); its `BUILD_TESTING` and `include(CTest)` do not |
| `CMAKE_CUDA_ARCHITECTURES` | `native` (`all-major` for the package) | GPUs to compile for |

The tests, all on synthetic images made at test time:

| Test | Needs | Checks |
| --- | --- | --- |
| `image_quality_host` | Nothing | The CPU reference and the evaluator: the metrics against a direct floating-point computation on small images, and every accuracy case above, with clean scenes, tiny (1x1 to 17x1) and constant images, and determinism; threshold validation, row pitch, and the obstruction window with a fraction below one |
| `image_quality_analyzer` | GPU | Every accuracy case on the GPU, from host and from device memory; the statistics bit for bit equal to the CPU reference's for every format, sizes from 1x1 to 1921x1081, pitched rows and other tile grids; the same for any split of tiles into blocks |
| `image_quality_operator` | GPU | `ImageQualityOp` through the SDK's operator test harness: host and device images in three formats, metrics, flags, tiles and overlay equal to the reference's, provenance on every output, and the rejection of unusable encodings and settings |
| `image_quality_benchmark` | GPU | The timings above; fails only if a 1080p frame from device memory takes longer than 33 ms |

`image_quality_host` is also a standalone project, [`tests/host/`](tests/host/), which the hosted CI
job runs without the SDK or a GPU. [`tests/installed_consumer/`](tests/installed_consumer/) is a
separate project that uses the installed package: a small graph sends five frames, the last two
repeats, and checks that exactly those are frozen. Build it with the install prefix on
`CMAKE_PREFIX_PATH` and run `ctest` there.

## Debian package

Inside the repository's CI image, with the module's folder as the Holoscan CLI project root:

```bash
holoscan --project-root modules/image-quality package holoscan-image-quality --language cpp \
  --pkg-generator DEB --local
```

`--language cpp` selects the module rather than its package entry, which has the same name. This
builds for every major GPU architecture and writes `holoscan-image-quality_1.0.0_<arch>.deb` to the
current directory. It installs to `/usr` and depends on `holoscan (>= 5.0.0)`, so
`apt install ./holoscan-image-quality_1.0.0_<arch>.deb` needs NVIDIA's CUDA repository.

## Licenses

Apache-2.0. See [NOTICE](NOTICE).
