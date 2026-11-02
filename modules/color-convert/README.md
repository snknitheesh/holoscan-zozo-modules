# holoscan-color-convert

A Holoscan SDK 5 Module that converts camera images between pixel formats and resizes them, in one
pass on the GPU, or on the VIC of Jetson Orin and Thor. `ColorConvertOp` changes an image's
format; `ResizeOp` resizes it, with an optional crop and letterbox, and can convert it in the same
pass. Both read host or CUDA device images and write CUDA device images. The
[Color Convert demo](../../applications/color_convert/) runs both end to end.

| | |
| --- | --- |
| CMake package | `find_package(holoscan-color-convert)` |
| Target | `holoscan::color_convert` |
| Headers | `<holoscan/color_convert/...>` |
| C++ namespace | `holoscan::color_convert` |
| Debian package | `holoscan-color-convert` |
| Platforms | x86_64; aarch64 (Jetson Thor and Orin, JetPack 7) |

## What the SDK already does, and what this module adds

The Holoscan SDK 5.0 has two image operators of its own:

- **Holoscan Vision** (`holoscan::vision`, CV-CUDA; in the SDK's full container, but not in this
  repository's CI image) has `ResizeOp` (nearest, linear, cubic, area), `CustomCropOp`,
  `CopyMakeBorderOp`, `ConvertToOp` (element type), `ReformatOp` (HWC and CHW) and
  `BayerDemosaicOp`. They take interleaved 1-, 3- and 4-channel CUDA images and refuse packed and
  chroma-subsampled ones: no YUV, and no colour conversion at all.
- **`TensorPreprocessorOp`** (`holoscan::ops`, NPP), the preprocessing for inference, turns RGB,
  YUYV, NV12 or I420 into RGB (`uint8`, or normalized `float32` in NCHW or NHWC) at one fixed size
  with bilinear resizing. It has the BT.601 and BT.709 matrices, applies limited range only to
  NV12, and takes device input only.

So for interleaved RGB images on the GPU, Holoscan Vision's `ResizeOp`, `CustomCropOp` and
`CopyMakeBorderOp` already resize, crop and pad, in three passes. This module adds what neither
has:

- **YUV formats**, in and out: NV12, NV16, I420, YUYV and UYVY, besides RGB8, BGR8, RGBA8, BGRA8,
  MONO8, MONO16 and RGB16, with odd sizes and chroma at the edges handled as the SDK's image schema
  defines them.
- **Every colour matrix and range**: BT.601, BT.709 and BT.2020, full or limited, in both
  directions, taken from the image's colour space when it names one.
- **The VIC**, through VPI 4, for the conversions it can do, so they leave the GPU free.
- **One fused pass**: convert, crop, resize (nearest, bilinear or area) and letterbox at once, on
  any format, including resizing a YUV image without converting it.
- **Host input**, staged once per frame, with a one-time log that says so.

## Operators

### `ColorConvertOp`

| Port | Type | |
| --- | --- | --- |
| `image` (input) | `holoscan::schema::ImageT` | Any format below, in host or CUDA device memory |
| `converted` (output) | `holoscan::schema::ImageT` | The same image in `output_format`, in CUDA device memory |

### `ResizeOp`

| Port | Type | |
| --- | --- | --- |
| `image` (input) | `holoscan::schema::ImageT` | Any format below, in host or CUDA device memory |
| `resized` (output) | `holoscan::schema::ImageT` | `width` x `height`, in `output_format` or the input's format, in CUDA device memory |

`ColorConvertConfig` and `ResizeConfig` share their colour settings:

| Field | Default | |
| --- | --- | --- |
| `output_format` | `kRgb8`; for `ResizeOp`, none (keep the input's) | The format to write |
| `matrix` | `kBt601` | `kBt601`, `kBt709` or `kBt2020` (non-constant luminance) |
| `range` | `kLimited` | `kLimited` (Y 16-235, Cb and Cr 16-240) or `kFull` (0-255), for YUV formats |
| `use_input_color_space` | `true` | A YUV image whose colour space is `REC601`, `REC709` or `REC2020` is converted with that matrix instead |
| `backend` | `kCuda` | `kCuda`, `kVic` or `kAuto`; see [Backends](#backends) |
| `custom_format` | none | What `CUSTOM` images hold: `kNv16`, which the schema has no encoding for |
| `on_backend` | none | Called with the backend of the first frame, and again whenever `auto` changes its choice |

`ColorConvertConfig` also has `max_width` and `max_height` (3840 x 2160): the largest image, which
sizes the output's memory pool. `ResizeConfig` has `width` and `height` (the output size), `mode`
(`kNearest`, `kBilinear` or `kArea`), `crop` (a `Rect` of the input to read), `letterbox` and
`fill_rgb` (keep the aspect ratio and fill the rest with this colour).

The output's descriptor keeps the input's header, exposure and gain (and, for `ColorConvertOp`, its
ROI offsets), names the matrix's colour space (`REC601`, `REC709`, `REC2020`) for a YUV output,
and keeps an RGB input's colour space for an RGB output; the sample keeps the input's capture time
and frame id.

### Formats

| Format | Encoding | Tensor |
| --- | --- | --- |
| NV12, I420 | `NV12`, `I420` | `uint8` bytes with `plane_layouts`; chroma at half width and height, rounded up |
| NV16 | `CUSTOM` with `custom_format = kNv16` | `uint8` bytes with two `plane_layouts`; chroma at half width, rounded up |
| YUYV, UYVY | `YUYV`, `UYVY` | `uint8` `[H, W, 2]`; even width |
| RGB8, BGR8, RGBA8, BGRA8 | as named | `uint8` `[H, W, 3]` or `[H, W, 4]` |
| MONO8, MONO16 | `MONO8`, `MONO16` | `uint8` or `uint16` `[H, W]` or `[H, W, 1]`; full-range luma |
| RGB16 | `RGB16` | `uint16` `[H, W, 3]` |

Rows may be padded on input; outputs are packed, with their planes one after the other.

### What a conversion computes

Every backend follows one definition, which `reference_convert()` in
`<holoscan/color_convert/reference.hpp>` implements on the CPU:

1. Each source pixel is read in its own samples: Y, Cb, Cr with the chroma of the sample covering
   it (YUV is not interpolated in chroma); R, G, B; or luma. 16-bit samples are divided by 257.
2. Each destination pixel inside the content rectangle samples the source rectangle, with pixel
   centers at +0.5: **nearest** takes the pixel under its footprint's center, **bilinear**
   interpolates the four around it (clamped to the rectangle), **area** averages every pixel the
   footprint covers by how much it covers. Outside the content (letterbox), it takes the fill.
3. The samples go to the destination's colour model through the matrix and range (nothing is
   converted between two YUV or two RGB formats), are rounded half up and clamped. A subsampled
   destination stores the mean chroma of the pixels that share a chroma sample.

So converting between formats of one colour model is exact: RGB8 to BGRA8 and back, or NV12 to
I420 and back, give the same bytes. Letterboxing scales the source by the smaller ratio, centers
it, and makes its offset and size even where a subsampled destination has a border, so no chroma
sample mixes picture and fill.

## Backends

| Backend | Runs on | What it does |
| --- | --- | --- |
| `Backend::kCuda` | GPU | Everything: one CUDA kernel, one thread per pixel (per chroma sample for YUV output), queued on the operator's CUDA stream. The default. |
| `Backend::kVic` | VIC (Jetson) | NV12, YUYV, UYVY, RGBA8, BGRA8 and MONO8, nearest or bilinear, sides of 16 to 16384 and 1/16x to 16x, even offsets and sizes in chroma-subsampled images; between YUV and RGB only BT.709, and no MONO8 made from colour. Fails at construction where VPI reports no VIC, and on the first frame it cannot convert. |
| `Backend::kAuto` | VIC or GPU | Chosen on the first frame (and again when the image changes): the VIC where VPI reports one and it can do the conversion, CUDA otherwise. Opt-in. |

With `kAuto`, the operator logs its choice and why, for example
`ColorConvertOp: backend auto chose vic for yuyv 1920x1080 to rgba8 1920x1080: VPI reports a VIC,
which can do this conversion`, and tells `on_backend`. `resolve_auto()` and `vic_limitation()` in
`<holoscan/color_convert/backend.hpp>` are the rule and the VIC's limits, without a GPU.

The VIC reads and writes the images in place: VPI wraps the CUDA device memory, a crop or a
letterbox's content as a sub-image at an offset; the CUDA kernel fills a letterbox's border at the
same time. It runs on its own VPI stream, so the operator waits for its CUDA stream before the VIC
starts and for the VIC before it returns: a VIC conversion holds the operator's thread.

Between colour models, the VIC gets only what gives the same result on every Jetson the CI tests:
Thor's VIC applies BT.709 where BT.601 is asked for, VPI's BT.2020 also converts BT.2020 to BT.709
primaries, and luma the VIC computes from colour (MONO8) differs on Thor by tens of codes. So a
conversion between YUV and RGB takes the VIC only with BT.709, MONO8 made from colour stays on
CUDA, and `auto` sends the rest to CUDA. Within a colour model no matrix applies, and the VIC takes
any.

Every backend reads device memory. A host image is copied to the GPU once per frame, and the
operator says so once, at the first such frame:

```text
ColorConvertOp (cuda backend): the images are in host memory and the backend reads CUDA device memory, so each frame is copied host to device, 3110400 bytes (3.1 MB) per frame; to avoid the copy, send CUDA device images
```

## Accuracy

- The CUDA kernel and the CPU reference share their arithmetic, without floating-point
  contraction on either side, so the kernel's output equals the reference's byte for byte, for
  every format pair, matrix, range, mode, crop and letterbox (`color_convert_converter`).
- The reference is checked against the colour science in double precision: every pair of the 12
  formats, for every matrix and range, within 1 code (2 for 16-bit outputs), and exact between
  formats of one colour model; round trips within 2 codes; the 100% colour bars as published
  (BT.601 and BT.709 limited range, JPEG's BT.601 full range), exactly but for two samples that lie
  exactly half way between codes (full-range yellow's Cb and cyan's Cr, 0.5), which round up;
  resizing within 1 code of a double-precision resize.
- The VIC against the CUDA kernel, on Jetson Orin, over 432 conversions of a smooth picture (every
  conversion the VIC takes, in both ranges, same size and resized, nearest and bilinear, crop and
  letterbox): mean difference at most 0.9 codes, 99th percentile at most 3, largest 6. The VIC
  interpolates chroma where the kernel replicates it, so at hard colour edges they differ more:
  on the demo's colour bars, mean 0.7, 99th percentile 2, largest 181.

## Performance

The time one conversion takes, from submission until its result is in device memory: CUDA timed
with events over 200 calls queued back to back, the VIC on the host over 200 calls, each the best
of three rounds. The input is the synthetic camera's picture, BT.709, limited range.
x86_64: GeForce RTX 5090, driver 595.91. Orin: Jetson AGX Orin Developer Kit (64 GB), JetPack 7.2 (R39.2), MAXN,
VIC under its default clock governor. SDK 5.0.0, CUDA 13.2, VPI 4.1.4. Milliseconds; `n/a` is
a conversion the VIC cannot do.

| Input | Output | Operation | x86_64 cuda | Orin cuda | Orin vic |
| --- | --- | --- | ---: | ---: | ---: |
| NV12 1920x1080 | RGB8 1920x1080 | convert | 0.016 | 0.319 | n/a |
| NV12 1920x1080 | RGBA8 1920x1080 | convert | 0.016 | 0.365 | 2.684 |
| NV12 1920x1080 | MONO8 1920x1080 | convert | 0.015 | 0.267 | n/a |
| NV16 1920x1080 | RGB8 1920x1080 | convert | 0.016 | 0.463 | n/a |
| I420 1920x1080 | RGB8 1920x1080 | convert | 0.016 | 0.286 | n/a |
| YUYV 1920x1080 | RGB8 1920x1080 | convert | 0.016 | 0.287 | n/a |
| YUYV 1920x1080 | RGBA8 1920x1080 | convert | 0.016 | 0.299 | 4.946 |
| UYVY 1920x1080 | RGBA8 1920x1080 | convert | 0.016 | 0.388 | 8.566 |
| YUYV 1920x1080 | NV12 1920x1080 | convert | 0.011 | 0.247 | 4.191 |
| RGB8 1920x1080 | NV12 1920x1080 | convert | 0.011 | 0.219 | n/a |
| RGBA8 1920x1080 | NV12 1920x1080 | convert | 0.011 | 0.241 | 4.267 |
| RGB8 1920x1080 | BGR8 1920x1080 | convert | 0.015 | 0.283 | n/a |
| RGB16 1920x1080 | RGB8 1920x1080 | convert | 0.020 | 0.367 | n/a |
| NV12 1920x1080 | RGBA8 1280x720 | bilinear | 0.013 | 0.259 | 3.452 |
| NV12 1920x1080 | RGB8 640x640 | bilinear, letterbox | 0.004 | 0.049 | n/a |
| YUYV 1920x1080 | RGBA8 640x360 | nearest | 0.003 | 0.035 | 5.255 |
| NV12 1920x1080 | RGB8 1280x720 | area | 0.016 | 0.307 | n/a |
| NV12 1920x1080 | NV12 1280x720 | bilinear | 0.010 | 0.187 | 3.875 |
| NV12 3840x2160 | RGB8 3840x2160 | convert | 0.059 | 1.104 | n/a |
| NV12 3840x2160 | RGBA8 3840x2160 | convert | 0.062 | 1.150 | 21.853 |
| NV12 3840x2160 | MONO8 3840x2160 | convert | 0.056 | 1.069 | n/a |
| NV16 3840x2160 | RGB8 3840x2160 | convert | 0.059 | 1.113 | n/a |
| I420 3840x2160 | RGB8 3840x2160 | convert | 0.060 | 1.122 | n/a |
| YUYV 3840x2160 | RGB8 3840x2160 | convert | 0.062 | 1.129 | n/a |
| YUYV 3840x2160 | RGBA8 3840x2160 | convert | 0.063 | 1.174 | 23.767 |
| UYVY 3840x2160 | RGBA8 3840x2160 | convert | 0.060 | 1.178 | 22.400 |
| YUYV 3840x2160 | NV12 3840x2160 | convert | 0.040 | 0.771 | 10.433 |
| RGB8 3840x2160 | NV12 3840x2160 | convert | 0.042 | 0.789 | n/a |
| RGBA8 3840x2160 | NV12 3840x2160 | convert | 0.042 | 0.793 | 21.121 |
| RGB8 3840x2160 | BGR8 3840x2160 | convert | 0.056 | 1.106 | n/a |
| RGB16 3840x2160 | RGB8 3840x2160 | convert | 0.077 | 1.438 | n/a |
| NV12 3840x2160 | RGBA8 1280x720 | bilinear | 0.009 | 0.163 | 11.892 |
| NV12 3840x2160 | RGB8 640x640 | bilinear, letterbox | 0.005 | 0.232 | n/a |
| YUYV 3840x2160 | RGBA8 640x360 | nearest | 0.003 | 0.041 | 12.761 |
| NV12 3840x2160 | RGB8 1280x720 | area | 0.024 | 0.478 | n/a |
| NV12 3840x2160 | NV12 1280x720 | bilinear | 0.007 | 0.117 | 10.985 |

On the Orin the CUDA kernel is faster than the VIC in every case measured: 7 times for NV12 to
RGBA8 at 1080p, and over 100 times for a small resize.
The VIC also holds the operator's thread while it works. The VIC is for a GPU that is busy with something else: it converts without
taking any of the GPU's time. Its numbers are for VPI's own scheduling with the VIC's default clock
governor, which left it at 282 MHz of 730 MHz between runs; a VIC clock locked high is faster. The
`color_convert_benchmark` tool in the tests prints this table on any machine (see
[DEVELOPER.md](DEVELOPER.md)).

## Use it in your graph

```cmake
find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS core sensor_io)
find_package(holoscan-color-convert 1.0 REQUIRED CONFIG)
target_link_libraries(my_app PRIVATE holoscan::color_convert)
```

```cpp
#include <holoscan/color_convert/color_convert_op.hpp>
#include <holoscan/color_convert/resize_op.hpp>

namespace cc = holoscan::color_convert;

// A 1080p NV12 camera, BT.709 limited range: RGBA for display, and a 640x640
// letterboxed RGB8 copy for a detector, converted in the same pass.
auto to_rgba = graph.op<cc::ColorConvertOp>(
    "to_rgba", cc::ColorConvertConfig{.output_format = cc::PixelFormat::kRgba8,
                                      .matrix = cc::ColorMatrix::kBt709,
                                      .max_width = 1920,
                                      .max_height = 1080});
auto to_model = graph.op<cc::ResizeOp>(
    "to_model", cc::ResizeConfig{.width = 640,
                                 .height = 640,
                                 .letterbox = true,
                                 .fill_rgb = {114, 114, 114},
                                 .output_format = cc::PixelFormat::kRgb8,
                                 .matrix = cc::ColorMatrix::kBt709,
                                 .backend = cc::Backend::kAuto});
graph.add_flow(camera->image, to_rgba->image);   // Output<holoscan::schema::ImageT>
graph.add_flow(camera->image, to_model->image);
graph.add_flow(to_model->resized, detector->image);
```

Bind the outputs (`to_rgba/converted`, `to_model/resized`) to a CUDA device in the
`CompileOptions`, as for any CUDA output. `holoscan-color-convert_WITH_VPI` tells whether the
installed module has the `vic` backend. Without a graph, `holoscan::color_convert::Converter` in
`<holoscan/color_convert/converter.hpp>` converts device images on a CUDA stream, and
`reference_convert()` converts host images on the CPU; `draw_test_pattern()` and
`launch_test_pattern()` draw the synthetic camera's picture, for tests.

## Requirements

- Holoscan SDK 5.0 (Core and Sensor I/O; Testing for the tests), CUDA 13, CMake 4.0 or later and a
  C++20 compiler.
- Optional: NVIDIA VPI 4 (`vpi4-dev`) for the `vic` backend. The repository's CI images install it
  from NVIDIA's Jetson package repository; see [`.github/docker/`](../../.github/docker/). On
  x86_64 it builds, and VPI reports no VIC, so `auto` takes CUDA there.

## Build, test and install

From the repository root, in the repository's CI image or with the requirements above:

```bash
cmake -S modules/color-convert -B build/color-convert -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON
cmake --build build/color-convert
ctest --test-dir build/color-convert --output-on-failure
cmake --install build/color-convert --prefix "$PWD/install"
```

| CMake option | Default | |
| --- | --- | --- |
| `HOLOSCAN_COLOR_CONVERT_WITH_VPI` | `ON` | Build the `vic` backend when VPI 4 is found |
| `BUILD_TESTING` | `OFF` | Build the tests, when the module is built on its own |
| `HOLOSCAN_COLOR_CONVERT_BUILD_TESTING` | `OFF` | Build the tests inside another project |
| `CMAKE_CUDA_ARCHITECTURES` | `native` (`all-major` for the package) | GPUs to compile for, unless already set |

Another project can also build the module as part of its own build, with `add_subdirectory()` or
`FetchContent`, and link `holoscan::color_convert` without `find_package()`; it then builds a
shared library and none of its tests unless `HOLOSCAN_COLOR_CONVERT_BUILD_TESTING` is `ON`.

The tests, every one on synthetic images made at test time:

| Test | Needs | Checks |
| --- | --- | --- |
| `color_convert_converter` | GPU | The CUDA kernel equals the reference byte for byte: every format pair for every matrix and range, every resize mode with crops, letterboxes and fused conversions, odd sizes, padded rows, the synthetic picture, 1080p camera frames; determinism across streams and runs |
| `color_convert_vic_parity` | Jetson | The VIC against the CUDA kernel within tolerance (above), and `auto`'s choice; skipped elsewhere, and fails on a Jetson CI runner without a VIC |
| `color_convert_operators` | GPU | Both operators through the SDK's operator test harness: host and device input, packed and planar formats, NV16 as `CUSTOM`, the descriptors (encoding, plane layouts, colour space, header, ROI) and provenance, the input's colour space, resizing with crop and letterbox, and bad input and configuration |
| `color_convert_conversion_math` | Nothing | The reference against the colour science: every format pair, round trips, colour bars, odd sizes and chroma edges, resizing per mode, letterbox geometry, padded rows, determinism; also built on its own from `tests/host/` by the hosted CI job |
| `color_convert_backend_choice` | Nothing | `auto`'s choice and the VIC's limits; also built from `tests/host/` |

[`tests/installed_consumer/`](tests/installed_consumer/) is a separate project that uses the
installed package: a graph from host NV12 through `ColorConvertOp` and a letterboxing,
converting `ResizeOp`, every byte checked.

## Debian package

Inside the repository's CI image, with the module's folder as the Holoscan CLI project root:

```bash
holoscan --project-root modules/color-convert package holoscan-color-convert --language cpp \
  --pkg-generator DEB --local
```

This builds for every major GPU architecture and writes
`holoscan-color-convert_1.0.0_<arch>.deb` to the current directory. It installs to `/usr` and
depends on `holoscan (>= 5.0.0)` and, when built with VPI, `libnvvpi4`.

## Licenses

Apache-2.0. The module links the Holoscan SDK (Apache-2.0) and, for the `vic` backend, NVIDIA VPI
4, under NVIDIA's own terms, installed as `/opt/nvidia/vpi4/doc/VPI_EULA.txt`; neither is
redistributed. See [NOTICE](NOTICE).
